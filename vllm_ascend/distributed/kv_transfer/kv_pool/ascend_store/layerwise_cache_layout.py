from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import regex as re
from vllm.config import VllmConfig
from vllm.logger import logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    KVCacheTensor,
    UniformTypeKVCacheSpecs,
)

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend import (
    get_layerwise_protocol,
)
from vllm_ascend.utils import get_kv_cache_tensor_layers, vllm_version_is

_NUM_SHARED_BUFFERS = "layerwise_num_shared_buffers"
_PREFETCH_LAYERS = "layerwise_prefetch_layers"
_INDEPENDENT_LAYERS = "layerwise_independent_layers"
_DEFAULT_MAX_PREFETCH_LAYERS = 8
_INDEXER_CACHE_SUFFIX = ".indexer.k_cache"


def get_layerwise_physical_layer_index(layer_name: str, base_layers: int) -> int:
    match = re.search(
        r"(?:^|\.)mtp(?:\.layers)?\.(\d+)(?:\.|$)",
        layer_name,
    )
    if match:
        return base_layers + int(match.group(1))
    match = re.search(r"layers\.(\d+)", layer_name)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", layer_name)
    return int(match.group(1)) if match else 0


@dataclass(frozen=True)
class LayerwiseCacheLayout:
    num_shared_buffers: int
    num_prefetch_layers: int
    independent_layers: list[int]
    prefetch_layer_map: dict[int, int]
    storage_indices: list[list[int]]
    has_layer_reuse: bool


@dataclass(frozen=True)
class NamedKVCacheSpec:
    layer_name: str
    spec: KVCacheSpec


@dataclass(frozen=True)
class LayerwiseLayerCacheSpecs:
    main: NamedKVCacheSpec
    indexer: NamedKVCacheSpec | None = None
    extra_main_specs: tuple[NamedKVCacheSpec, ...] = ()


@dataclass(frozen=True)
class LayerwiseReuseLayout:
    layer_cache_specs: dict[int, LayerwiseLayerCacheSpecs]
    buffer_slots: tuple[tuple[int, ...], ...]
    prefetch_layer_map: dict[int, int]
    independent_layers: list[int]
    num_prefetch_layers: int
    has_layer_reuse: bool


@dataclass
class _PackedCacheLane:
    slot_id: int
    role: str
    spec: KVCacheSpec
    layer_names: list[str]


def get_layerwise_reuse_config(kv_transfer_config: Any) -> dict[str, Any] | None:
    """Return the extra config of the layerwise-reuse connector, if any.

    A connector opts into layerwise reuse when its backend carries a
    layerwise protocol and the protocol accepts the connector's extra
    config. Both checks resolve through the backend registry — the generic
    layer never names the protocol or the backend.
    """
    if kv_transfer_config is None:
        return None

    connector_name = getattr(kv_transfer_config, "kv_connector", None)
    root_extra_config = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    if connector_name in ("AscendStoreConnector", "MooncakeConnectorStoreV1"):
        connector_configs = [
            {
                "kv_connector": connector_name,
                "kv_connector_extra_config": root_extra_config,
            }
        ]
    elif connector_name == "MultiConnector":
        connector_configs = root_extra_config.get("connectors", [])
    else:
        return None

    for connector_config in connector_configs:
        if not isinstance(connector_config, dict):
            continue
        if connector_config.get("kv_connector") not in (
            "AscendStoreConnector",
            "MooncakeConnectorStoreV1",
        ):
            continue
        extra_config = connector_config.get("kv_connector_extra_config") or {}
        protocol = get_layerwise_protocol(str(extra_config.get("backend", "mooncake")))
        if protocol is None:
            continue
        layerwise_config = protocol.extract_layout_config(extra_config)
        if layerwise_config is not None:
            return layerwise_config
    return None


def _parse_int_config(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool")
    try:
        return int(value)
    except (TypeError, ValueError) as err:
        raise TypeError(f"{name} must be an integer, got {value!r}") from err


def build_layerwise_cache_layout(
    num_layers: int,
    extra_config: dict[str, Any] | None = None,
) -> LayerwiseCacheLayout:
    shared_buffers_value = extra_config.get(_NUM_SHARED_BUFFERS) if extra_config else None
    if shared_buffers_value is None:
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        num_shared_buffers = num_layers
    else:
        num_shared_buffers = _parse_int_config(shared_buffers_value, _NUM_SHARED_BUFFERS)
        if num_shared_buffers < 1:
            raise ValueError(f"{_NUM_SHARED_BUFFERS} must be at least 1")

    prefetch_value = extra_config.get(_PREFETCH_LAYERS) if extra_config else None
    if prefetch_value is None:
        num_prefetch_layers = min(num_shared_buffers, _DEFAULT_MAX_PREFETCH_LAYERS)
    else:
        num_prefetch_layers = _parse_int_config(prefetch_value, _PREFETCH_LAYERS)
        if num_prefetch_layers < 1:
            raise ValueError(f"{_PREFETCH_LAYERS} must be at least 1")

    independent_value = extra_config.get(_INDEPENDENT_LAYERS) if extra_config else None
    if independent_value is None:
        layer_indices = [0]
    elif isinstance(independent_value, str) and independent_value.strip().lower() == "all":
        layer_indices = list(range(num_layers))
    elif isinstance(independent_value, list):
        layer_indices = [_parse_int_config(index, _INDEPENDENT_LAYERS) for index in independent_value]
    else:
        raise TypeError(f"{_INDEPENDENT_LAYERS} must be a list of integers or 'all'")

    normalized_indices = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            layer_index += num_layers
        if layer_index < 0 or layer_index >= num_layers:
            raise ValueError(
                f"{_INDEPENDENT_LAYERS} contains out-of-range layer index "
                f"{layer_index}; valid range is [0, {num_layers - 1}]"
            )
        normalized_indices.add(layer_index)
    independent_layers = sorted(normalized_indices)

    independent_layer_set = set(independent_layers)
    reused_layers = [index for index in range(num_layers) if index not in independent_layer_set]
    has_layer_reuse = len(reused_layers) > num_shared_buffers
    prefetch_layer_map = {
        reused_layers[next_index]: reused_layers[next_index - num_shared_buffers]
        for next_index in range(num_shared_buffers, len(reused_layers))
    }
    storage_indices = [[layer] for layer in independent_layers]
    for slot in range(num_shared_buffers):
        members = list(range(slot, len(reused_layers), num_shared_buffers))
        if members:
            storage_indices.append([reused_layers[index] for index in members])

    return LayerwiseCacheLayout(
        num_shared_buffers=num_shared_buffers,
        num_prefetch_layers=num_prefetch_layers,
        independent_layers=independent_layers,
        prefetch_layer_map=prefetch_layer_map,
        storage_indices=storage_indices,
        has_layer_reuse=has_layer_reuse,
    )


def get_layerwise_kv_cache_specs(
    kv_cache_config: KVCacheConfig,
) -> dict[str, KVCacheSpec]:
    """Expand group specs into a cache spec for every logical layer."""
    layer_specs: dict[str, KVCacheSpec] = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if isinstance(group_spec, UniformTypeKVCacheSpecs):
                layer_specs[layer_name] = group_spec.kv_cache_specs[layer_name]
            else:
                layer_specs[layer_name] = group_spec
    return layer_specs


def build_layerwise_reuse_layout(
    layer_specs: dict[str, KVCacheSpec],
    base_layers: int,
    extra_config: dict[str, Any],
) -> LayerwiseReuseLayout:
    """Build reusable physical-layer slots by grouping layers on their main cache spec."""
    named_specs_by_layer: dict[int, list[NamedKVCacheSpec]] = {}
    for layer_name, layer_spec in layer_specs.items():
        physical_layer = get_layerwise_physical_layer_index(layer_name, base_layers)
        named_specs_by_layer.setdefault(physical_layer, []).append(NamedKVCacheSpec(layer_name, layer_spec))

    physical_layers = sorted(named_specs_by_layer)
    base_layout = build_layerwise_cache_layout(len(physical_layers), extra_config)
    independent_layers = [physical_layers[index] for index in base_layout.independent_layers]
    independent_layer_set = set(independent_layers)

    layer_cache_specs: dict[int, LayerwiseLayerCacheSpecs] = {}
    for physical_layer, named_specs in named_specs_by_layer.items():
        if len(named_specs) == 1:
            layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(main=named_specs[0])
            continue

        indexer_specs = [spec for spec in named_specs if spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        main_specs = [spec for spec in named_specs if not spec.layer_name.endswith(_INDEXER_CACHE_SUFFIX)]
        if len(main_specs) < 1:
            raise ValueError(
                f"Physical layer {physical_layer} has no main cache spec; "
                f"got {[spec.layer_name for spec in named_specs]}."
            )
        # Select '.attn' as main spec, rest as extra
        main_spec = next((s for s in main_specs if s.layer_name.endswith(".attn")), main_specs[0])
        extra_specs = tuple(s for s in main_specs if s is not main_spec)
        indexer_spec = indexer_specs[0] if indexer_specs else None
        layer_cache_specs[physical_layer] = LayerwiseLayerCacheSpecs(
            main=main_spec,
            indexer=indexer_spec,
            extra_main_specs=extra_specs,
        )

    signature_buckets: list[tuple[KVCacheSpec, list[int]]] = []
    for physical_layer in physical_layers:
        if physical_layer in independent_layer_set:
            continue
        # TODO(lf): Plan shared buffers independently for every cache spec.
        # Slots are grouped by main spec. Indexer specs are validated separately.
        signature = layer_cache_specs[physical_layer].main.spec
        for bucket_signature, bucket_layers in signature_buckets:
            if signature == bucket_signature:
                bucket_layers.append(physical_layer)
                break
        else:
            signature_buckets.append((signature, [physical_layer]))

    buffer_slots: list[tuple[int, ...]] = [(layer,) for layer in independent_layers]
    prefetch_layer_map: dict[int, int] = {}
    for _, bucket_layers in signature_buckets:
        num_shared_buffers = min(base_layout.num_shared_buffers, len(bucket_layers))
        for buffer_index in range(num_shared_buffers):
            layers_sharing_buffer = tuple(bucket_layers[buffer_index::num_shared_buffers])
            buffer_slots.append(layers_sharing_buffer)
            for owner_index in range(1, len(layers_sharing_buffer)):
                prefetch_layer_map[layers_sharing_buffer[owner_index]] = layers_sharing_buffer[owner_index - 1]

    if prefetch_layer_map:
        unsupported_specs = [
            named_spec
            for named_specs in named_specs_by_layer.values()
            for named_spec in named_specs
            if not isinstance(named_spec.spec, AttentionSpec)
        ]
        if unsupported_specs:
            named_spec = unsupported_specs[0]
            raise NotImplementedError(
                "Layerwise KV cache reuse supports attention cache specs only; "
                f"{named_spec.layer_name} uses {type(named_spec.spec).__name__}."
            )

    return LayerwiseReuseLayout(
        layer_cache_specs=layer_cache_specs,
        buffer_slots=tuple(buffer_slots),
        prefetch_layer_map=prefetch_layer_map,
        independent_layers=independent_layers,
        num_prefetch_layers=base_layout.num_prefetch_layers,
        has_layer_reuse=bool(prefetch_layer_map),
    )


def _dsv4_cache_role(layer_name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.\d+\.(.+)$", layer_name)
    if match is None:
        raise ValueError(f"DeepSeek-V4 cache layer has no model-layer role: {layer_name}.")
    return match.group(1)


def _apply_dsv4_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    layer_specs: dict[str, KVCacheSpec],
    reuse_layout: LayerwiseReuseLayout,
    base_layers: int,
) -> None:
    """Compact DSV4 tuple slots while preserving cross-group block-ID reuse.

    Each scheduler group needs separate tuple slots for its live components in
    a page-size bucket. Different groups may use the same tuple slots because
    their block IDs are disjoint, as in the existing DSV4 packed planner.
    """
    if kv_cache_config.num_blocks <= 0:
        return

    groups = kv_cache_config.kv_cache_groups
    if not groups or not isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs):
        raise ValueError("DeepSeek-V4 layerwise reuse requires packed uniform KV cache groups.")
    page_sizes = sorted({spec.page_size_bytes for spec in groups[0].kv_cache_spec.kv_cache_specs.values()})
    if not page_sizes:
        raise ValueError("DeepSeek-V4 layerwise reuse has no page-size buckets.")

    page_offsets: dict[int, int] = {}
    page_prefix = 0
    for page_size in page_sizes:
        page_offsets[page_size] = page_prefix * kv_cache_config.num_blocks
        page_prefix += page_size
    tuple_stride = page_prefix * kv_cache_config.num_blocks

    group_buckets: list[dict[int, list[str]]] = []
    mtp_names: list[str] = []
    seen_names: set[str] = set()
    for group in groups:
        if not isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs):
            raise ValueError("DeepSeek-V4 layerwise reuse requires uniform specs in every group.")
        buckets: dict[int, list[str]] = {}
        for name in group.layer_names:
            if name in seen_names or name not in group.kv_cache_spec.kv_cache_specs:
                raise ValueError(f"Invalid or duplicate DeepSeek-V4 cache layer: {name}.")
            seen_names.add(name)
            page_size = layer_specs[name].page_size_bytes
            if page_size not in page_offsets:
                raise ValueError(f"DeepSeek-V4 cache layer {name} has no packed page-size bucket.")
            if "mtp" in name:
                mtp_names.append(name)
            else:
                buckets.setdefault(page_size, []).append(name)
        group_buckets.append(buckets)
    if seen_names != set(layer_specs):
        raise ValueError("DeepSeek-V4 cache groups do not cover every named cache spec.")

    normal_slot_count = max((len(names) for buckets in group_buckets for names in buckets.values()), default=0)
    old_slot_count = normal_slot_count + len(mtp_names)
    old_backing_size = tuple_stride * old_slot_count
    expected_positions: dict[str, int] = {}
    for buckets in group_buckets:
        for page_size, names in buckets.items():
            for index, name in enumerate(names):
                expected_positions[name] = index * tuple_stride + page_offsets[page_size]
    for index, name in enumerate(mtp_names):
        expected_positions[name] = (normal_slot_count + index) * tuple_stride + page_offsets[
            layer_specs[name].page_size_bytes
        ]

    old_names: set[str] = set()
    for tensor in kv_cache_config.kv_cache_tensors:
        if tensor.size != old_backing_size:
            raise ValueError("DeepSeek-V4 packed descriptors must share the planned backing size.")
        for index, name in enumerate(get_kv_cache_tensor_layers(tensor)):
            if name in old_names or name not in expected_positions:
                raise ValueError(f"Invalid or duplicate DeepSeek-V4 packed view: {name}.")
            old_names.add(name)
            if (
                tensor.block_stride != layer_specs[name].page_size_bytes
                or tensor.offset + index * tensor.layer_stride != expected_positions[name]
            ):
                raise ValueError(f"Unexpected DeepSeek-V4 packed view geometry for {name}.")
    if old_names != seen_names:
        raise ValueError("DeepSeek-V4 packed descriptors do not cover every cache layer.")

    physical_to_slot: dict[int, int] = {}
    for slot_id, physical_layers in enumerate(reuse_layout.buffer_slots):
        for physical_layer in physical_layers:
            if physical_layer in physical_to_slot:
                raise ValueError(f"DeepSeek-V4 physical layer {physical_layer} has multiple reuse slots.")
            physical_to_slot[physical_layer] = slot_id

    # Keep the planner's MTP tail slots independent. Reusing them with target
    # layers requires a separate lifetime guarantee from the MTP execution path.
    group_lanes: list[dict[int, list[_PackedCacheLane]]] = []
    for buckets in group_buckets:
        lanes_by_page: dict[int, list[_PackedCacheLane]] = {}
        for page_size, names in buckets.items():
            lanes = lanes_by_page.setdefault(page_size, [])
            for name in names:
                physical_layer = get_layerwise_physical_layer_index(name, base_layers)
                slot_id = physical_to_slot[physical_layer]
                role = _dsv4_cache_role(name)
                spec = layer_specs[name]
                lane = next(
                    (
                        lane
                        for lane in lanes
                        if lane.slot_id == slot_id and lane.role == role and lane.spec == spec
                    ),
                    None,
                )
                if lane is None:
                    lanes.append(_PackedCacheLane(slot_id, role, spec, [name]))
                else:
                    lane.layer_names.append(name)
        group_lanes.append(lanes_by_page)

    reused_normal_slots = max((len(lanes) for pages in group_lanes for lanes in pages.values()), default=0)
    reused_slot_count = reused_normal_slots + len(mtp_names)
    if reused_slot_count >= old_slot_count:
        return

    backing_size = tuple_stride * reused_slot_count
    new_tensors: list[KVCacheTensor] = []
    for lanes_by_page in group_lanes:
        for page_size in page_sizes:
            for slot_index, lane in enumerate(lanes_by_page.get(page_size, [])):
                new_tensors.append(
                    KVCacheTensor(
                        size=backing_size,
                        layers=lane.layer_names,
                        offset=slot_index * tuple_stride + page_offsets[page_size],
                        layer_stride=0,
                        block_stride=page_size,
                    )
                )
    for index, name in enumerate(mtp_names):
        page_size = layer_specs[name].page_size_bytes
        new_tensors.append(
            KVCacheTensor(
                size=backing_size,
                layers=[name],
                offset=(reused_normal_slots + index) * tuple_stride + page_offsets[page_size],
                layer_stride=0,
                block_stride=page_size,
            )
        )

    new_names: set[str] = set()
    for tensor in new_tensors:
        for name in tensor.layers:
            layer_end = tensor.offset + kv_cache_config.num_blocks * layer_specs[name].page_size_bytes
            if name in new_names or layer_end > backing_size:
                raise ValueError(f"Invalid DeepSeek-V4 layerwise packed view for {name}.")
            new_names.add(name)
    if new_names != seen_names:
        raise ValueError("DeepSeek-V4 layerwise plan does not cover every cache layer.")

    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "DeepSeek-V4 layerwise KV reuse compacted %d packed tuple slots into %d at fixed num_blocks=%d.",
        old_slot_count,
        reused_slot_count,
        kv_cache_config.num_blocks,
    )


def apply_layerwise_kv_cache_plan(
    kv_cache_config: KVCacheConfig,
    vllm_config: VllmConfig,
) -> None:
    """Rewrite logical layer tensors to use shared physical KV buffers."""
    extra_config = get_layerwise_reuse_config(vllm_config.kv_transfer_config)
    if extra_config is None:
        return

    old_tensors = kv_cache_config.kv_cache_tensors
    if not old_tensors:
        return

    layer_specs = get_layerwise_kv_cache_specs(kv_cache_config)
    is_dsv4_main = not vllm_version_is("0.28.0") and any(
        getattr(spec, "model_version", None) == "deepseek_v4" for spec in layer_specs.values()
    )
    if len(old_tensors) == 1 and not is_dsv4_main:
        return
    base_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    reuse_layout = build_layerwise_reuse_layout(
        layer_specs,
        base_layers,
        extra_config,
    )
    actual_layers = len(reuse_layout.layer_cache_specs)
    if not reuse_layout.has_layer_reuse:
        return
    if is_dsv4_main:
        base_physical_layers = {
            get_layerwise_physical_layer_index(name, base_layers)
            for name in layer_specs
            if "mtp" not in name
        }
        if len(base_physical_layers) < base_layers:
            logger.warning("DeepSeek-V4 layerwise reuse has an incomplete base-layer cache layout; skip compaction.")
            return
        _apply_dsv4_layerwise_kv_cache_plan(kv_cache_config, layer_specs, reuse_layout, base_layers)
        return
    if any(
        len(get_kv_cache_tensor_layers(tensor)) != 1 or tensor.offset != 0 or tensor.block_stride != 0
        for tensor in old_tensors
    ):
        raise NotImplementedError(
            "Layerwise KV cache reuse does not support pre-shared or packed KV cache tensor descriptors."
        )

    if actual_layers < base_layers:
        logger.warning(
            "Layer reuse expected at least %d layers, got %d; skip tensor merge.",
            base_layers,
            actual_layers,
        )
        return
    if actual_layers > base_layers:
        logger.info(
            "Layer reuse includes %d base and %d MTP/spec-decode layer(s).",
            base_layers,
            actual_layers - base_layers,
        )

    tensors_by_name = {get_kv_cache_tensor_layers(tensor)[0]: tensor for tensor in old_tensors}

    def _merge_specs(named_specs: list[NamedKVCacheSpec]) -> None:
        shared_by = [named_spec.layer_name for named_spec in named_specs]
        cache_tensors = [tensors_by_name[layer_name] for layer_name in shared_by]
        tensor_sizes = {tensor.size for tensor in cache_tensors}
        if len(tensor_sizes) != 1:
            raise ValueError("Layers sharing layerwise KV buffers must have equal tensor sizes for every cache spec.")
        reference_spec = layer_specs[shared_by[0]]
        if any(layer_specs[layer_name] != reference_spec for layer_name in shared_by[1:]):
            raise ValueError(
                "Layers sharing layerwise KV buffers must have identical cache specs for every named cache spec."
            )
        if vllm_version_is("0.28.0"):
            new_tensors.append(KVCacheTensor(shared_by=shared_by, size=cache_tensors[0].size))
        else:
            new_tensors.append(
                KVCacheTensor(
                    layers=shared_by,
                    size=cache_tensors[0].size,
                    layer_stride=cache_tensors[0].layer_stride,
                    block_stride=cache_tensors[0].block_stride,
                    offset=cache_tensors[0].offset,
                )
            )

    new_tensors: list[KVCacheTensor] = []
    for slot in reuse_layout.buffer_slots:
        _merge_specs([reuse_layout.layer_cache_specs[layer].main for layer in slot])
        indexer_specs: list[NamedKVCacheSpec] = []
        for layer in slot:
            indexer = reuse_layout.layer_cache_specs[layer].indexer
            if indexer is not None:
                indexer_specs.append(indexer)
        if indexer_specs:
            _merge_specs(indexer_specs)
    kv_cache_config.kv_cache_tensors = new_tensors
    logger.info(
        "Layerwise KV cache reuse merged %d descriptors into %d descriptors using %d buffer assignments.",
        len(old_tensors),
        len(new_tensors),
        len(reuse_layout.buffer_slots),
    )
