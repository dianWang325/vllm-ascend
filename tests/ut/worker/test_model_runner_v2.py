from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.worker.v2.model_runner import NPUModelRunner


def _make_runner(
    need_timing: bool = True,
    execution_mode_trace_enabled: bool = False,
):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(
                need_timing=need_timing,
                execution_mode_trace_enabled=execution_mode_trace_enabled,
            )
        )
    )
    runner.vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY)
    runner.execute_model_state = None
    runner.is_last_pp_rank = False
    return runner


def _dispatch_once(need_eager=False):
    from vllm_ascend.worker.v2 import model_runner as model_runner_module

    return model_runner_module.vllm_model_runner.dispatch_cg_and_sync_dp(
        None,
        1,
        1,
        True,
        1,
        0,
        need_eager=need_eager,
    )


def test_cpp_startup_profile_captures_final_eager_mode():
    runner = _make_runner(
        need_timing=False,
        execution_mode_trace_enabled=True,
    )
    runner._cpp_startup_profile_active = True
    descriptor = SimpleNamespace(cg_mode=CUDAGraphMode.NONE)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            side_effect=lambda *args, **kwargs: _dispatch_once(need_eager=False),
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.vllm_model_runner.dispatch_cg_and_sync_dp",
            return_value=(descriptor, None, None, None, None),
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.enable_sp",
            return_value=False,
        ),
    ):
        runner.execute_model(SimpleNamespace(), dummy_run=True)

    assert runner._cpp_profile_execution_mode == "NONE"
    assert runner._cpp_profile_need_eager is True


def test_normal_inference_trace_is_bounded_per_observed_mode():
    runner = _make_runner(
        need_timing=False,
        execution_mode_trace_enabled=True,
    )
    descriptor = SimpleNamespace(cg_mode=CUDAGraphMode.FULL)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            side_effect=lambda *args, **kwargs: _dispatch_once(),
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.vllm_model_runner.dispatch_cg_and_sync_dp",
            return_value=(descriptor, None, None, None, None),
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.enable_sp",
            return_value=False,
        ),
        patch("vllm_ascend.worker.v2.model_runner.get_pp_group") as mock_pp_group,
        patch("vllm_ascend.worker.v2.model_runner.get_tp_group") as mock_tp_group,
        patch("vllm_ascend.worker.v2.model_runner.log_execution_mode_event") as mock_log,
    ):
        mock_pp_group.return_value.rank_in_group = 0
        mock_tp_group.return_value.rank_in_group = 0
        runner.execute_model(SimpleNamespace())
        runner.execute_model(SimpleNamespace())

    mock_log.assert_called_once_with(
        "inference_execution_mode_observed",
        runner="mrv2",
        configured_cudagraph_mode="FULL_DECODE_ONLY",
        actual_execution_mode="FULL",
        need_eager=False,
        pp_rank=0,
        tp_rank=0,
    )


def test_execute_model_records_profiling_time():
    runner = _make_runner()
    scheduler_output = SimpleNamespace(disable_profiling_timing=False)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ) as mock_execute_model,
        patch(
            "vllm_ascend.worker.v2.model_runner.enable_sp",
            return_value=False,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch(
            "vllm_ascend.core.profiling_chunk_predictor.time.perf_counter",
            side_effect=[10.0, 10.125],
        ),
    ):
        output = runner.execute_model(scheduler_output)

    assert output is None
    assert runner._cpp_execution_time_ms == pytest.approx(125.0)
    assert mock_synchronize.call_count == 2
    mock_execute_model.assert_called_once_with(
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
    )


def test_execute_model_disables_profiling_timer_and_clears_stale_time():
    runner = _make_runner()
    runner._cpp_execution_time_ms = 123.0
    scheduler_output = SimpleNamespace(disable_profiling_timing=True)

    with (
        patch.object(
            GPUModelRunner,
            "execute_model",
            return_value=None,
        ),
        patch(
            "vllm_ascend.worker.v2.model_runner.enable_sp",
            return_value=False,
        ),
        patch("vllm_ascend.core.profiling_chunk_predictor.torch.npu.synchronize") as mock_synchronize,
        patch("vllm_ascend.core.profiling_chunk_predictor.time.perf_counter") as mock_perf_counter,
    ):
        runner.execute_model(scheduler_output)

    profiling_config = runner.ascend_config.scheduler_config.profiling_chunk_config
    assert not profiling_config.need_timing
    assert runner._cpp_execution_time_ms is None
    mock_synchronize.assert_not_called()
    mock_perf_counter.assert_not_called()
