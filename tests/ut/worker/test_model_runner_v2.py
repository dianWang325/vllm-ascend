from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.config.compilation import CUDAGraphMode

from vllm_ascend.worker.v2.model_runner import (
    NPUModelRunner,
    flashcomm_dispatch_wrapper,
)


def _make_runner(need_timing: bool = True, trace_enabled: bool = False):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.ascend_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(
            profiling_chunk_config=SimpleNamespace(
                need_timing=need_timing,
                trace_enabled=trace_enabled,
            )
        )
    )
    runner.vllm_config = SimpleNamespace()
    runner.execute_model_state = None
    runner.is_last_pp_rank = False
    return runner


def test_dummy_run_is_inherited_from_upstream_model_runner():
    assert NPUModelRunner._dummy_run is GPUModelRunner._dummy_run


def test_dispatch_wrapper_captures_final_execution_mode():
    descriptor = SimpleNamespace(cg_mode=CUDAGraphMode.NONE)
    callback = MagicMock()
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(tensor_parallel_size=1))

    with (
        patch("vllm_ascend.worker.v2.model_runner.enable_sp", return_value=False),
        patch(
            "vllm_ascend.worker.v2.model_runner.vllm_model_runner.dispatch_cg_and_sync_dp",
            return_value=(descriptor, None),
        ),
        flashcomm_dispatch_wrapper(vllm_config, callback),
    ):
        from vllm.v1.worker.gpu import model_runner as upstream_model_runner

        upstream_model_runner.dispatch_cg_and_sync_dp(
            None, 1, 512, None, 1, 0, need_eager=False
        )

    callback.assert_called_once_with(descriptor, False)


def test_execute_model_starts_profiling_timer():
    runner = _make_runner()
    scheduler_output = SimpleNamespace(
        disable_profiling_timing=False
    )

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
        patch(
            "vllm_ascend.worker.v2.model_runner.torch.npu.synchronize"
        ) as mock_synchronize,
        patch(
            "vllm_ascend.worker.v2.model_runner.time.perf_counter",
            return_value=10.0,
        ),
    ):
        output = runner.execute_model(scheduler_output)

    assert output is None
    assert runner._execution_start_time == 10.0
    mock_synchronize.assert_called_once_with()
    mock_execute_model.assert_called_once_with(
        scheduler_output,
        intermediate_tensors=None,
        dummy_run=False,
        skip_attn_for_dummy_run=False,
        is_profile=False,
    )


def test_execute_model_disables_profiling_timer():
    runner = _make_runner()
    scheduler_output = SimpleNamespace(
        disable_profiling_timing=True
    )

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
        patch(
            "vllm_ascend.worker.v2.model_runner.torch.npu.synchronize"
        ) as mock_synchronize,
        patch(
            "vllm_ascend.worker.v2.model_runner.time.perf_counter"
        ) as mock_perf_counter,
    ):
        runner.execute_model(scheduler_output)

    profiling_config = (
        runner.ascend_config.scheduler_config.profiling_chunk_config
    )
    assert not profiling_config.need_timing
    mock_synchronize.assert_not_called()
    mock_perf_counter.assert_not_called()


@pytest.mark.parametrize("async_output", [False, True])
def test_sample_tokens_records_execution_time(async_output):
    runner = _make_runner()
    runner._execution_start_time = 10.0

    model_runner_output = SimpleNamespace()
    output = (
        SimpleNamespace(model_runner_output=model_runner_output)
        if async_output
        else model_runner_output
    )
    grammar_output = SimpleNamespace()

    with (
        patch.object(
            GPUModelRunner,
            "sample_tokens",
            return_value=output,
        ) as mock_sample_tokens,
        patch(
            "vllm_ascend.worker.v2.model_runner.torch.npu.synchronize"
        ) as mock_synchronize,
        patch(
            "vllm_ascend.worker.v2.model_runner.time.perf_counter",
            return_value=10.125,
        ),
    ):
        result = runner.sample_tokens(grammar_output)

    assert result is output
    assert model_runner_output.execution_time_ms == pytest.approx(125.0)
    mock_synchronize.assert_called_once_with()
    mock_sample_tokens.assert_called_once_with(grammar_output)
