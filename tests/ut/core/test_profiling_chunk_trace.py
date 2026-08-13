from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.patch.platform.patch_profiling_chunk import _record_execution_timing


def test_online_calibration_completion_is_traced_and_propagated():
    predictor = SimpleNamespace(history_fitted=False, with_history_ready=True)
    manager = SimpleNamespace(
        is_ready=True,
        history_ready=True,
        predictor=predictor,
        _set_time_done=True,
        chunked_fit_data=[[1, 1, 1, 1.0]] * 8,
    )

    def finish_history_fit(request_chunks, elapsed_time):
        predictor.history_fitted = True
        manager.chunked_fit_data.append([1, 1, 1, elapsed_time * 1000])
        return False

    manager.record_batch_execution_time = finish_history_fit
    scheduler = SimpleNamespace(
        profiling_chunk_manager=manager,
        profiling_chunk_config=SimpleNamespace(
            trace_enabled=True,
            need_timing=True,
        ),
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=512,
        num_scheduled_tokens={"req": 512},
        scheduled_new_reqs=[SimpleNamespace(req_id="req", num_computed_tokens=0)],
        scheduled_cached_reqs=None,
        cpp_trace_records=[
            {
                "iteration": 9,
                "req_id": "req",
                "num_computed_tokens": 0,
                "hist_seq_len": 0,
                "remaining_prefill_tokens": 10000,
                "target_latency_ms": 10.0,
                "predicted_chunk_size": 512,
                "actual_scheduled_chunk_size": 512,
                "predicted_latency_ms": 9.5,
            }
        ],
    )
    model_output = SimpleNamespace(execution_time_ms=11.0)

    with patch(
        "vllm_ascend.patch.platform.patch_profiling_chunk.log_cpp_trace"
    ) as trace:
        _record_execution_timing(scheduler, scheduler_output, model_output)

    assert scheduler._profiling_timing_done
    assert not scheduler.profiling_chunk_config.need_timing
    events = [call.args[0] for call in trace.call_args_list]
    assert "scheduler_iteration" in events
    assert "online_calibration_completed" in events
    iteration_call = next(
        call for call in trace.call_args_list if call.args[0] == "scheduler_iteration"
    )
    assert iteration_call.kwargs["actual_execution_time_ms"] == 11.0
    assert iteration_call.kwargs["batch_execution_time_ms"] == 11.0
    assert iteration_call.kwargs["execution_time_scope"] == "batch"
    assert iteration_call.kwargs["history_fitted"] is True
    assert iteration_call.kwargs["disable_profiling_timing"] is True


def test_trace_disabled_emits_no_cpp_events():
    manager = SimpleNamespace(
        is_ready=True,
        history_ready=False,
        predictor=SimpleNamespace(history_fitted=False),
        _set_time_done=False,
        _set_time_count=0,
        chunked_fit_data=[],
        record_batch_execution_time=MagicMock(return_value=False),
    )
    scheduler = SimpleNamespace(
        profiling_chunk_manager=manager,
        profiling_chunk_config=SimpleNamespace(
            trace_enabled=False,
            need_timing=True,
        ),
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=512,
        num_scheduled_tokens={"req": 512},
        scheduled_new_reqs=[SimpleNamespace(req_id="req", num_computed_tokens=1)],
        scheduled_cached_reqs=None,
    )

    with patch(
        "vllm_ascend.patch.platform.patch_profiling_chunk.log_cpp_trace"
    ) as trace:
        _record_execution_timing(
            scheduler,
            scheduler_output,
            SimpleNamespace(execution_time_ms=10.0),
        )

    trace.assert_not_called()
