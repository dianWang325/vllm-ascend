from types import SimpleNamespace
from unittest.mock import MagicMock

import vllm_ascend.patch.platform.patch_profiling_chunk as profiling_patch


def _scheduler_output():
    return SimpleNamespace(
        total_num_scheduled_tokens=16,
        num_scheduled_tokens={"new": 10, "cached": 6},
        scheduled_new_reqs=[SimpleNamespace(request_id="new", num_computed_tokens=0)],
        scheduled_cached_reqs=SimpleNamespace(req_ids=["cached"], num_computed_tokens=[4]),
    )


def test_collect_step_chunks_uses_scheduler_output_histories():
    scheduler = SimpleNamespace(requests={})

    chunks = profiling_patch._collect_step_chunks(scheduler, _scheduler_output())

    assert chunks == [("new", 10, 0), ("cached", 6, 4)]


def test_pp_step_timing_wrapper_logs_features_without_changing_output(monkeypatch):
    output = _scheduler_output()

    class FakeScheduler:
        requests = {}

        def schedule(self):
            return output

        def update_from_output(self, scheduler_output, model_output):
            return scheduler_output, model_output

    scheduler = FakeScheduler()
    timestamps = iter([1_000_000_000, 1_012_500_000])
    monkeypatch.setattr(profiling_patch.time, "perf_counter_ns", lambda: next(timestamps))
    log = MagicMock()
    monkeypatch.setattr(profiling_patch.logger, "info", log)

    profiling_patch._ensure_pp_step_timing_wrapped(scheduler)
    scheduled_output = scheduler.schedule()
    result = scheduler.update_from_output(scheduled_output, "model-output")

    assert scheduled_output is output
    assert result == (output, "model-output")
    log.assert_called_once_with(
        "[PPStepTiming] step=%d latency_ms=%.3f num_requests=%d total_tokens=%d x1=%s x2=%s chunks=%s",
        1,
        12.5,
        2,
        16,
        160,
        20,
        [("new", 10, 0), ("cached", 6, 4)],
    )
