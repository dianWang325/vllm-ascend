import json
from unittest.mock import patch

from vllm_ascend.core.profiling_chunk_execution_trace import (
    TRACE_PREFIX,
    log_execution_mode_event,
)


def test_execution_mode_trace_is_structured_json():
    with patch("vllm_ascend.core.profiling_chunk_execution_trace.logger.info") as mock_info:
        log_execution_mode_event(
            "startup_profile_execution_mode",
            runner="mrv2",
            actual_execution_mode="NONE",
            need_eager=True,
        )

    prefix, payload = mock_info.call_args.args[1:]
    assert prefix == TRACE_PREFIX
    assert json.loads(payload) == {
        "actual_execution_mode": "NONE",
        "event": "startup_profile_execution_mode",
        "need_eager": True,
        "runner": "mrv2",
    }
