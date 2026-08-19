"""Bounded structured tracing for CPP execution-mode validation."""

from __future__ import annotations

import json
from typing import Any

from vllm.logger import logger

TRACE_PREFIX = "[CPP_EXECUTION_MODE_TRACE]"


def log_execution_mode_event(event: str, **fields: Any) -> None:
    """Emit one machine-readable execution-mode event.

    Callers are responsible for bounding event cardinality. Keeping that policy
    outside this helper lets startup profiling emit one record per sample while
    normal inference emits only the first observation of each execution mode.
    """

    payload = {"event": event, **fields}
    logger.info("%s %s", TRACE_PREFIX, json.dumps(payload, sort_keys=True))
