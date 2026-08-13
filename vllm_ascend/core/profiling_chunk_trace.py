#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured diagnostics for profiling-based dynamic chunk sizing."""

import json
from typing import Any

from vllm.logger import logger


def log_cpp_trace(event: str, **fields: Any) -> None:
    """Emit one machine-readable CPP trace event at INFO level."""
    payload = {"event": event, **fields}
    logger.info(
        "[CPP_TRACE] %s",
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )
