import io
import tarfile

import pytest

from tests.e2e.weekly.multi_node.external_dp.scripts.check_dsv4_cache_reuse_log import (  # noqa: E501
    check_cache_reuse_archive,
    check_cache_reuse_log,
)


def test_cache_reuse_log_accepts_compaction():
    log = (
        "INFO DeepSeek-V4 layerwise KV reuse compacted 80 packed tuple slots "
        "into 64 at fixed num_blocks=1024.\n"
    )
    assert check_cache_reuse_log(log) == [(80, 64, 1024)]


@pytest.mark.parametrize(
    "log",
    [
        "startup complete",
        "DeepSeek-V4 layerwise KV reuse compacted 80 packed tuple slots "
        "into 80 at fixed num_blocks=1024.",
        "DeepSeek-V4 layerwise KV reuse compacted 80 packed tuple slots "
        "into 81 at fixed num_blocks=1024.",
        "DeepSeek-V4 layerwise KV reuse compacted 80 packed tuple slots "
        "into 64 at fixed num_blocks=0.",
    ],
)
def test_cache_reuse_log_rejects_missing_or_invalid_compaction(log):
    with pytest.raises(ValueError):
        check_cache_reuse_log(log)


def test_cache_reuse_archive_reads_prefill_rank_logs(tmp_path):
    archive_path = tmp_path / "node_0_external_dp_logs.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, contents in (
            (
                "node-0/rank-0.log",
                "DeepSeek-V4 layerwise KV reuse compacted 80 packed tuple slots "
                "into 64 at fixed num_blocks=1024.",
            ),
            ("node-0/rank-1.log", "startup complete"),
        ):
            data = contents.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    assert check_cache_reuse_archive(archive_path) == [
        ("node-0/rank-0.log", 80, 64, 1024)
    ]
