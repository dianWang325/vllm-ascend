"""Require physical layerwise KV slot compaction in the DSV4 prefill log."""

import argparse
import re
import tarfile
from pathlib import Path

COMPACTION_LOG = re.compile(
    r"DeepSeek-V4 layerwise KV reuse compacted (\d+) packed tuple slots "
    r"into (\d+) at fixed num_blocks=(\d+)\."
)


def parse_cache_reuse_log(log: str) -> list[tuple[int, int, int]]:
    matches = [tuple(map(int, match)) for match in COMPACTION_LOG.findall(log)]
    for old_slots, new_slots, num_blocks in matches:
        if not 0 < new_slots < old_slots or num_blocks <= 0:
            raise ValueError(
                "Invalid DeepSeek-V4 layerwise KV reuse compaction: "
                f"{old_slots} -> {new_slots}, num_blocks={num_blocks}"
            )
    return matches


def check_cache_reuse_log(log: str) -> list[tuple[int, int, int]]:
    matches = parse_cache_reuse_log(log)
    if not matches:
        raise ValueError("No DeepSeek-V4 layerwise KV reuse compaction log found")
    return matches


def check_cache_reuse_archive(archive_path: Path) -> list[tuple[str, int, int, int]]:
    matches = []
    with tarfile.open(archive_path, "r:gz") as archive:
        rank_logs = [
            member
            for member in archive.getmembers()
            if member.isfile() and re.fullmatch(r"node-0/rank-\d+\.log", member.name)
        ]
        if not rank_logs:
            raise ValueError("No prefill rank logs found in external DP archive")
        for member in rank_logs:
            log_file = archive.extractfile(member)
            if log_file is None:
                raise ValueError(f"Cannot read {member.name} from external DP archive")
            log = log_file.read().decode(errors="replace")
            for old_slots, new_slots, num_blocks in parse_cache_reuse_log(log):
                matches.append((member.name, old_slots, new_slots, num_blocks))
    if not matches:
        raise ValueError("No DeepSeek-V4 layerwise KV reuse compaction log found in prefill rank logs")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    matches = check_cache_reuse_archive(args.archive)
    print(f"Verified {len(matches)} DeepSeek-V4 layerwise KV reuse compaction log(s)")
    for rank_log, old_slots, new_slots, num_blocks in matches:
        print(f"  {rank_log}: {old_slots} -> {new_slots} at fixed num_blocks={num_blocks}")


if __name__ == "__main__":
    main()
