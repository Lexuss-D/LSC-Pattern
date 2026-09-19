#!/usr/bin/env python3
"""Extract CHJ/SHC long-unit usage examples for Iwanami cherrypick words."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHJ_DATA_PATH = REPO_ROOT / ".local" / "inputs" / "default_chj_data_path"
DEFAULT_SHC_DATA_PATH = REPO_ROOT / ".local" / "inputs" / "default_shc_data_path"
DEFAULT_WORD_LIST_PATH = REPO_ROOT / ".local" / "inputs" / "default_word_list_path"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local" / "inputs" / "default_output_root"

LEMMA_FIELD = "長単位語彙素"
READING_FIELD = "長単位語彙素読み"
SURFACE_FIELD = "長単位出現書字形"
TIMESTAMP_FIELD = "成立年"
REQUIRED_FIELDS = [LEMMA_FIELD, READING_FIELD, SURFACE_FIELD, TIMESTAMP_FIELD]
PROGRESS_INTERVAL = 1_000_000


@dataclass(frozen=True)
class Target:
    lemma: str
    reading: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.lemma, self.reading)


@dataclass
class OutputState:
    final_path: Path
    handle: TextIO
    count: int = 0


def sanitize_filename_part(value: str) -> str:
    if value == "":
        return "EMPTY"
    sanitized = re.sub(r'[\\/:\*\?"<>\|\x00-\x1f]+', "_", value)
    sanitized = sanitized.strip(". ")
    return sanitized or "EMPTY"


def output_filename(target: Target, context_window_size: int, start_time: int | None, end_time: int | None) -> str:
    parts = [sanitize_filename_part(target.lemma), sanitize_filename_part(target.reading)]
    if start_time is not None or end_time is not None:
        start = str(start_time) if start_time is not None else "min"
        end = str(end_time) if end_time is not None else "max"
        parts.append(f"{start}-{end}")
    parts.append(f"n{context_window_size}")
    return "_".join(parts) + ".jsonl"


def timestamp_in_range(timestamp: str, start_time: int | None, end_time: int | None) -> bool:
    if start_time is None and end_time is None:
        return True

    try:
        year = int(timestamp)
    except ValueError:
        return False

    if start_time is not None and year < start_time:
        return False
    if end_time is not None and year > end_time:
        return False
    return True


def load_targets(word_list_path: Path) -> list[Target]:
    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()

    with word_list_path.open("r", encoding="utf-8") as word_list:
        for line_number, line in enumerate(word_list, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            entry = item.get("data", {}).get("entry", {})
            lemma = entry.get("lemma")
            reading = entry.get("lemma_reading")

            target = Target(lemma=lemma, reading=reading)
            if target.key not in seen:
                targets.append(target)
                seen.add(target.key)

    return targets


def open_outputs(
    output_dir: Path,
    targets: list[Target],
    context_window_size: int,
    start_time: int | None,
    end_time: int | None,
) -> dict[tuple[str, str], OutputState]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[tuple[str, str], OutputState] = {}

    for target in targets:
        final_path = output_dir / output_filename(target, context_window_size, start_time, end_time)
        handle = final_path.open("w", encoding="utf-8")
        outputs[target.key] = OutputState(final_path=final_path, handle=handle)

    return outputs


def close_outputs(outputs):
    for output in outputs.values():
        output.handle.close()


def write_usage(output: OutputState, timestamp: str, usage: list[str]) -> None:
    json.dump({"timestamp": timestamp, "usage": usage}, output.handle, ensure_ascii=False)
    output.handle.write("\n")
    output.count += 1


def emit_pending_usage(pending: dict[str, object], outputs: dict[tuple[str, str], OutputState]) -> None:
    key = pending["key"]
    before = pending["before"]
    target = pending["target"]
    after = pending["after"]
    timestamp = pending["timestamp"]


    write_usage(outputs[key], timestamp, before + [target] + after)


def extract_for_corpus(
    corpus_label: str,
    data_path: Path,
    output_root: Path,
    targets: list[Target],
    context_window_size: int,
    start_time: int | None,
    end_time: int | None,
) -> dict[str, int]:
    target_keys = {target.key for target in targets}
    output_dir = output_root / f"{corpus_label}_cherry"
    outputs = open_outputs(output_dir, targets, context_window_size, start_time, end_time)
    previous_surfaces: deque[str] = deque(maxlen=context_window_size)
    pending_matches: list[dict[str, object]] = []
    row_count = 0

    try:
        with data_path.open("r", encoding="utf-8", newline="") as corpus_file:
            reader = csv.DictReader(corpus_file)


            for line_number, row in enumerate(reader, start=2):


                surface = row[SURFACE_FIELD]

                still_pending: list[dict[str, object]] = []
                for pending in pending_matches:
                    after = pending["after"]
                    after.append(surface)

                    remaining = int(pending["remaining"]) - 1
                    if remaining == 0:
                        emit_pending_usage(pending, outputs)
                    else:
                        pending["remaining"] = remaining
                        still_pending.append(pending)
                pending_matches = still_pending

                key = (row[LEMMA_FIELD], row[READING_FIELD])
                timestamp = row[TIMESTAMP_FIELD]
                if key in target_keys and timestamp_in_range(timestamp, start_time, end_time):
                    if context_window_size == 0:
                        write_usage(outputs[key], timestamp, [surface])
                    else:
                        pending_matches.append(
                            {
                                "key": key,
                                "before": list(previous_surfaces),
                                "target": surface,
                                "timestamp": timestamp,
                                "after": [],
                                "remaining": context_window_size,
                            }
                        )

                previous_surfaces.append(surface)
                row_count += 1
                if row_count % PROGRESS_INTERVAL == 0:
                    print(f"{corpus_label}: processed {row_count:,} row(s)", flush=True)

        for pending in pending_matches:
            emit_pending_usage(pending, outputs)

    finally:
        close_outputs(outputs)

    total_usages = sum(output.count for output in outputs.values())
    return {
        "row_count": row_count,
        "target_count": len(targets),
        "file_count": len(outputs),
        "usage_count": total_usages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CHJ/SHC long-unit usage examples for cherrypick words."
    )
    parser.add_argument("--corpus", choices=["CHJ", "SHC", "both"], default="both")
    parser.add_argument("--context-window-size", type=int, default=30)
    parser.add_argument("--word-list-jsonl", type=Path, default=DEFAULT_WORD_LIST_PATH)
    parser.add_argument("--chj-data-path", type=Path, default=DEFAULT_CHJ_DATA_PATH)
    parser.add_argument("--shc-data-path", type=Path, default=DEFAULT_SHC_DATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-time", "--start_time", dest="start_time", type=int)
    parser.add_argument("--end-time", "--end_time", dest="end_time", type=int)

    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()

    try:
        targets = load_targets(args.word_list_jsonl.resolve())
        output_root = args.output_root.resolve()
        corpus_jobs: list[tuple[str, Path]] = []
        if args.corpus in {"CHJ", "both"}:
            corpus_jobs.append(("CHJ", args.chj_data_path.resolve()))
        if args.corpus in {"SHC", "both"}:
            corpus_jobs.append(("SHC", args.shc_data_path.resolve()))

        for corpus_label, data_path in corpus_jobs:
            stats = extract_for_corpus(
                corpus_label=corpus_label,
                data_path=data_path,
                output_root=output_root,
                targets=targets,
                context_window_size=args.context_window_size,
                start_time=args.start_time,
                end_time=args.end_time,
            )
            output_dir = output_root / f"{corpus_label}_general"
            print(
                f"{corpus_label}: wrote {stats['usage_count']} usage(s) "
                f"across {stats['file_count']} file(s) to {output_dir}"
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
