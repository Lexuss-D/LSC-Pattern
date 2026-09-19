#!/usr/bin/env python3
"""Extract usage examples from the BCCWJ LUW PM corpus."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import deque
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = REPO_ROOT / ".local" / "inputs" / "default_data_path"
DEFAULT_WORD_LIST_PATH = REPO_ROOT / ".local" / "inputs" / "default_word_list_path"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".local" / "inputs" / "default_output_dir"

SEARCH_FIELDS = [
    ("語彙素", "lemma"),
    ("語彙素読み", "lemma_reading"),
    ("語種", "word_origin"),
    ("品詞", "pos"),
    ("書字形出現形", "orthographic_form"),
]
SURFACE_FIELD = "書字形出現形"
TIMESTAMP_FIELD = "出版年"
Usage = dict[str, object]
ConditionKey = tuple[tuple[str, str], ...]


def build_conditions(args: argparse.Namespace) -> dict[str, str]:
    conditions: dict[str, str] = {}
    for field_name, attr_name in SEARCH_FIELDS:
        value = getattr(args, attr_name)
        if value is not None:
            conditions[field_name] = value
    return conditions


def condition_key(conditions: dict[str, str]) -> ConditionKey:
    return tuple(
        (field_name, conditions[field_name])
        for field_name, _ in SEARCH_FIELDS
        if field_name in conditions
    )


def conditions_from_key(key: ConditionKey) -> dict[str, str]:
    return dict(key)


def sanitize_filename_part(value: str) -> str:
    if value == "":
        return "EMPTY"
    sanitized = re.sub(r'[\\/:\*\?"<>\|\x00-\x1f]+', "_", value)
    sanitized = sanitized.strip(". ")
    return sanitized or "EMPTY"


def output_filename(
    conditions: dict[str, str],
    context_window_size: int,
    start_time: int | None,
    end_time: int | None,
) -> str:
    parts = [
        sanitize_filename_part(conditions[field_name])
        for field_name, _ in SEARCH_FIELDS
        if field_name in conditions
    ]
    if start_time is not None or end_time is not None:
        start = str(start_time) if start_time is not None else "min"
        end = str(end_time) if end_time is not None else "max"
        parts.append(f"{start}-{end}")
    parts.append(f"n{context_window_size}")
    return "_".join(parts) + ".jsonl"


def row_matches(row: dict[str, str], conditions: dict[str, str]) -> bool:
    return all(row[field_name] == expected for field_name, expected in conditions.items())


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


def load_word_list_conditions(word_list_path: Path) -> list[dict[str, str]]:
    conditions: list[dict[str, str]] = []
    seen: set[ConditionKey] = set()

    with word_list_path.open("r", encoding="utf-8") as word_list:
        for line_number, line in enumerate(word_list, start=1):
            if not line.strip():
                continue

            item = json.loads(line)
            entry = item.get("data", {}).get("entry", {})
            lemma = entry.get("lemma")
            lemma_reading = entry.get("lemma_reading")


            word_conditions = {"語彙素": lemma, "語彙素読み": lemma_reading}
            key = condition_key(word_conditions)
            if key not in seen:
                conditions.append(word_conditions)
                seen.add(key)

    return conditions


def extract_usages(
    data_path: Path,
    conditions: dict[str, str],
    context_window_size: int,
    start_time: int | None,
    end_time: int | None,
) -> list[Usage]:
    usages: list[Usage] = []
    previous_surfaces: deque[str] = deque(maxlen=context_window_size)
    pending_matches: list[dict[str, object]] = []

    with data_path.open("r", encoding="utf-8", newline="") as corpus:
        reader = csv.DictReader(corpus)
        required_fields = [SURFACE_FIELD, TIMESTAMP_FIELD, *conditions.keys()]


        for line_number, row in enumerate(reader, start=2):


            surface = row[SURFACE_FIELD]

            still_pending: list[dict[str, object]] = []
            for pending in pending_matches:
                after = pending["after"]
                after.append(surface)

                remaining = int(pending["remaining"]) - 1
                if remaining == 0:
                    before = pending["before"]
                    target = pending["target"]
                    timestamp = pending["timestamp"]
                    usages.append({"timestamp": timestamp, "usage": before + [target] + after})
                else:
                    pending["remaining"] = remaining
                    still_pending.append(pending)
            pending_matches = still_pending

            timestamp = row[TIMESTAMP_FIELD]
            if row_matches(row, conditions) and timestamp_in_range(timestamp, start_time, end_time):
                if context_window_size == 0:
                    usages.append({"timestamp": timestamp, "usage": [surface]})
                else:
                    pending_matches.append(
                        {
                            "before": list(previous_surfaces),
                            "target": surface,
                            "timestamp": timestamp,
                            "after": [],
                            "remaining": context_window_size,
                        }
                    )

            previous_surfaces.append(surface)

    for pending in pending_matches:
        before = pending["before"]
        target = pending["target"]
        after = pending["after"]
        timestamp = pending["timestamp"]
        usages.append({"timestamp": timestamp, "usage": before + [target] + after})

    return usages


def extract_usages_batch(
    data_path: Path,
    target_conditions: list[dict[str, str]],
    context_window_size: int,
    start_time: int | None,
    end_time: int | None,
) -> dict[ConditionKey, list[Usage]]:
    target_by_lemma_reading: dict[tuple[str, str], ConditionKey] = {}
    usages_by_key: dict[ConditionKey, list[Usage]] = {}

    for conditions in target_conditions:
        key = condition_key(conditions)
        target_by_lemma_reading[(conditions["語彙素"], conditions["語彙素読み"])] = key
        usages_by_key[key] = []

    previous_surfaces: deque[str] = deque(maxlen=context_window_size)
    pending_matches: list[dict[str, object]] = []

    with data_path.open("r", encoding="utf-8", newline="") as corpus:
        reader = csv.DictReader(corpus)
        required_fields = [SURFACE_FIELD, TIMESTAMP_FIELD, "語彙素", "語彙素読み"]


        for line_number, row in enumerate(reader, start=2):


            surface = row[SURFACE_FIELD]

            still_pending: list[dict[str, object]] = []
            for pending in pending_matches:
                after = pending["after"]
                after.append(surface)

                remaining = int(pending["remaining"]) - 1
                if remaining == 0:
                    before = pending["before"]
                    target = pending["target"]
                    timestamp = pending["timestamp"]
                    key = pending["key"]
                    usages_by_key[key].append({"timestamp": timestamp, "usage": before + [target] + after})
                else:
                    pending["remaining"] = remaining
                    still_pending.append(pending)
            pending_matches = still_pending

            timestamp = row[TIMESTAMP_FIELD]
            key = target_by_lemma_reading.get((row["語彙素"], row["語彙素読み"]))
            if key is not None and timestamp_in_range(timestamp, start_time, end_time):
                if context_window_size == 0:
                    usages_by_key[key].append({"timestamp": timestamp, "usage": [surface]})
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

    for pending in pending_matches:
        before = pending["before"]
        target = pending["target"]
        after = pending["after"]
        timestamp = pending["timestamp"]
        key = pending["key"]
        usages_by_key[key].append({"timestamp": timestamp, "usage": before + [target] + after})

    return usages_by_key


def write_jsonl(usages: Iterable[Usage], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for usage in usages:
            json.dump(usage, output_file, ensure_ascii=False)
            output_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract usage examples from BCCWJ LUW PM_with_bibliography.csv."
    )
    parser.add_argument(
        "--context-window-size",
        type=int,
        default=30,
        help="Number of rows before and after a match to include.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Path to PM_with_bibliography.csv. Defaults to {DEFAULT_DATA_PATH}.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the generated JSON file. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--word-list-jsonl",
        nargs="?",
        const=DEFAULT_WORD_LIST_PATH,
        type=Path,
        help=(
            "JSONL dictionary word list. Each line must contain "
            "data.entry.lemma and data.entry.lemma_reading."
            f" If the flag is used without a value, defaults to {DEFAULT_WORD_LIST_PATH}."
        ),
    )
    parser.add_argument("--lemma", "--語彙素", dest="lemma", help="Exact value for 語彙素.")
    parser.add_argument(
        "--lemma-reading",
        "--語彙素読み",
        dest="lemma_reading",
        help="Exact value for 語彙素読み.",
    )
    parser.add_argument("--word-origin", "--語種", dest="word_origin", help="Exact value for 語種.")
    parser.add_argument("--pos", "--品詞", dest="pos", help="Exact value for 品詞.")
    parser.add_argument(
        "--orthographic-form",
        "--書字形出現形",
        dest="orthographic_form",
        help="Exact value for 書字形出現形.",
    )
    parser.add_argument(
        "--start-time",
        "--start_time",
        dest="start_time",
        type=int,
        help="Inclusive lower bound for 出版年.",
    )
    parser.add_argument(
        "--end-time",
        "--end_time",
        dest="end_time",
        type=int,
        help="Inclusive upper bound for 出版年.",
    )

    args = parser.parse_args()
    conditions = build_conditions(args)
    return args


def main() -> int:
    args = parse_args()
    data_path = args.data_path.resolve()
    output_dir = args.output_dir.resolve()

    try:
        if args.word_list_jsonl is not None:
            target_conditions = load_word_list_conditions(args.word_list_jsonl.resolve())
            usages_by_key = extract_usages_batch(
                data_path,
                target_conditions,
                args.context_window_size,
                args.start_time,
                args.end_time,
            )
            total_usages = 0
            written_files = 0
            for key, usages in usages_by_key.items():
                conditions = conditions_from_key(key)
                output_path = output_dir / output_filename(
                    conditions,
                    args.context_window_size,
                    args.start_time,
                    args.end_time,
                )
                write_jsonl(usages, output_path)
                total_usages += len(usages)
                written_files += 1
            print(f"Wrote {total_usages} usage(s) across {written_files} file(s) to {output_dir}")
            return 0

        conditions = build_conditions(args)
        usages = extract_usages(data_path, conditions, args.context_window_size, args.start_time, args.end_time)
        output_path = output_dir / output_filename(conditions, args.context_window_size, args.start_time, args.end_time)
        write_jsonl(usages, output_path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(usages)} usage(s) to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
