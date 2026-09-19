#!/usr/bin/env python3
"""Build JaSemChange full-sample WSD prompts from marked CSV usages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from build_lscp_wsd_prompts import (
    build_prompt,
    load_sense_index,
    normalize_lemma,
    sanitize_filename_part,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_ROOT = REPO_ROOT / ".local" / "inputs" / "default_samples_root"
DEFAULT_IWANAMI_WORDLIST = REPO_ROOT / ".local" / "inputs" / "default_iwanami_wordlist"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".local" / "inputs" / "default_output_dir"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_OUTPUT_DIR / "jasemchange_usage_manifest.tsv"
RAW_PRIORITY = {
    "samples": ("chj", "bccwj", "before1980", "after1980"),
    "samples_shc_bccwj": ("shc", "bccwj", "before1980", "after1980"),
}
MARK_RE = re.compile(r"\*\*(.*?)\*\*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build prompt JSONL files for JaSemChange full usage CSVs."
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=DEFAULT_SAMPLES_ROOT,
        help=f"JaSemChange_full root. Default: {DEFAULT_SAMPLES_ROOT}",
    )
    parser.add_argument(
        "--iwanami-wordlist",
        type=Path,
        default=DEFAULT_IWANAMI_WORDLIST,
        help=f"Iwanami wordlist JSONL. Default: {DEFAULT_IWANAMI_WORDLIST}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output prompt directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT,
        help=f"Manifest TSV path. Default: {DEFAULT_MANIFEST_OUTPUT}",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Input/output encoding. Default: utf-8",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing JSONL prompt files in output-dir before writing.",
    )
    return parser.parse_args()


def sniff_dialect(path: Path) -> csv.Dialect:
    if path.suffix == ".txt":
        return csv.excel_tab
    return csv.excel


def read_wordlist_readings(wordlist_path: Path, encoding: str) -> dict[str, str]:
    readings: dict[str, set[str]] = defaultdict(set)
    with wordlist_path.open("r", encoding=encoding) as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            entry = obj.get("data", {}).get("entry", {})
            lemma = entry.get("lemma")
            reading = entry.get("lemma_reading")
            readings[normalize_lemma(lemma)].add(reading)

    ambiguous = {lemma: sorted(values) for lemma, values in readings.items() if len(values) != 1}
    return {lemma: next(iter(values)) for lemma, values in readings.items()}


def iter_sample_files(samples_root: Path) -> list[Path]:
    paths: list[Path] = []
    for dataset in RAW_PRIORITY:
        dataset_root = samples_root / dataset
        paths.extend(sorted(dataset_root.glob("*/full_*.csv")))
    return paths


def read_rawdata_indexes(samples_root: Path, targets: set[str]) -> dict[str, dict[tuple[str, str, str, str], dict[str, str]]]:
    rawdata_root = samples_root / "rawdata"
    indexes: dict[str, dict[tuple[str, str, str, str], dict[str, str]]] = {
        dataset: {}
        for dataset in RAW_PRIORITY
    }

    for dataset, raw_dirs in RAW_PRIORITY.items():
        for raw_dir in raw_dirs:
            for target in targets:
                for suffix in (".txt", ".csv"):
                    path = rawdata_root / raw_dir / f"{target}{suffix}"
                    if not path.exists():
                        continue
                    with path.open("r", encoding="utf-8-sig", newline="") as handle:
                        reader = csv.DictReader(handle, dialect=sniff_dialect(path))
                        for row in reader:
                            key = (
                                target,
                                str(row.get("サンプル ID") or "").strip(),
                                str(row.get("開始位置") or "").strip(),
                                str(row.get("連番") or "").strip(),
                            )
                            if not all(key):
                                continue
                            year = str(row.get("成立年") or row.get("出版年") or "").strip()
                            value = {
                                "rawdata_source": raw_dir,
                                "year": year,
                                "work_title": str(row.get("作品名") or row.get("書名/出典") or "").strip(),
                                "raw_before": str(row.get("前文脈") or ""),
                                "raw_key": str(row.get("キー") or ""),
                                "raw_after": str(row.get("後文脈") or ""),
                            }
                            indexes[dataset].setdefault(key, value)
    return indexes


def clean_raw_context(text: str) -> str:
    return text.replace("|", "").replace("#", "")


def marked_from_raw(raw: dict[str, str], target_lemma: str) -> tuple[str, str, str, str]:
    target_surface = raw.get("raw_key") or target_lemma
    before = clean_raw_context(raw.get("raw_before", ""))
    after = clean_raw_context(raw.get("raw_after", ""))
    source_usage_text_marked = f"{before}**{target_surface}**{after}"
    usage_text = f"{before}{target_surface}{after}"
    usage_marked_text = f"{before}[[{target_surface}]]{after}"
    return source_usage_text_marked, target_surface, usage_text, usage_marked_text


def parse_marked_usage(text: str) -> tuple[str, str, str] | None:
    matches = MARK_RE.findall(text)
    if len(matches) != 1:
        return None
    target_surface = matches[0]
    usage_text = MARK_RE.sub(target_surface, text, count=1)
    usage_marked_text = MARK_RE.sub(f"[[{target_surface}]]", text, count=1)
    return target_surface, usage_text, usage_marked_text


def repair_malformed_row(row: dict[str, Any]) -> dict[str, str] | None:
    sample_id = str(row.get("サンプル ID") or "").strip()
    start_position = str(row.get("開始位置") or "").strip()
    serial = str(row.get("連番") or "").strip()
    if row.get("用例文") is not None:
        return None
    if " " not in sample_id or not start_position:
        return None
    repaired_sample_id, repaired_start = sample_id.rsplit(" ", 1)
    return {
        "サンプル ID": repaired_sample_id,
        "開始位置": repaired_start,
        "連番": start_position,
        "用例文": "",
        "repair_reason": "malformed_csv_row",
        "malformed_serial_fragment": serial,
    }


def usage_id_for(target_lemma: str, sample_id: str, start_position: str, serial: str) -> str:
    digest = hashlib.sha1(
        "\t".join([target_lemma, sample_id, start_position, serial]).encode("utf-8")
    ).hexdigest()[:16]
    return f"jasemchange_{digest}"


def output_path_for_target(output_dir: Path, target_lemma: str, reading: str) -> Path:
    filename = (
        f"JaSemChange_full_{sanitize_filename_part(target_lemma)}_"
        f"{sanitize_filename_part(reading)}.jsonl"
    )
    return output_dir / filename


def write_jsonl(path, records, encoding='utf-8'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_manifest(path: Path, records: list[dict[str, Any]], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "usage_id",
        "target_lemma",
        "lemma_reading",
        "target_surface",
        "sample_id",
        "start_position",
        "serial",
        "year",
        "rawdata_source",
        "datasets",
        "periods",
        "source_files",
        "source_rows",
        "output_prompt_file",
    ]
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {}
            for field in fields:
                value = record.get(field, "")
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                row[field] = value
            writer.writerow(row)


def merge_membership(existing: dict[str, Any], new_record: dict[str, Any]) -> None:
    for field in ("datasets", "periods", "source_files", "source_rows", "target_surfaces"):
        merged = set(existing.get(field, []))
        merged.update(new_record.get(field, []))
        existing[field] = sorted(merged)

    if "**" not in str(existing.get("source_usage_text_marked", "")) and "**" in str(
        new_record.get("source_usage_text_marked", "")
    ):
        for field in ("source_usage_text_marked", "target_surface", "usage_text", "usage_marked_text"):
            existing[field] = new_record[field]


def build_records(
    sample_files: list[Path],
    samples_root: Path,
    readings: dict[str, str],
    candidates_by_key: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    targets = {path.stem.removeprefix("full_") for path in sample_files}
    raw_indexes = read_rawdata_indexes(samples_root, targets)
    records_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    stats = {"source_files": len(sample_files), "source_rows": 0, "reconstructed_usages": 0, "deduped_rows": 0}
    missing_year: list[tuple[Path, int, tuple[str, str, str, str]]] = []

    for sample_path in sample_files:
        dataset = sample_path.parents[1].name
        period = sample_path.parent.name
        target_lemma = sample_path.stem.removeprefix("full_")
        reading = readings.get(target_lemma)
        candidates = candidates_by_key.get((target_lemma, reading))

        with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for source_row_number, row in enumerate(reader, start=2):
                stats["source_rows"] += 1
                repaired = repair_malformed_row(row)
                if repaired is not None:
                    row = repaired

                sample_id = str(row.get("サンプル ID") or "").strip()
                start_position = str(row.get("開始位置") or "").strip()
                serial = str(row.get("連番") or "").strip()
                key = (target_lemma, sample_id, start_position, serial)
                raw = raw_indexes[dataset].get(key)
                if raw is None or not raw.get("year"):
                    missing_year.append((sample_path, source_row_number, key))
                    continue

                source_usage_text_marked = str(row.get("用例文") or "")
                parsed = parse_marked_usage(source_usage_text_marked)
                if parsed is None:
                    stats["reconstructed_usages"] += 1
                    (
                        source_usage_text_marked,
                        target_surface,
                        usage_text,
                        usage_marked_text,
                    ) = marked_from_raw(raw, target_lemma)
                else:
                    target_surface, usage_text, usage_marked_text = parsed

                source_row_id = f"{sample_path.relative_to(samples_root)}:{source_row_number}"
                record = {
                    "usage_id": usage_id_for(target_lemma, sample_id, start_position, serial),
                    "timestamp": raw["year"],
                    "year": raw["year"],
                    "corpus": "JaSemChange_full",
                    "target_lemma": target_lemma,
                    "lemma": target_lemma,
                    "lemma_reading": reading,
                    "target_surface": target_surface,
                    "target_surfaces": [target_surface],
                    "sample_id": sample_id,
                    "start_position": start_position,
                    "serial": serial,
                    "rawdata_source": raw["rawdata_source"],
                    "work_title": raw["work_title"],
                    "datasets": [dataset],
                    "periods": [period],
                    "source_files": [str(sample_path)],
                    "source_rows": [source_row_id],
                    "source_usage_text_marked": source_usage_text_marked,
                    "usage_text": usage_text,
                    "usage_marked_text": usage_marked_text,
                    "target_token_index": None,
                    "source_usage_file": str(sample_path),
                    "source_usage_row": source_row_number,
                    "wsd_candidate_synsets": [dict(candidate) for candidate in candidates],
                    "prompt": build_prompt(usage_marked_text, candidates),
                }

                if key in records_by_key:
                    stats["deduped_rows"] += 1
                    merge_membership(records_by_key[key], record)
                else:
                    records_by_key[key] = record


    return sorted(records_by_key.values(), key=lambda item: (item["target_lemma"], item["sample_id"], item["start_position"], item["serial"])), stats


def main() -> None:
    args = parse_args()

    from common import ROOT, read_csv
    frozen = {r['target_lemma'] for r in read_csv(ROOT/'data/jasemchange_targets.csv')}
    sample_files = [p for p in iter_sample_files(args.samples_root) if p.stem.removeprefix('full_') in frozen]
    readings = read_wordlist_readings(args.iwanami_wordlist, args.encoding)
    candidates_by_key = load_sense_index(args.iwanami_wordlist, args.encoding)
    records, stats = build_records(sample_files, args.samples_root, readings, candidates_by_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for old_path in args.output_dir.glob("*.jsonl"):
            old_path.unlink()

    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_target[record["target_lemma"]].append(record)

    manifest_rows: list[dict[str, Any]] = []
    for target_lemma, target_records in sorted(by_target.items()):
        reading = target_records[0]["lemma_reading"]
        output_path = output_path_for_target(args.output_dir, target_lemma, reading)
        for record in target_records:
            record["output_prompt_file"] = str(output_path)
        write_jsonl(output_path, target_records, args.encoding)
        manifest_rows.extend(target_records)

    write_manifest(args.manifest_output, manifest_rows, args.encoding)
    print(f"Source files: {stats['source_files']}")
    print(f"Source rows: {stats['source_rows']}")
    print(f"Unique usages: {len(records)}")
    print(f"Deduped rows: {stats['deduped_rows']}")
    print(f"Reconstructed usages: {stats['reconstructed_usages']}")
    print(f"Prompt files: {len(by_target)}")
    print(f"Prompt dir: {args.output_dir}")
    print(f"Manifest TSV: {args.manifest_output}")


if __name__ == "__main__":
    main()
