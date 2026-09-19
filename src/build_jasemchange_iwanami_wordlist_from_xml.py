#!/usr/bin/env python3
"""Build a JaSemChange Iwanami wordlist JSONL from iwanami_corpus.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_lscp_wsd_prompts import collect_eg_texts, collect_nonempty_senses, collect_texts


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLES_ROOT = REPO_ROOT / ".local" / "inputs" / "default_samples_root"
DEFAULT_RAWDATA_ROOT = REPO_ROOT / ".local" / "inputs" / "default_rawdata_root"
DEFAULT_IWANAMI_JSONL = REPO_ROOT / ".local" / "inputs" / "default_iwanami_jsonl"
DEFAULT_OUTPUT_JSONL = REPO_ROOT / ".local" / "inputs" / "default_output_jsonl"
DEFAULT_COVERAGE_OUTPUT = REPO_ROOT / ".local" / "inputs" / "default_coverage_output"
SAMPLE_DIRS = ("samples", "samples_shc_bccwj")
RAWDATA_DIRS = ("chj", "shc", "bccwj", "before1980", "after1980")
SURFACE_SPLIT_RE = re.compile(r"[・･／/、,，\s]+")
DECORATION_RE = re.compile(r"[×△▲▽＊*{}｛｝〈〉《》「」『』（）()]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract JaSemChange target words from iwanami_corpus.jsonl and write "
            "the JSONL schema consumed by the LUW WSD prompt builder."
        )
    )
    parser.add_argument(
        "--iwanami-jsonl",
        type=Path,
        default=DEFAULT_IWANAMI_JSONL,
        help=f"Path to iwanami_corpus.jsonl. Default: {DEFAULT_IWANAMI_JSONL}",
    )
    parser.add_argument(
        "--xml-root",
        type=Path,
        default=None,
        help=(
            "Deprecated compatibility alias for --iwanami-jsonl. "
            "Pass the iwanami_corpus.jsonl path."
        ),
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=DEFAULT_SAMPLES_ROOT,
        help=f"JaSemChange_full root. Default: {DEFAULT_SAMPLES_ROOT}",
    )
    parser.add_argument(
        "--target-list",
        type=Path,
        default=None,
        help="Optional text/CSV file listing target lemmas. Defaults to full_*.csv stems.",
    )
    parser.add_argument(
        "--rawdata-root",
        type=Path,
        default=DEFAULT_RAWDATA_ROOT,
        help=f"Rawdata root used to infer readings. Default: {DEFAULT_RAWDATA_ROOT}",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=DEFAULT_OUTPUT_JSONL,
        help=f"Output wordlist JSONL. Default: {DEFAULT_OUTPUT_JSONL}",
    )
    parser.add_argument(
        "--coverage-output",
        type=Path,
        default=DEFAULT_COVERAGE_OUTPUT,
        help=f"Coverage TSV path. Default: {DEFAULT_COVERAGE_OUTPUT}",
    )
    return parser.parse_args()


def normalize_surface(value: str) -> str:
    value = value.split("-", 1)[0].strip()
    value = DECORATION_RE.sub("", value)
    return value.strip()


def normalize_reading(value: str) -> str:
    return value.strip().replace("・", "").replace("･", "")


def katakana_to_hiragana(value: str) -> str:
    chars: list[str] = []
    for char in value:
        code = ord(char)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(char)
    return "".join(chars)


def collect_field_texts(entry: dict[str, Any], field: str) -> list[str]:
    value = entry.get(field)
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [text.strip() for text in collect_texts(value) if text.strip()]


def candidate_surfaces(entry: dict[str, Any]) -> set[str]:
    surfaces: set[str] = set()

    for field in ("lemma", "orth", "old", "hd"):
        for text in collect_field_texts(entry, field):
            add_surface_variants(text, surfaces)

    entry_label = entry.get("@entry")
    if isinstance(entry_label, str):
        add_surface_variants(entry_label, surfaces)

    return {surface for surface in surfaces if surface}


def add_surface_variants(text: str, surfaces: set[str]) -> None:
    raw_values = [text]
    raw_values.extend(SURFACE_SPLIT_RE.split(text))
    for raw_value in raw_values:
        cleaned = normalize_surface(raw_value)
        if cleaned:
            surfaces.add(cleaned)


def populate_sense_definitions(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            populate_sense_definitions(item)
        return
    if not isinstance(value, dict):
        return

    if "pos" in value and not isinstance(value["pos"], str):
        value["pos"] = "".join(collect_texts(value["pos"])).strip()

    if "sense" in value or "@senseid" in value:
        sense_definition = str(value.get("sense_definition") or "").strip()
        if not sense_definition:
            sense_definition = "".join(collect_texts(value.get("su"))).strip()
        value["sense_definition"] = sense_definition

        eg_definition = str(value.get("eg_definition") or "").strip()
        if not eg_definition:
            eg_definition = "|".join(collect_eg_texts(value.get("eg"))).strip()
        value["eg_definition"] = eg_definition

    for child in value.values():
        populate_sense_definitions(child)


def load_targets_from_samples(samples_root: Path) -> list[str]:
    targets: set[str] = set()
    for sample_dir in SAMPLE_DIRS:
        for path in (samples_root / sample_dir).glob("*/full_*.csv"):
            targets.add(path.stem.removeprefix("full_"))
    return sorted(targets)


def load_targets_from_file(path: Path) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        if "," in first_line:
            reader = csv.DictReader(handle)
            field = "target_lemma" if reader.fieldnames and "target_lemma" in reader.fieldnames else None
            field = field or ("target" if reader.fieldnames and "target" in reader.fieldnames else None)
            if field is None:
                handle.seek(0)
                for row in csv.reader(handle):
                    if row and row[0].strip() and not row[0].startswith("#"):
                        target = row[0].strip()
                        if target not in seen:
                            targets.append(target)
                            seen.add(target)
            else:
                for row in reader:
                    target = str(row.get(field) or "").strip()
                    if target and target not in seen:
                        targets.append(target)
                        seen.add(target)
        else:
            for line in handle:
                target = line.strip()
                if target and not target.startswith("#") and target not in seen:
                    targets.append(target)
                    seen.add(target)
    return targets


def sniff_dialect(path: Path) -> csv.Dialect:
    if path.suffix == ".txt":
        return csv.excel_tab
    return csv.excel


def infer_readings(rawdata_root: Path, targets: list[str]) -> dict[str, str]:
    readings: dict[str, Counter[str]] = {target: Counter() for target in targets}
    target_set = set(targets)

    for rawdata_dir in RAWDATA_DIRS:
        corpus_dir = rawdata_root / rawdata_dir
        if not corpus_dir.exists():
            continue
        for target in target_set:
            for suffix in (".txt", ".csv"):
                path = corpus_dir / f"{target}{suffix}"
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle, dialect=sniff_dialect(path))
                    if not reader.fieldnames or "語彙素読み" not in reader.fieldnames:
                        continue
                    for row in reader:
                        reading = str(row.get("語彙素読み") or "").strip()
                        if reading:
                            readings[target][reading] += 1

    missing = [target for target, counter in readings.items() if not counter]

    return {
        target: counter.most_common(1)[0][0]
        for target, counter in readings.items()
    }


def load_iwanami_entries(iwanami_jsonl: Path) -> list[tuple[int, dict[str, Any]]]:
    entries: list[tuple[int, dict[str, Any]]] = []
    with iwanami_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            entry = obj.get("data", {}).get("entry", {})
            if isinstance(entry, dict):
                entries.append((line_number, obj))

    return entries


def entry_reading(entry: dict[str, Any]) -> str:
    lemma_reading = entry.get("lemma_reading")
    if isinstance(lemma_reading, str) and lemma_reading.strip():
        return normalize_reading(lemma_reading)

    hd = entry.get("hd")
    if isinstance(hd, str) and hd.strip():
        return normalize_reading(hd)

    entry_label = entry.get("@entry")
    if isinstance(entry_label, str) and entry_label.strip():
        return normalize_reading(entry_label.replace("　", " ").split()[0])

    return ""


def match_entry(
    target: str,
    target_reading: str,
    entries: list[tuple[int, dict[str, Any]]],
) -> tuple[int, dict[str, Any], str] | None:
    normalized_reading = katakana_to_hiragana(normalize_reading(target_reading))

    exact_matches: list[tuple[int, dict[str, Any], str]] = []
    for line_number, obj in entries:
        entry = obj.get("data", {}).get("entry", {})
        surfaces = candidate_surfaces(entry)
        if target not in surfaces:
            continue
        reading = katakana_to_hiragana(entry_reading(entry))
        match_type = "headword"
        if reading and reading == normalized_reading:
            return line_number, obj, match_type
        exact_matches.append((line_number, obj, match_type))

    if exact_matches:
        return exact_matches[0]

    # Some JaSemChange spellings are historical kanji forms not used as the Iwanami headword.
    # Restrict fallback to entries with matching reading and an explicit target string in the entry.
    for line_number, obj in entries:
        entry = obj.get("data", {}).get("entry", {})
        reading = katakana_to_hiragana(entry_reading(entry))
        if reading != normalized_reading:
            continue
        if target in json.dumps(entry, ensure_ascii=False):
            return line_number, obj, "reading_text_fallback"

    # 英語 is not an independent headword in this corpus; it appears under 英.
    # Keep this fallback explicit so arbitrary example text does not cause a mis-match.
    if target == "英語":
        for line_number, obj in entries:
            entry = obj.get("data", {}).get("entry", {})
            surfaces = candidate_surfaces(entry)
            if "英" in surfaces and target in json.dumps(entry.get("sense"), ensure_ascii=False):
                return line_number, obj, "sense_text_fallback"

    return None


def build_wordlist(
    entries: list[tuple[int, dict[str, Any]]],
    targets: list[str],
    readings: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for target in targets:
        match = match_entry(target, readings[target], entries)
        if match is None:
            missing.append(target)
            continue

        line_number, original, match_type = match
        record = json.loads(json.dumps(original, ensure_ascii=False))
        entry = record.setdefault("data", {}).setdefault("entry", {})
        entry["lemma"] = target
        entry["lemma_reading"] = readings[target]
        populate_sense_definitions(entry.get("sense"))

        candidates = collect_nonempty_senses(entry.get("sense"))

        records.append(record)
        coverage_rows.append(
            {
                "target_lemma": target,
                "lemma_reading": readings[target],
                "sense_count": len(candidates),
                "match_type": match_type,
                "source_line": line_number,
                "source_path": str(record.get("path") or ""),
                "source_entry": str(entry.get("@entry") or ""),
            }
        )


    return records, coverage_rows


def write_jsonl(path, records, encoding='utf-8'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_coverage(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_lemma",
                "lemma_reading",
                "sense_count",
                "match_type",
                "source_line",
                "source_path",
                "source_entry",
            ],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    iwanami_jsonl = args.xml_root if args.xml_root is not None else args.iwanami_jsonl

    targets = (
        load_targets_from_file(args.target_list)
        if args.target_list is not None
        else load_targets_from_samples(args.samples_root)
    )
    readings = infer_readings(args.rawdata_root, targets)
    entries = load_iwanami_entries(iwanami_jsonl)
    records, coverage_rows = build_wordlist(entries, targets, readings)
    write_jsonl(args.output_jsonl, records)
    write_coverage(args.coverage_output, coverage_rows)
    print(f"Targets: {len(targets)}")
    print(f"Iwanami entries scanned: {len(entries)}")
    print(f"Wordlist JSONL: {args.output_jsonl}")
    print(f"Coverage TSV: {args.coverage_output}")


if __name__ == "__main__":
    main()
