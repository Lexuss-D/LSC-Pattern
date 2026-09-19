#!/usr/bin/env python3
"""Build LUW WSD prompt JSONL files from Iwanami senses and usage JSONL files."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORDLIST = REPO_ROOT / ".local" / "inputs" / "default_wordlist"
DEFAULT_USAGES_ROOT = REPO_ROOT / ".local" / "inputs" / "default_usages_root"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / ".local" / "inputs" / "default_output_root"
DEFAULT_SUMMARY = REPO_ROOT / ".local" / "inputs" / "default_summary"
CORPORA = ("BCCWJ_all", "CHJ_all", "SHC_all")
PROMPT_TEMPLATE = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build flat JSONL WSD prompt files for LUW usage examples."
    )
    parser.add_argument(
        "--wordlist",
        type=Path,
        default=DEFAULT_WORDLIST,
        help=f"Iwanami wordlist JSONL. Default: {DEFAULT_WORDLIST}",
    )
    parser.add_argument(
        "--usages-root",
        type=Path,
        default=DEFAULT_USAGES_ROOT,
        help=f"Root directory containing LUW usage JSONL files. Default: {DEFAULT_USAGES_ROOT}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Directory for flat prompt JSONL files. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY,
        help=f"Summary TSV path. Default: {DEFAULT_SUMMARY}",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files. Default: utf-8",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing prompt files.",
    )
    parser.add_argument(
        "--write-empty-files",
        action="store_true",
        help="Write empty prompt JSONL files for empty usage files.",
    )
    return parser.parse_args()


def normalize_lemma(text: str) -> str:
    return text.split("-", 1)[0].strip()


def sanitize_filename_part(value: str) -> str:
    if value == "":
        return "EMPTY"
    sanitized = re.sub(r'[\\/:\*\?"<>\|\x00-\x1f]+', "_", value)
    sanitized = sanitized.strip(". ")
    return sanitized or "EMPTY"


def parse_usage_filename(path: Path) -> tuple[str, str, int]:
    parts = path.stem.rsplit("_", 2)
    lemma, reading, window_text = parts
    window_size = int(window_text[1:])
    return lemma, reading, window_size


def iter_usage_files(usages_root: Path) -> list[Path]:
    paths: list[Path] = []
    for corpus in CORPORA:
        corpus_dir = usages_root / corpus
        paths.extend(sorted(corpus_dir.glob("*.jsonl")))
    return paths


def collect_texts(value: Any) -> list[str]:
    texts: list[str] = []
    if isinstance(value, dict):
        text = value.get("#text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
        for child in value.values():
            if child is not text:
                texts.extend(collect_texts(child))
    elif isinstance(value, list):
        for item in value:
            texts.extend(collect_texts(item))
    elif isinstance(value, str) and value.strip():
        texts.append(value)
    return texts


def collect_eg_texts(value: Any) -> list[str]:
    if isinstance(value, dict) and "su" in value:
        return collect_texts(value.get("su"))
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(collect_eg_texts(item))
        return texts
    return collect_texts(value)


def collect_nonempty_senses(node: Any) -> list[dict[str, str]]:
    collected: list[dict[str, str]] = []

    def visit(current: Any) -> None:
        if not isinstance(current, dict):
            return

        sense_definition = str(current.get("sense_definition") or "").strip()
        if not sense_definition:
            sense_definition = "".join(collect_texts(current.get("su"))).strip()

        eg_definition = str(current.get("eg_definition") or "").strip()
        if not eg_definition:
            eg_definition = "|".join(collect_eg_texts(current.get("eg"))).strip()

        if sense_definition:
            collected.append(
                {
                    "senseid": str(current.get("@senseid") or ""),
                    "pos": str(current.get("pos") or ""),
                    "definition": sense_definition,
                    "examples": eg_definition,
                    "synset_id": str(
                        current.get("id_synset")
                        or current.get("matched_synset_id")
                        or ""
                    ),
                    "synset_definition": str(
                        current.get("sense_synset")
                        or current.get("matched_synset_definition")
                        or ""
                    ),
                    "depth_to_root": str(current.get("depth_to_root") or ""),
                }
            )

        child = current.get("sense")
        if isinstance(child, list):
            for item in child:
                visit(item)
        elif isinstance(child, dict):
            visit(child)

    visit(node)
    return collected


def sense_dedupe_key(sense: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        sense["senseid"],
        sense["definition"],
        sense["examples"],
        sense["synset_id"],
        sense["synset_definition"],
    )


def load_sense_index(
    wordlist: Path,
    encoding: str,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_by_key: dict[tuple[str, str], set[tuple[str, str, str, str, str]]] = {}

    with wordlist.open("r", encoding=encoding) as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)

            entry = obj.get("data", {}).get("entry", {})
            if not isinstance(entry, dict):
                continue
            lemma = entry.get("lemma")
            reading = entry.get("lemma_reading")
            if not isinstance(lemma, str) or not isinstance(reading, str):
                continue

            key = (normalize_lemma(lemma), reading)
            index.setdefault(key, [])
            seen_by_key.setdefault(key, set())
            for sense in collect_nonempty_senses(entry.get("sense")):
                dedupe_key = sense_dedupe_key(sense)
                if dedupe_key in seen_by_key[key]:
                    continue
                seen_by_key[key].add(dedupe_key)
                sense_index = len(index[key]) + 1
                candidate = {
                    "sense_index": sense_index,
                    "senseid": sense["senseid"],
                    "pos": sense["pos"],
                    "definition": sense["definition"],
                    "examples": sense["examples"],
                    "synset_id": sense["synset_id"],
                    "synset_definition": sense["synset_definition"],
                    "depth_to_root": sense["depth_to_root"],
                }
                index[key].append(candidate)

    return index


def render_senses(candidates: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for candidate in candidates:
        lines.append(
            f"[{candidate['sense_index']}]\t品詞：{candidate['pos']}\t"
            f"語義：{candidate['definition']}\t例文：「{candidate['examples']}」"
        )
    return "\n".join(lines)


def choose_target_token_index(tokens: list[str], lemma: str, window_size: int) -> int:
    exact_positions = [index for index, token in enumerate(tokens) if token == lemma]
    if exact_positions:
        return min(exact_positions, key=lambda index: (abs(index - window_size), index))
    if 0 <= window_size < len(tokens):
        return window_size
    return len(tokens) // 2 if tokens else 0


def render_usage(tokens: list[str], target_index: int) -> tuple[str, str]:
    rendered_tokens = [str(token) for token in tokens]
    if rendered_tokens:
        target_index = min(max(target_index, 0), len(rendered_tokens) - 1)
        rendered_tokens[target_index] = f"[[{rendered_tokens[target_index]}]]"
    usage_text = "".join(str(token) for token in tokens)
    usage_marked_text = "".join(rendered_tokens)
    return usage_text, usage_marked_text


def build_prompt(usage_marked_text: str, candidates: list[dict[str, Any]]) -> str:
    return load_private_template().format(
        usage=usage_marked_text,
        senses=render_senses(candidates),
    )


def output_path_for_usage(input_path: Path, output_root: Path) -> Path:
    corpus = input_path.parent.name
    lemma, reading, window_size = parse_usage_filename(input_path)
    output_name = (
        f"{sanitize_filename_part(corpus)}_"
        f"{sanitize_filename_part(lemma)}_"
        f"{sanitize_filename_part(reading)}_"
        f"n{window_size}.jsonl"
    )
    return output_root / output_name


def read_usage_records(path: Path, encoding: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding=encoding) as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            usage = record.get("usage")
            records.append(record)
    return records


def write_jsonl(path, records, encoding='utf-8'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding=encoding) as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')


def build_prompt_records(
    usage_records: list[dict[str, Any]],
    corpus: str,
    lemma: str,
    reading: str,
    window_size: int,
    candidates: list[dict[str, Any]],
    source_path: Path,
) -> list[dict[str, Any]]:
    prompt_records: list[dict[str, Any]] = []
    for row_index, record in enumerate(usage_records, start=1):
        tokens = [str(token) for token in record["usage"]]
        target_index = choose_target_token_index(tokens, lemma, window_size)
        target_surface = tokens[target_index] if tokens else ""
        usage_text, usage_marked_text = render_usage(tokens, target_index)
        prompt = build_prompt(usage_marked_text, candidates)

        annotated = dict(record)
        annotated["corpus"] = corpus
        annotated["lemma"] = lemma
        annotated["lemma_reading"] = reading
        annotated["usage_text"] = usage_text
        annotated["usage_marked_text"] = usage_marked_text
        annotated["target_token_index"] = target_index
        annotated["target_surface"] = target_surface
        annotated["source_usage_file"] = str(source_path)
        annotated["source_usage_row"] = row_index
        annotated["wsd_candidate_synsets"] = [dict(candidate) for candidate in candidates]
        annotated["prompt"] = prompt
        prompt_records.append(annotated)
    return prompt_records


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    encoding: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "corpus",
        "lemma",
        "lemma_reading",
        "usage_file",
        "output_file",
        "usage_count",
        "prompt_rows",
        "sense_count",
        "wsd_senseids",
        "empty_usage",
        "skipped_existing",
    ]
    with path.open("w", encoding=encoding, newline="") as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def main() -> None:
    args = parse_args()

    sense_index = load_sense_index(args.wordlist, args.encoding)
    usage_files = iter_usage_files(args.usages_root)
    summary_rows: list[dict[str, Any]] = []
    processed_files = 0
    skipped_files = 0
    total_prompt_rows = 0

    for file_index, usage_path in enumerate(usage_files, start=1):
        corpus = usage_path.parent.name
        lemma, reading, window_size = parse_usage_filename(usage_path)
        key = (normalize_lemma(lemma), reading)
        candidates = sense_index.get(key)

        output_path = output_path_for_usage(usage_path, args.output_root)
        usage_records = read_usage_records(usage_path, args.encoding)
        empty_usage = not usage_records
        skipped_existing = output_path.exists() and not args.overwrite

        prompt_rows = 0
        if skipped_existing:
            skipped_files += 1
        elif empty_usage and not args.write_empty_files:
            skipped_files += 1
        else:
            prompt_records = build_prompt_records(
                usage_records=usage_records,
                corpus=corpus,
                lemma=lemma,
                reading=reading,
                window_size=window_size,
                candidates=candidates,
                source_path=usage_path,
            )
            write_jsonl(output_path, prompt_records, args.encoding)
            prompt_rows = len(prompt_records)
            total_prompt_rows += prompt_rows
            processed_files += 1

        summary_rows.append(
            {
                "corpus": corpus,
                "lemma": lemma,
                "lemma_reading": reading,
                "usage_file": str(usage_path),
                "output_file": str(output_path),
                "usage_count": len(usage_records),
                "prompt_rows": prompt_rows,
                "sense_count": len(candidates),
                "wsd_senseids": json.dumps(
                    [candidate["senseid"] for candidate in candidates],
                    ensure_ascii=False,
                ),
                "empty_usage": int(empty_usage),
                "skipped_existing": int(skipped_existing),
            }
        )

        print(
            f"[{file_index}/{len(usage_files)}] "
            f"{'skip' if skipped_existing or (empty_usage and not args.write_empty_files) else 'done'} "
            f"{usage_path.name} usages={len(usage_records)} senses={len(candidates)}"
        )

    write_summary(args.summary_output, summary_rows, args.encoding)
    print(f"Usage files considered: {len(usage_files)}")
    print(f"Prompt files written: {processed_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Prompt rows written: {total_prompt_rows}")
    print(f"Summary TSV: {args.summary_output}")


def load_private_template():
    return Path(os.environ['LSCP_PROMPT_TEMPLATE']).read_text(encoding='utf-8').strip()

if __name__ == "__main__":
    main()
