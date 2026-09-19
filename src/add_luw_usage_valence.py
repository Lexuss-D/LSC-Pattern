#!/usr/bin/env python3
"""Annotate LUW usage JSONL files with context-level valence values."""

from __future__ import annotations

import argparse
import csv
import json
import os
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / ".local" / "inputs" / "default_input_dir"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".local" / "inputs" / "default_output_dir"
DEFAULT_PN_DICT = REPO_ROOT / ".local" / "inputs" / "default_pn_dict"
VALID_POLARITIES = {"p": 1.0, "n": -1.0, "e": 0.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute context-level valence values for LUW WSD/metaphor JSONL "
            "usage files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing JSONL files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for annotated JSONL files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--pn-dict",
        type=Path,
        default=DEFAULT_PN_DICT,
        help=f"Japanese PN dictionary path. Default: {DEFAULT_PN_DICT}",
    )
    parser.add_argument(
        "--field",
        default="valence",
        help="Output JSON field name. Default: valence",
    )
    parser.add_argument(
        "--dicdir",
        type=Path,
        help="Path to a UniDic-style MeCab dictionary directory.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files. Default: utf-8",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping them.",
    )
    return parser.parse_args()


def import_mecab():
    import MeCab  # type: ignore
    return MeCab


def default_dicdir() -> str | None:
    try:
        import unidic_lite  # type: ignore
    except ImportError:
        return None
    return str(unidic_lite.DICDIR)


def build_tagger(dicdir: Path | None):
    MeCab = import_mecab()
    selected_dicdir = str(dicdir) if dicdir else default_dicdir()
    args = []
    if selected_dicdir:
        args.extend(["-d", selected_dicdir])
    tagger = MeCab.Tagger(" ".join(args))
    tagger.parse("")
    return tagger


def iter_surface_nodes(tagger, text: str):
    node = tagger.parseToNode(text)
    while node:
        if node.surface:
            yield node
        node = node.next


def token_lemma(node) -> str:
    features = node.feature.split(",")
    pos_major = features[0] if features else ""
    if pos_major == "補助記号":
        return node.surface
    if len(features) > 7 and features[7] and features[7] != "*":
        return features[7]
    return node.surface


def load_pn_dictionary(path: Path, encoding: str) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    with path.open("r", encoding=encoding, newline="") as infile:
        reader = csv.reader(infile, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            lemma = row[0].strip()
            polarity = row[1].strip()
            score = VALID_POLARITIES.get(polarity)
            if not lemma or score is None:
                continue
            values[lemma].append(score)
    return {lemma: sum(scores) / len(scores) for lemma, scores in values.items()}


def is_punctuation_or_symbol(char: str) -> bool:
    return unicodedata.category(char).startswith(("P", "S"))


def is_punctuation_or_symbol_only(text: str) -> bool:
    stripped = "".join(ch for ch in text if not ch.isspace())
    return bool(stripped) and all(is_punctuation_or_symbol(ch) for ch in stripped)


def normalize_usage_token(token: Any) -> str:
    text = str(token).replace("#", "").strip(" \t\r\n\u3000")
    if not text or is_punctuation_or_symbol_only(text):
        return ""
    return text


def context_text(record: dict[str, Any]) -> str:
    usage = record.get("usage")
    if isinstance(usage, list):
        tokens = [normalize_usage_token(token) for token in usage]
        text = " ".join(token for token in tokens if token)
        if text:
            return text

    usage_text = record.get("usage_text")
    if isinstance(usage_text, str):
        return usage_text.replace("#", " ").strip()
    return ""


def mecab_lemmas_from_text(tagger, text: str) -> list[str]:
    lemmas: list[str] = []
    for node in iter_surface_nodes(tagger, text):
        lemma = token_lemma(node).strip()
        if not lemma or is_punctuation_or_symbol_only(lemma):
            continue
        lemmas.append(lemma)
    return lemmas


def compute_valence(
    record: dict[str, Any],
    tagger,
    pn_scores: dict[str, float],
) -> float | None:
    text = context_text(record)
    if not text:
        return None
    matched_scores = [
        pn_scores[lemma]
        for lemma in mecab_lemmas_from_text(tagger, text)
        if lemma in pn_scores
    ]
    if not matched_scores:
        return None
    return sum(matched_scores) / len(matched_scores)


def iter_input_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.jsonl"))
    return files


def annotate_file(input_path, output_path, tagger, pn_scores, field, encoding):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = valued_count = 0
    with input_path.open(encoding=encoding) as infile, output_path.open('w', encoding=encoding) as outfile:
        for line in infile:
            if not line.strip():
                continue
            record = json.loads(line)
            valence = compute_valence(record, tagger, pn_scores)
            valued_count += valence is not None
            record[field] = valence
            outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
            row_count += 1
    return row_count, valued_count


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    pn_dict = args.pn_dict.resolve()


    input_files = iter_input_files(input_dir)
    tagger = build_tagger(args.dicdir)
    pn_scores = load_pn_dictionary(pn_dict, args.encoding)

    processed_files = 0
    skipped_files = 0
    total_rows = 0
    total_valued_rows = 0

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"PN dictionary: {pn_dict}")
    print(f"Loaded PN lemmas: {len(pn_scores)}")
    print(f"Input files selected: {len(input_files)}")

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name
        if output_path.exists() and not args.overwrite:
            skipped_files += 1
            print(f"[{index}/{len(input_files)}] skip {input_path.name}")
            continue

        row_count, valued_count = annotate_file(
            input_path=input_path,
            output_path=output_path,
            encoding=args.encoding,
            field=args.field,
            tagger=tagger,
            pn_scores=pn_scores,
        )
        processed_files += 1
        total_rows += row_count
        total_valued_rows += valued_count
        print(
            f"[{index}/{len(input_files)}] done {input_path.name} "
            f"rows={row_count} valued_rows={valued_count}"
        )

    print(f"Processed files: {processed_files}")
    print(f"Skipped files: {skipped_files}")
    print(f"Total rows written: {total_rows}")
    print(f"Rows with non-null {args.field}: {total_valued_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
