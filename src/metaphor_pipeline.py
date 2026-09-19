#!/usr/bin/env python3
"""Run-artifact helper for WSD usage metaphor inference.

This file intentionally lives under the new output run directory. It does not
modify project source code. The helper prepares README-compatible raw inputs,
and merges predictions produced by the existing release
package scripts.
"""

from __future__ import annotations

import argparse
import os
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("LSCP_PROJECT_ROOT", str(REPO_ROOT / ".local" / "inputs" / "project_root")))
WSD_RUN = Path(os.environ.get("LSCP_WSD_RUN", str(REPO_ROOT / ".local" / "inputs" / "wsd_run")))
GENERATED_DIR = Path(os.environ.get("LSCP_GENERATED_DIR", str(REPO_ROOT / ".local" / "inputs" / "generated_dir")))
WORDNET_MATCHED = Path(os.environ.get("LSCP_WORDNET_MATCHED", str(REPO_ROOT / ".local" / "inputs" / "wordnet_matched")))
BUNRUI_DATA = Path(os.environ.get("LSCP_BUNRUI_DATA", str(REPO_ROOT / ".local" / "inputs" / "bunrui_data")))
PREPROCESS_SCRIPT = Path(os.environ.get("LSCP_PREPROCESS_SCRIPT", str(REPO_ROOT / ".local" / "inputs" / "preprocess_script")))

CORPUS_CONFIG = json.loads(os.environ.get("LSCP_CORPUS_CONFIG", "{}"))

# Keep aligned with data/metaphor/release_package/preprocess_for_inference.py.
TARGET_POS = {
    "名詞-普通名詞-一般",
    "名詞-数詞",
    "名詞-助動詞語幹",
    "名詞-普通名詞-サ変可能",
    "名詞-普通名詞-サ変形状詞可能",
    "名詞-普通名詞-形状詞可能",
    "名詞-普通名詞-副詞可能",
    "名詞-普通名詞-助数詞可能",
    "名詞-固有名詞-一般",
    "名詞-固有名詞-地名-一般",
    "名詞-固有名詞-地名-国",
    "名詞-固有名詞-人名-一般",
    "名詞-固有名詞-人名-名",
    "名詞-固有名詞-人名-姓",
    "動詞-一般",
    "動詞-非自立可能",
    "形容詞-一般",
    "形容詞-非自立可能",
    "副詞",
    "形状詞-一般",
    "形状詞-助動詞語幹",
    "形状詞-タリ",
    "連体詞",
    "代名詞",
    "英単語 ",
    "漢文",
}

UID_PREFIX_RE = re.compile(r"[^A-Za-z0-9_.-]+")
WINDOW_RE = re.compile(r"_n(\d+)\.jsonl$")


@dataclass(frozen=True)
class TokenMeta:
    surface: str
    lemma: str
    reading: str
    pos: str
    timestamp: str


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            yield item


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json_dumps(row) + "\n")
            count += 1
    return count


def corpus_from_filename(path: Path) -> str:
    name = path.name
    for corpus in CORPUS_CONFIG:
        if name.startswith(f"{corpus}_"):
            return corpus
    raise ValueError(f"Cannot infer corpus from generated filename: {path.name}")


def parse_window_size(path: Path) -> int:
    match = WINDOW_RE.search(path.name)
    return int(match.group(1))


def sanitize_uid_part(text: str) -> str:
    return UID_PREFIX_RE.sub("_", text).strip("_") or "EMPTY"


def stable_relpath_hash(relpath: Path) -> str:
    return hashlib.sha1(str(relpath).encode("utf-8")).hexdigest()[:12]


def predicted_senseid(row: dict[str, Any]) -> str:
    pred = row.get("predicted_sense_index")
    if pred in (None, 0, "0"):
        return ""
    for candidate in row.get("wsd_candidate_synsets", []) or []:
        if candidate.get("sense_index") == pred:
            return str(candidate.get("senseid") or "")
    return ""


def iter_generated_files(generated_dir: Path, include_files: list[str]) -> list[Path]:
    files = sorted(generated_dir.glob("*.jsonl"))
    if not include_files:
        return files
    wanted = set(include_files)
    selected = [
        path for path in files
        if path.name in wanted or str(path) in wanted or str(path.relative_to(generated_dir)) in wanted
    ]
    return selected


def cmd_build_manifest(args: argparse.Namespace) -> int:
    generated_dir = args.generated_dir.resolve()
    manifest_output = args.manifest_output.resolve()
    summary_output = args.summary_output.resolve()
    log_output = args.log_output.resolve()

    files = iter_generated_files(generated_dir, args.include_file)

    counters = Counter()
    by_corpus = Counter()

    def rows() -> Iterable[dict[str, Any]]:
        for path in files:
            rel = path.relative_to(generated_dir)
            corpus = corpus_from_filename(path)
            window_size = parse_window_size(path)
            for line_number, row in enumerate(read_jsonl(path), start=1):
                counters["records"] += 1
                by_corpus[corpus] += 1
                usage = row.get("usage")
                target_index = row.get("target_token_index")
                target_surface = row.get("target_surface")

                lemma = str(row.get("lemma") or "")
                reading = str(row.get("lemma_reading") or "")
                source_usage_row = int(row.get("source_usage_row") or line_number)
                uid = f"{sanitize_uid_part(corpus)}__{stable_relpath_hash(rel)}__{source_usage_row:08d}"
                yield {
                    "metaphor_uid": uid,
                    "source_generated_relpath": str(rel),
                    "source_generated_line": line_number,
                    "source_usage_file": row.get("source_usage_file"),
                    "source_usage_row": source_usage_row,
                    "corpus": corpus,
                    "window_size": window_size,
                    "timestamp": row.get("timestamp"),
                    "lemma": lemma,
                    "lemma_reading": reading,
                    "target_token_index": target_index,
                    "target_surface": target_surface,
                    "predicted_sense_index": row.get("predicted_sense_index"),
                    "predicted_senseid": predicted_senseid(row),
                    "predicted_synset_id": row.get("predicted_synset_id"),
                    "usage": usage,
                }

    manifest_count = write_jsonl(manifest_output, rows())
    counters["files"] = len(files)
    counters["manifest_records"] = manifest_count

    summary = {
        "generated_dir": str(generated_dir),
        "manifest_output": str(manifest_output),
        "file_count": len(files),
        "record_count": manifest_count,
        "by_corpus": dict(by_corpus),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_output.parent.mkdir(parents=True, exist_ok=True)
    with log_output.open("w", encoding="utf-8") as log:
        log.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")

    print(f"Wrote manifest: {manifest_output} ({manifest_count} records)")
    return 0


def parse_wlsp_info(value: Any) -> list[dict[str, Any]]:
    if value in (None, "", "[]", []):
        return []
    raw_items: Any
    if isinstance(value, str):
        try:
            raw_items = json.loads(value)
        except json.JSONDecodeError:
            raw_items = [value]
    else:
        raw_items = value
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    parsed: list[dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, str) or not item.strip():
            continue
        parts = item.split(",")
        bunnrui_number = parts[7].strip() if len(parts) > 7 else ""
        if not bunnrui_number:
            continue
        parsed.append({"bunnrui_number": bunnrui_number, "wlsp_record": item})
    return parsed


def iter_sense_nodes(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        if "@senseid" in node:
            yield node
        child = node.get("sense")
        if isinstance(child, list):
            for item in child:
                yield from iter_sense_nodes(item)
        elif isinstance(child, dict):
            yield from iter_sense_nodes(child)


def build_wordnet_sense_index(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for obj in read_jsonl(path):
        entry = obj.get("data", {}).get("entry", {})
        if not isinstance(entry, dict):
            continue
        lemma = str(entry.get("lemma") or "")
        reading = str(entry.get("lemma_reading") or "")
        for sense in iter_sense_nodes(entry.get("sense")):
            senseid = str(sense.get("@senseid") or "")
            if lemma and reading and senseid:
                index[(lemma, reading, senseid)] = sense
    return index


def build_bunnrui_fallback(path: Path) -> dict[str, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    fallback: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in data:
        lemma = entry.get("lemma")
        if not lemma:
            continue
        nums = entry.get("bunnrui_numbers", []) or []
        groups = entry.get("bunnrui_group", []) or []
        confs = entry.get("basic_meaning_confidence_level", []) or []
        for idx, num in enumerate(nums):
            fallback[str(lemma)].append(
                {
                    "bunnrui_number": str(num),
                    "bunnrui_group": groups[idx] if idx < len(groups) else None,
                    "confidence": float(confs[idx]) if idx < len(confs) else 0.0,
                }
            )
    return fallback


def resolve_bunnrui(
    record: dict[str, Any],
    sense_index: dict[tuple[str, str, str], dict[str, Any]],
    fallback: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    lemma = str(record.get("lemma") or "")
    reading = str(record.get("lemma_reading") or "")
    senseid = str(record.get("predicted_senseid") or "")

    if senseid:
        sense = sense_index.get((lemma, reading, senseid))
        if sense is not None:
            wlsp_candidates = parse_wlsp_info(sense.get("wlsp_info"))
            if wlsp_candidates:
                return {
                    "bunnrui_number": wlsp_candidates[0]["bunnrui_number"],
                    "source": "wlsp_info",
                    "candidates": wlsp_candidates,
                    "senseid": senseid,
                }
            ref_candidates = parse_wlsp_info(sense.get("wlsp_info_ref"))
            if ref_candidates:
                return {
                    "bunnrui_number": ref_candidates[0]["bunnrui_number"],
                    "source": "wlsp_info_ref",
                    "candidates": ref_candidates,
                    "senseid": senseid,
                }

    fallback_candidates = fallback.get(lemma, [])
    if fallback_candidates:
        best = max(fallback_candidates, key=lambda item: item.get("confidence", 0.0))
        return {
            "bunnrui_number": best["bunnrui_number"],
            "source": "bunnrui_data",
            "candidates": fallback_candidates,
            "senseid": senseid,
        }
    return {"bunnrui_number": "nan", "source": "nan", "candidates": [], "senseid": senseid}


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return list(read_jsonl(path))


def csv_row_to_meta(row: dict[str, str], config: dict[str, Any]) -> TokenMeta:
    return TokenMeta(
        surface=row.get(config["surface"], ""),
        lemma=row.get(config["lemma"], ""),
        reading=row.get(config["reading"], ""),
        pos=row.get(config["pos"], ""),
        timestamp=row.get(config["timestamp"], ""),
    )


class ShardWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_index = -1
        self.count_in_shard = 0
        self.total = 0
        self.handle = None
        self.paths: list[Path] = []

    def _open_next(self) -> None:
        if self.handle is not None:
            self.handle.write("\n]\n")
            self.handle.close()
        self.shard_index += 1
        self.count_in_shard = 0
        path = self.output_dir / f"usage_targets_raw_shard_{self.shard_index:03d}.json"
        self.paths.append(path)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8")
        self.handle.write("[\n")

    def write(self, item: dict[str, Any]) -> None:
        if self.handle is None or self.count_in_shard >= self.shard_size:
            self._open_next()
        if self.count_in_shard:
            self.handle.write(",\n")
        self.handle.write(json_dumps(item))
        self.count_in_shard += 1
        self.total += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.write("\n]\n")
            self.handle.close()
            self.handle = None


def make_raw_sentence(record: dict[str, Any], target_meta: TokenMeta, bunnrui: dict[str, Any]) -> dict[str, Any]:
    tokens = [str(token) for token in record["usage"]]
    target_index = int(record["target_token_index"])
    lemmas = [str(token) for token in tokens]
    pos_tags = ["UNKNOWN"] * len(tokens)
    bunnrui_numbers = ["nan"] * len(tokens)
    lemmas[target_index] = str(record["lemma"])
    pos_tags[target_index] = target_meta.pos
    bunnrui_numbers[target_index] = bunnrui["bunnrui_number"]
    return {
        "id": record["metaphor_uid"],
        "tokens": tokens,
        "lemmas": lemmas,
        "bunnrui_numbers": bunnrui_numbers,
        "pos_tags": pos_tags,
    }


def cmd_build_raw(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest.resolve())
    raw_dir = args.raw_dir.resolve()
    metadata_output = args.metadata_output.resolve()
    summary_output = args.summary_output.resolve()
    log_output = args.log_output.resolve()

    sense_index = build_wordnet_sense_index(args.wordnet_matched.resolve())
    fallback = build_bunnrui_fallback(args.bunnrui_data.resolve())

    records_by_lookup: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    wanted_rows: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for record in manifest:
        key = (record["corpus"], record["lemma"], record["lemma_reading"])
        source_row = int(record["source_usage_row"])
        wanted_rows[key].add(source_row)
        records_by_lookup[(record["corpus"], record["lemma"], record["lemma_reading"], source_row)] = record

    counters = Counter()
    by_bunnrui_source = Counter()
    metadata_by_uid: list[dict[str, Any]] = []
    unsupported_pos_samples: list[dict[str, Any]] = []
    emitted_uids: set[str] = set()
    shard_writer = ShardWriter(raw_dir, args.shard_size)

    try:
        for corpus, config in CORPUS_CONFIG.items():
            keys_for_corpus = {
                (lemma, reading)
                for (rec_corpus, lemma, reading), rows in wanted_rows.items()
                if rec_corpus == corpus and rows
            }
            if not keys_for_corpus:
                continue

            csv_path = Path(config["csv"])
            print(f"Scanning {corpus}: {csv_path}", flush=True)
            previous: deque[TokenMeta] = deque(maxlen=args.context_window_size)
            pending: list[dict[str, Any]] = []
            occurrence_count: Counter[tuple[str, str]] = Counter()

            with csv_path.open("r", encoding="utf-8", newline="") as infile:
                reader = csv.DictReader(infile)

                for csv_line, row in enumerate(reader, start=2):
                    meta = csv_row_to_meta(row, config)

                    still_pending: list[dict[str, Any]] = []
                    for pending_item in pending:
                        pending_item["after"].append(meta)
                        remaining = int(pending_item["remaining"]) - 1
                        if remaining == 0:
                            emit_raw_record(
                                pending_item,
                                records_by_lookup,
                                sense_index,
                                fallback,
                                shard_writer,
                                metadata_by_uid,
                                emitted_uids,
                                counters,
                                by_bunnrui_source,
                                unsupported_pos_samples,
                            )
                        else:
                            pending_item["remaining"] = remaining
                            still_pending.append(pending_item)
                    pending = still_pending

                    key_lr = (meta.lemma, meta.reading)
                    if key_lr in keys_for_corpus:
                        occurrence_count[key_lr] += 1
                        source_usage_row = occurrence_count[key_lr]
                        full_key = (corpus, meta.lemma, meta.reading)
                        if source_usage_row in wanted_rows.get(full_key, set()):
                            if args.context_window_size == 0:
                                pending_item = {
                                    "corpus": corpus,
                                    "lemma": meta.lemma,
                                    "reading": meta.reading,
                                    "source_usage_row": source_usage_row,
                                    "before": [],
                                    "target": meta,
                                    "after": [],
                                }
                                emit_raw_record(
                                    pending_item,
                                    records_by_lookup,
                                    sense_index,
                                    fallback,
                                    shard_writer,
                                    metadata_by_uid,
                                    emitted_uids,
                                    counters,
                                    by_bunnrui_source,
                                    unsupported_pos_samples,
                                )
                            else:
                                pending.append(
                                    {
                                        "corpus": corpus,
                                        "lemma": meta.lemma,
                                        "reading": meta.reading,
                                        "source_usage_row": source_usage_row,
                                        "before": list(previous),
                                        "target": meta,
                                        "after": [],
                                        "remaining": args.context_window_size,
                                    }
                                )

                    previous.append(meta)
                    if csv_line % 1_000_000 == 0:
                        print(f"{corpus}: scanned {csv_line:,} CSV lines", flush=True)

            for pending_item in pending:
                emit_raw_record(
                    pending_item,
                    records_by_lookup,
                    sense_index,
                    fallback,
                    shard_writer,
                    metadata_by_uid,
                    emitted_uids,
                    counters,
                    by_bunnrui_source,
                    unsupported_pos_samples,
                )
    finally:
        shard_writer.close()

    missing_records = [record for record in manifest if record["metaphor_uid"] not in emitted_uids]
    counters["missing_metadata_count"] = len(missing_records)
    counters["manifest_records"] = len(manifest)
    counters["raw_records"] = shard_writer.total

    write_jsonl(metadata_output, metadata_by_uid)
    summary = {
        "manifest_records": len(manifest),
        "raw_records": shard_writer.total,
        "raw_shards": [str(path) for path in shard_writer.paths],
        "metadata_records": len(metadata_by_uid),
        "unsupported_target_pos_count": counters["unsupported_target_pos_count"],
        "missing_metadata_count": counters["missing_metadata_count"],
        "bunnrui_source_counts": dict(by_bunnrui_source),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_output.parent.mkdir(parents=True, exist_ok=True)
    with log_output.open("w", encoding="utf-8") as log:
        log.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        if unsupported_pos_samples:
            log.write("\nFirst unsupported target POS:\n")
            for item in unsupported_pos_samples:
                log.write(json_dumps(item) + "\n")
        if missing_records:
            log.write("\nFirst missing metadata records:\n")
            for item in missing_records[:100]:
                log.write(json_dumps(item) + "\n")

    print(f"Wrote raw shards to {raw_dir} ({shard_writer.total} records)")
    return 0


def emit_raw_record(
    pending_item: dict[str, Any],
    records_by_lookup: dict[tuple[str, str, str, int], dict[str, Any]],
    sense_index: dict[tuple[str, str, str], dict[str, Any]],
    fallback: dict[str, list[dict[str, Any]]],
    shard_writer: ShardWriter,
    metadata_by_uid: list[dict[str, Any]],
    emitted_uids: set[str],
    counters: Counter,
    by_bunnrui_source: Counter,
    unsupported_pos_samples: list[dict[str, Any]],
) -> None:
    corpus = pending_item["corpus"]
    lemma = pending_item["lemma"]
    reading = pending_item["reading"]
    source_usage_row = int(pending_item["source_usage_row"])
    record = records_by_lookup.get((corpus, lemma, reading, source_usage_row))
    if record is None:
        return

    metas = list(pending_item["before"]) + [pending_item["target"]] + list(pending_item["after"])
    target_index = int(record["target_token_index"])

    target_meta = metas[target_index]

    bunnrui = resolve_bunnrui(record, sense_index, fallback)
    by_bunnrui_source[bunnrui["source"]] += 1
    metadata_status = "ok"
    if target_meta.pos not in TARGET_POS:
        metadata_status = "unsupported_target_pos"
        counters["unsupported_target_pos_count"] += 1
        if len(unsupported_pos_samples) < 100:
            unsupported_pos_samples.append(
                {
                    "metaphor_uid": record["metaphor_uid"],
                    "corpus": corpus,
                    "lemma": record["lemma"],
                    "target_surface": record["target_surface"],
                    "target_pos": target_meta.pos,
                }
            )

    raw_sentence = make_raw_sentence(record, target_meta, bunnrui)
    shard_writer.write(raw_sentence)
    emitted_uids.add(record["metaphor_uid"])
    metadata_by_uid.append(
        {
            "metaphor_uid": record["metaphor_uid"],
            "source_generated_relpath": record["source_generated_relpath"],
            "source_generated_line": record["source_generated_line"],
            "corpus": corpus,
            "target_pos": target_meta.pos,
            "metaphor_target_bunnrui_number": bunnrui["bunnrui_number"],
            "metaphor_bunnrui_source": bunnrui["source"],
            "metaphor_bunnrui_candidates": bunnrui["candidates"],
            "metaphor_metadata_status": metadata_status,
            "predicted_senseid": record.get("predicted_senseid"),
        }
    )


def cmd_merge(args: argparse.Namespace) -> int:
    generated_dir = args.generated_dir.resolve()
    predictions_dir = args.predictions_dir.resolve()
    manifest = load_manifest(args.manifest.resolve())
    metadata_by_uid = {row["metaphor_uid"]: row for row in read_jsonl(args.metadata.resolve())}
    output_dir = args.output_dir.resolve()
    summary_output = args.summary_output.resolve()
    log_output = args.log_output.resolve()

    prediction_by_uid: dict[str, int] = {}
    for pred_path in sorted(predictions_dir.glob("predictions_shard_*.json")):
        with pred_path.open("r", encoding="utf-8") as infile:
            predictions = json.load(infile)
        for row in predictions:
            uid = row.get("original_sentence", {}).get("id")
            if uid:
                prediction_by_uid[str(uid)] = int(row.get("prediction", -1))

    uid_by_file_line = {
        (row["source_generated_relpath"], int(row["source_generated_line"])): row["metaphor_uid"]
        for row in manifest
    }
    relpaths = sorted({row["source_generated_relpath"] for row in manifest})
    counters = Counter()
    by_prediction = Counter()
    by_corpus_prediction = defaultdict(Counter)

    output_dir.mkdir(parents=True, exist_ok=True)
    for relpath in relpaths:
        input_path = generated_dir / relpath
        output_path = output_dir / relpath
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
            for line_number, line in enumerate(infile, start=1):
                if not line.strip():
                    continue
                uid = uid_by_file_line.get((relpath, line_number))
                if uid is None:
                    continue
                row = json.loads(line)
                pred = prediction_by_uid.get(uid, -1)
                metadata = metadata_by_uid.get(uid, {})
                row["metaphor_prediction"] = pred
                row["metaphor_label"] = "metaphor" if pred == 1 else "literal" if pred == 0 else "error"
                row["metaphor_model"] = "bert-base-japanese-v3/fold_4_best"
                row["metaphor_target_bunnrui_number"] = metadata.get("metaphor_target_bunnrui_number", "nan")
                row["metaphor_bunnrui_source"] = metadata.get("metaphor_bunnrui_source", "nan")
                row["metaphor_bunnrui_candidates"] = metadata.get("metaphor_bunnrui_candidates", [])
                row["metaphor_metadata_status"] = metadata.get("metaphor_metadata_status", "missing_metadata")
                outfile.write(json_dumps(row) + "\n")
                counters["output_records"] += 1
                by_prediction[str(pred)] += 1
                by_corpus_prediction[str(row.get("corpus"))][str(pred)] += 1

    counters["manifest_records"] = len(manifest)
    counters["prediction_records"] = len(prediction_by_uid)
    counters["output_files"] = len(relpaths)
    summary = {
        "manifest_records": len(manifest),
        "prediction_records": len(prediction_by_uid),
        "output_records": counters["output_records"],
        "output_files": counters["output_files"],
        "prediction_counts": dict(by_prediction),
        "prediction_counts_by_corpus": {k: dict(v) for k, v in by_corpus_prediction.items()},
        "output_dir": str(output_dir),
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log_output.parent.mkdir(parents=True, exist_ok=True)
    log_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote merged JSONL files to {output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and merge WSD metaphor inference artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-manifest")
    p.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    p.add_argument("--manifest-output", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--log-output", type=Path, required=True)
    p.add_argument("--include-file", action="append", default=[])
    p.set_defaults(func=cmd_build_manifest)

    p = sub.add_parser("build-raw")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--metadata-output", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--log-output", type=Path, required=True)
    p.add_argument("--wordnet-matched", type=Path, default=WORDNET_MATCHED)
    p.add_argument("--bunnrui-data", type=Path, default=BUNRUI_DATA)
    p.add_argument("--context-window-size", type=int, default=30)
    p.add_argument("--shard-size", type=int, default=50000)
    p.set_defaults(func=cmd_build_raw)

    p = sub.add_parser("merge")
    p.add_argument("--generated-dir", type=Path, default=GENERATED_DIR)
    p.add_argument("--predictions-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--summary-output", type=Path, required=True)
    p.add_argument("--log-output", type=Path, required=True)
    p.set_defaults(func=cmd_merge)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
