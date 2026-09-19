#!/usr/bin/env python3
"""Evaluate WSD models on Iwanami human annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer


REPO_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = REPO_DIR.parent
DEFAULT_ANNOTATION_CSV = REPO_ROOT / ".local" / "inputs" / "default_annotation_csv"
DEFAULT_MATCHED_JSONL = REPO_ROOT / ".local" / "inputs" / "default_matched_jsonl"
DEFAULT_GENERATED_DIR = REPO_ROOT / ".local" / "inputs" / "default_generated_dir"
DEFAULT_WORDNET_DB = REPO_ROOT / ".local" / "inputs" / "default_wordnet_db"
DEFAULT_BEST_MODEL = REPO_ROOT / ".local" / "inputs" / "default_best_model"
DEFAULT_OUTPUT_DIR = REPO_ROOT / ".local" / "inputs" / "default_output_dir"
DEFAULT_EMBEDDING_MODEL = "sbintuitions/sarashina-embedding-v2-1b"
DEFAULT_TOHOKU_BERT_MODEL = "tohoku-nlp/bert-base-japanese-v3"
START_MARKER = "<TARGET>"
END_MARKER = "</TARGET>"
ANNOTATOR_LABEL_COLUMNS = {
    "annotator_1": "label_annotator_1",
    "annotator_2": "label_annotator_2",
    "annotator_3": "label_annotator_3",
}


@dataclass
class IwanamiSense:
    word_sheet: str
    label: int
    sense_id: str
    definition: str
    examples: str
    synset_id: str
    synset_definition: str
    similarity: str
    source: str = "matched_jsonl"


def normalize_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.fullmatch(r"(-?\d+)(?:\.0+)?", text)
    if match:
        return match.group(1)
    bracket = re.fullmatch(r"\[(-?\d+)\]", text)
    if bracket:
        return bracket.group(1)
    return text


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def entry_word_sheet(entry: dict[str, Any]) -> str | None:
    lemma = entry.get("lemma") or entry.get("orth") or entry.get("hd") or entry.get("@entry")
    reading = entry.get("lemma_reading")
    if isinstance(lemma, str) and isinstance(reading, str):
        return f"{lemma}_{reading}"
    return None


def collect_senses(node: Any, word_sheet: str, out: list[IwanamiSense]) -> None:
    if isinstance(node, dict):
        definition = node.get("sense_definition")
        if isinstance(definition, str) and definition.strip():
            out.append(
                IwanamiSense(
                    word_sheet=word_sheet,
                    label=len(out) + 1,
                    sense_id=str(node.get("@senseid") or ""),
                    definition=definition.strip(),
                    examples=str(node.get("eg_definition") or "").strip(),
                    synset_id=str(node.get("id_synset") or "").strip(),
                    synset_definition=str(node.get("sense_synset") or "").strip(),
                    similarity=str(node.get("similarity") or "").strip(),
                )
            )
        for value in node.values():
            collect_senses(value, word_sheet, out)
    elif isinstance(node, list):
        for item in node:
            collect_senses(item, word_sheet, out)


def load_iwanami_senses(path: Path) -> dict[str, list[IwanamiSense]]:
    senses_by_sheet: dict[str, list[IwanamiSense]] = {}
    for item in load_jsonl(path):
        entry = item.get("data", {}).get("entry", {})
        word_sheet = entry_word_sheet(entry)
        if not word_sheet:
            continue
        senses: list[IwanamiSense] = []
        collect_senses(entry.get("sense", {}), word_sheet, senses)
        if senses and word_sheet not in senses_by_sheet:
            senses_by_sheet[word_sheet] = senses
    return senses_by_sheet


def parse_generated_label(value: Any) -> int:
    text = normalize_label(value)
    return int(text)


def load_generated_senses(
    generated_dir: Path,
    word_sheets: set[str],
) -> dict[str, list[IwanamiSense]]:
    senses_by_sheet: dict[str, list[IwanamiSense]] = {}
    for word_sheet in sorted(word_sheets):
        path = generated_dir / f"{word_sheet}_n30.jsonl"
        if not path.exists():
            continue
        candidates = None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_candidates = row.get("wsd_candidate_synsets") or []
                if row_candidates:
                    candidates = row_candidates
                    break
        if not candidates:
            continue
        senses: list[IwanamiSense] = []
        seen_labels: set[int] = set()
        for candidate in candidates:
            definition = str(candidate.get("definition") or "").strip()
            if not definition:
                continue
            label = parse_generated_label(candidate.get("sense_index"))
            if label in seen_labels:
                continue
            seen_labels.add(label)
            senses.append(
                IwanamiSense(
                    word_sheet=word_sheet,
                    label=label,
                    sense_id=str(candidate.get("senseid") or "").strip(),
                    definition=definition,
                    examples=str(candidate.get("examples") or "").strip(),
                    synset_id=str(candidate.get("synset_id") or "").strip(),
                    synset_definition=str(candidate.get("synset_definition") or "").strip(),
                    similarity=str(candidate.get("similarity") or "").strip(),
                    source="generated",
                )
            )
        if senses:
            senses_by_sheet[word_sheet] = sorted(senses, key=lambda sense: sense.label)
    return senses_by_sheet


def merge_sense_sources(
    matched_senses: dict[str, list[IwanamiSense]],
    generated_senses: dict[str, list[IwanamiSense]],
    word_sheets: set[str],
) -> tuple[dict[str, list[IwanamiSense]], dict[str, Any]]:
    merged = dict(matched_senses)
    audit_rows = []
    status_counts: Counter[str] = Counter()
    generated_overrides = 0
    generated_fills = 0
    generated_diffs = 0

    def signature(senses: list[IwanamiSense]) -> list[tuple[int, str, str, str]]:
        return [
            (sense.label, sense.sense_id, sense.definition, sense.synset_id)
            for sense in senses
        ]

    for word_sheet in sorted(word_sheets):
        matched = matched_senses.get(word_sheet, [])
        generated = generated_senses.get(word_sheet, [])
        status = "matched_only"
        if generated:
            generated_overrides += 1
            if not matched:
                generated_fills += 1
                status = "generated_fill"
            elif signature(matched) != signature(generated):
                generated_diffs += 1
                status = "generated_override_diff"
            else:
                status = "generated_override_same"
            merged[word_sheet] = generated
        elif matched:
            status = "matched_only"
        else:
            status = "missing_candidates"
        status_counts[status] += 1
        audit_rows.append(
            {
                "word_sheet": word_sheet,
                "matched_candidate_count": len(matched),
                "generated_candidate_count": len(generated),
                "final_candidate_count": len(merged.get(word_sheet, [])),
                "status": status,
                "final_source": merged.get(word_sheet, [IwanamiSense(word_sheet, 0, "", "", "", "", "", "")])[0].source
                if merged.get(word_sheet)
                else None,
                "final_wordnet_linked_candidate_count": sum(
                    1 for sense in merged.get(word_sheet, []) if sense.synset_id
                ),
            }
        )

    audit = {
        "annotation_word_sheets": len(word_sheets),
        "generated_overrides": generated_overrides,
        "generated_fills": generated_fills,
        "generated_diffs": generated_diffs,
        "status_counts": dict(sorted(status_counts.items())),
        "rows": audit_rows,
    }
    return merged, audit


def load_annotations(path: Path) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        majority = normalize_label(record.get("majority_label"))
        record["gold_label"] = int(majority) if majority else None
        for column in ANNOTATOR_LABEL_COLUMNS.values():
            label = normalize_label(record.get(column))
            record[column] = int(label) if label else None
        record["usage_context"] = (
            str(record.get("left_context") or "")
            + str(record.get("target") or "")
            + str(record.get("right_context") or "")
        )
        record["usage_marked"] = (
            str(record.get("left_context") or "")
            + "[["
            + str(record.get("target") or "")
            + "]]"
            + str(record.get("right_context") or "")
        )
        rows.append(record)
    return rows


def filter_individual_annotator_labels(
    annotations: list[dict[str, Any]],
    senses_by_sheet: dict[str, list[IwanamiSense]],
) -> dict[str, Any]:
    invalid_rows = []
    for row in annotations:
        valid_labels = {0}
        valid_labels.update(
            sense.label for sense in senses_by_sheet.get(row["word_sheet"], [])
        )
        for annotator, column in ANNOTATOR_LABEL_COLUMNS.items():
            label = row.get(column)
            if label is None or label in valid_labels:
                continue
            invalid_rows.append(
                {
                    "source_id": row.get("source_id"),
                    "word_sheet": row.get("word_sheet"),
                    "annotator": annotator,
                    "column": column,
                    "invalid_label": label,
                    "valid_labels": sorted(valid_labels),
                }
            )
            row[column] = None
    return {
        "invalid_label_count": len(invalid_rows),
        "invalid_rows": invalid_rows,
        "policy": "Labels outside {0} plus the word_sheet candidate labels are excluded from annotator-specific metrics.",
    }


def fetch_wordnet_glosses(db_path: Path) -> dict[str, str]:
    import sqlite3

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    by_synset: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    query = """
    select synset, lang, sid, def
    from synset_def
    where lang in ('jpn', 'eng')
    order by synset, lang, cast(sid as integer), sid
    """
    for row in con.execute(query):
        text = (row["def"] or "").strip()
        if not text:
            continue
        sid = str(row["sid"])
        sid_number = int(sid) if sid.isdigit() else 0
        by_synset[row["synset"]][row["lang"]].append((sid_number, text))
    glosses = {}
    for synset_id, by_lang in by_synset.items():
        for lang in ("jpn", "eng"):
            values = by_lang.get(lang, [])
            if values:
                glosses[synset_id] = sorted(values)[0][1]
                break
    return glosses


def crop_usage(text: str, start: int, end: int, max_chars: int) -> tuple[str, list[int]]:
    if len(text) <= max_chars:
        return text, [start, end]
    target_len = end - start
    room = max(max_chars - target_len, 0)
    crop_start = max(0, start - room // 2)
    crop_end = min(len(text), crop_start + max_chars)
    crop_start = max(0, crop_end - max_chars)
    return text[crop_start:crop_end], [start - crop_start, end - crop_start]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def encode_bert_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    content_mode: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    vectors = []
    excluded = {
        token_id
        for token_id in (
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
        )
        if token_id is not None
    }
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                hidden = model(**inputs).last_hidden_state
            ids = inputs["input_ids"]
            if content_mode == "non_special":
                mask = torch.ones_like(ids, dtype=torch.bool)
                for token_id in excluded:
                    mask &= ids != token_id
            else:
                raise ValueError(f"Unsupported content_mode: {content_mode}")
            vectors.append(mean_pool(hidden, mask).detach().float().cpu())
    return torch.cat(vectors, dim=0)


def encode_embedding_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    vectors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                output = model(**inputs)
            hidden = output.last_hidden_state
            mask = inputs["attention_mask"].bool()
            vectors.append(mean_pool(hidden, mask).detach().float().cpu())
    return F.normalize(torch.cat(vectors, dim=0), dim=-1)


def encode_sentence_transformer_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=str(device),
    )
    return embeddings.detach().float().cpu()


def encode_bert_target_usages(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_length: int,
    max_chars: int,
    device: torch.device,
) -> torch.Tensor:
    start_marker_id = tokenizer.convert_tokens_to_ids(START_MARKER)
    end_marker_id = tokenizer.convert_tokens_to_ids(END_MARKER)
    marked_texts = []
    for row in rows:
        left = str(row.get("left_context") or "")
        target = str(row.get("target") or "")
        right = str(row.get("right_context") or "")
        text = left + target + right
        usage, span = crop_usage(text, len(left), len(left) + len(target), max_chars)
        s, e = span
        marked_texts.append(f"{usage[:s]}{START_MARKER}{usage[s:e]}{END_MARKER}{usage[e:]}")

    vectors = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(marked_texts), batch_size):
            batch = marked_texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            ids = inputs["input_ids"].tolist()
            masks = torch.zeros_like(inputs["input_ids"], dtype=torch.bool)
            for row_index, token_ids in enumerate(ids):
                s = token_ids.index(start_marker_id)
                e = token_ids.index(end_marker_id)
                masks[row_index, s + 1 : e] = True
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                hidden = model(**inputs).last_hidden_state
            vectors.append(mean_pool(hidden, masks.to(device)).detach().float().cpu())
    return torch.cat(vectors, dim=0)


def cosine_scores(query: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    return F.normalize(query, dim=-1) @ F.normalize(candidates, dim=-1).T


def candidate_senses_for_word(
    senses_by_sheet: dict[str, list[IwanamiSense]],
    word_sheet: str,
    require_synset: bool,
) -> list[IwanamiSense]:
    senses = senses_by_sheet.get(word_sheet, [])
    if require_synset:
        return [sense for sense in senses if sense.synset_id]
    return senses


def compute_label_metrics(
    predictions: list[dict[str, Any]],
    label_key: str,
) -> dict[str, Any]:
    labeled = [row for row in predictions if row.get(label_key) is not None]
    covered = [row for row in labeled if row.get("predicted_label") is not None]
    nonzero = [row for row in covered if row.get(label_key) != 0]
    gold_zero = [row for row in covered if row.get(label_key) == 0]

    def acc(rows: list[dict[str, Any]]) -> float | None:
        if not rows:
            return None
        return sum(
            1 for row in rows if row.get("predicted_label") == row.get(label_key)
        ) / len(rows)

    by_corpus = {}
    for corpus, rows in group_by(covered, "corpus").items():
        by_corpus[corpus] = {"n": len(rows), "accuracy": acc(rows)}
    by_word = {}
    for word, rows in group_by(covered, "word_sheet").items():
        by_word[word] = {"n": len(rows), "accuracy": acc(rows)}

    return {
        "total_rows": len(predictions),
        "labeled_rows": len(labeled),
        "covered_rows": len(covered),
        "uncovered_labeled_rows": len(labeled) - len(covered),
        "gold_zero_rows": len(gold_zero),
        "gold_nonzero_rows": len(nonzero),
        "accuracy_all_labeled": acc(covered),
        "accuracy_nonzero_gold": acc(nonzero),
        "accuracy_gold_zero": acc(gold_zero),
        "by_corpus": by_corpus,
        "by_word": by_word,
        "gold_label_counts": dict(sorted(Counter(str(row.get(label_key)) for row in labeled).items())),
        "predicted_label_counts": dict(sorted(Counter(str(row.get("predicted_label")) for row in covered).items())),
    }


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = compute_label_metrics(predictions, "gold_label")
    individual = {
        annotator: compute_label_metrics(predictions, column)
        for annotator, column in ANNOTATOR_LABEL_COLUMNS.items()
    }

    def macro(metric_key: str) -> float | None:
        values = [item[metric_key] for item in individual.values() if item[metric_key] is not None]
        return sum(values) / len(values) if values else None

    metrics["individual_annotators"] = individual
    metrics["macro_average_over_annotators"] = {
        "annotator_count": len(individual),
        "accuracy_all_labeled": macro("accuracy_all_labeled"),
        "accuracy_nonzero_gold": macro("accuracy_nonzero_gold"),
        "accuracy_gold_zero": macro("accuracy_gold_zero"),
    }
    return metrics


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return grouped


def run_experiment1(
    annotations: list[dict[str, Any]],
    senses_by_sheet: dict[str, list[IwanamiSense]],
    wordnet_glosses: dict[str, str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.best_model)
    model = AutoModel.from_pretrained(args.best_model).to(device)

    usage_vectors = encode_bert_target_usages(
        model,
        tokenizer,
        annotations,
        batch_size=args.bert_usage_batch_size,
        max_length=args.bert_usage_max_length,
        max_chars=args.bert_usage_max_chars,
        device=device,
    )

    unique_synsets = sorted(
        {
            sense.synset_id
            for senses in senses_by_sheet.values()
            for sense in senses
            if sense.synset_id and wordnet_glosses.get(sense.synset_id)
        }
    )
    gloss_texts = [wordnet_glosses[synset_id] for synset_id in unique_synsets]
    gloss_vectors = encode_bert_texts(
        model,
        tokenizer,
        gloss_texts,
        content_mode="non_special",
        batch_size=args.bert_gloss_batch_size,
        max_length=args.bert_gloss_max_length,
        device=device,
    )
    synset_to_index = {synset_id: idx for idx, synset_id in enumerate(unique_synsets)}

    predictions = []
    duplicate_synset_mappings = 0
    for row_index, row in enumerate(annotations):
        candidates = [
            sense
            for sense in candidate_senses_for_word(senses_by_sheet, row["word_sheet"], True)
            if sense.synset_id in synset_to_index
        ]
        if not candidates:
            predictions.append(base_prediction_row(row, "experiment1", None))
            continue
        cand_indexes = [synset_to_index[sense.synset_id] for sense in candidates]
        scores = cosine_scores(usage_vectors[row_index : row_index + 1], gloss_vectors[cand_indexes])[0]
        best_pos = int(torch.argmax(scores).item())
        best_sense = candidates[best_pos]
        if sum(1 for sense in candidates if sense.synset_id == best_sense.synset_id) > 1:
            duplicate_synset_mappings += 1
        pred = base_prediction_row(row, "experiment1", best_sense)
        pred.update(
            {
                "predicted_synset_id": best_sense.synset_id,
                "predicted_synset_definition": wordnet_glosses.get(best_sense.synset_id, ""),
                "predicted_score": float(scores[best_pos].item()),
                "candidate_count": len(candidates),
                "candidate_scores": [
                    {
                        "label": sense.label,
                        "sense_id": sense.sense_id,
                        "synset_id": sense.synset_id,
                        "score": float(scores[i].item()),
                    }
                    for i, sense in enumerate(candidates)
                ],
            }
        )
        predictions.append(pred)

    metrics = compute_metrics(predictions)
    metrics["unique_candidate_synsets_encoded"] = len(unique_synsets)
    metrics["duplicate_synset_mapping_predictions"] = duplicate_synset_mappings
    return predictions, metrics


def base_prediction_row(
    row: dict[str, Any],
    experiment: str,
    predicted_sense: IwanamiSense | None,
) -> dict[str, Any]:
    predicted_label = predicted_sense.label if predicted_sense is not None else None
    gold = row.get("gold_label")
    return {
        "experiment": experiment,
        "source_id": row.get("source_id"),
        "word_sheet": row.get("word_sheet"),
        "target": row.get("target"),
        "corpus": row.get("corpus"),
        "timestamp": row.get("timestamp"),
        "line_no": row.get("line_no"),
        "left_context": row.get("left_context"),
        "right_context": row.get("right_context"),
        "usage_marked": row.get("usage_marked"),
        "gold_label": gold,
        **{
            column: row.get(column)
            for column in ANNOTATOR_LABEL_COLUMNS.values()
        },
        "predicted_label": predicted_label,
        "correct": (predicted_label == gold) if gold is not None and predicted_label is not None else None,
        "predicted_iwanami_sense_id": predicted_sense.sense_id if predicted_sense else None,
        "predicted_iwanami_definition": predicted_sense.definition if predicted_sense else None,
        "predicted_iwanami_examples": predicted_sense.examples if predicted_sense else None,
    }


def run_experiment2(
    annotations: list[dict[str, Any]],
    senses_by_sheet: dict[str, list[IwanamiSense]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = SentenceTransformer(args.embedding_model, device=str(device))
    usage_texts = [row["usage_context"] for row in annotations]
    usage_vectors = encode_sentence_transformer_texts(
        model,
        usage_texts,
        batch_size=args.embedding_batch_size,
        device=device,
    )

    all_sense_keys: list[tuple[str, int]] = []
    all_sense_texts = []
    for word_sheet, senses in sorted(senses_by_sheet.items()):
        for sense in senses:
            all_sense_keys.append((word_sheet, sense.label))
            text = sense.definition
            if sense.examples:
                text = f"{text} {sense.examples}"
            all_sense_texts.append(text)
    sense_vectors = encode_sentence_transformer_texts(
        model,
        all_sense_texts,
        batch_size=args.embedding_batch_size,
        device=device,
    )
    sense_key_to_index = {key: idx for idx, key in enumerate(all_sense_keys)}

    predictions = []
    for row_index, row in enumerate(annotations):
        candidates = candidate_senses_for_word(senses_by_sheet, row["word_sheet"], False)
        if not candidates:
            predictions.append(base_prediction_row(row, "experiment2", None))
            continue
        cand_indexes = [sense_key_to_index[(sense.word_sheet, sense.label)] for sense in candidates]
        scores = usage_vectors[row_index] @ sense_vectors[cand_indexes].T
        best_pos = int(torch.argmax(scores).item())
        best_sense = candidates[best_pos]
        pred = base_prediction_row(row, "experiment2", best_sense)
        pred.update(
            {
                "predicted_score": float(scores[best_pos]),
                "candidate_count": len(candidates),
                "candidate_scores": [
                    {
                        "label": sense.label,
                        "sense_id": sense.sense_id,
                        "score": float(scores[i].item()),
                    }
                    for i, sense in enumerate(candidates)
                ],
            }
        )
        predictions.append(pred)

    metrics = compute_metrics(predictions)
    metrics["unique_iwanami_senses_encoded"] = len(all_sense_keys)
    metrics["embedding_model"] = args.embedding_model
    metrics["embedding_pooling"] = "SentenceTransformer encode, normalized embeddings"
    return predictions, metrics


def run_experiment2_tohoku_bert(
    annotations: list[dict[str, Any]],
    senses_by_sheet: dict[str, list[IwanamiSense]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(args.tohoku_bert_model)
    tokenizer.add_special_tokens({"additional_special_tokens": [START_MARKER, END_MARKER]})
    model = AutoModel.from_pretrained(args.tohoku_bert_model)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    usage_vectors = encode_bert_target_usages(
        model,
        tokenizer,
        annotations,
        batch_size=args.bert_usage_batch_size,
        max_length=args.bert_usage_max_length,
        max_chars=args.bert_usage_max_chars,
        device=device,
    )

    all_sense_keys: list[tuple[str, int]] = []
    all_sense_texts = []
    for word_sheet, senses in sorted(senses_by_sheet.items()):
        for sense in senses:
            if not sense.definition:
                continue
            all_sense_keys.append((word_sheet, sense.label))
            all_sense_texts.append(sense.definition)

    sense_vectors = encode_bert_texts(
        model,
        tokenizer,
        all_sense_texts,
        content_mode="non_special",
        batch_size=args.bert_gloss_batch_size,
        max_length=args.bert_gloss_max_length,
        device=device,
    )
    sense_key_to_index = {key: idx for idx, key in enumerate(all_sense_keys)}

    predictions = []
    for row_index, row in enumerate(annotations):
        candidates = [
            sense
            for sense in candidate_senses_for_word(senses_by_sheet, row["word_sheet"], False)
            if (sense.word_sheet, sense.label) in sense_key_to_index
        ]
        if not candidates:
            predictions.append(base_prediction_row(row, "experiment2_tohoku_bert", None))
            continue
        cand_indexes = [sense_key_to_index[(sense.word_sheet, sense.label)] for sense in candidates]
        scores = cosine_scores(usage_vectors[row_index : row_index + 1], sense_vectors[cand_indexes])[0]
        best_pos = int(torch.argmax(scores).item())
        best_sense = candidates[best_pos]
        pred = base_prediction_row(row, "experiment2_tohoku_bert", best_sense)
        pred.update(
            {
                "predicted_score": float(scores[best_pos].item()),
                "candidate_count": len(candidates),
                "candidate_scores": [
                    {
                        "label": sense.label,
                        "sense_id": sense.sense_id,
                        "sense_definition": sense.definition,
                        "score": float(scores[i].item()),
                    }
                    for i, sense in enumerate(candidates)
                ],
            }
        )
        predictions.append(pred)

    metrics = compute_metrics(predictions)
    metrics["unique_iwanami_senses_encoded"] = len(all_sense_keys)
    metrics["embedding_model"] = args.tohoku_bert_model
    metrics["embedding_pooling"] = (
        "usage target-token mean pooling with <TARGET> markers; "
        "gloss sense_definition non-special-token mean pooling"
    )
    metrics["candidate_space"] = "Iwanami candidate senses within each word_sheet"
    return predictions, metrics


def read_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_existing_evaluation(
    prediction_path: Path,
    metrics_path: Path,
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not prediction_path.exists():
        return read_metrics(metrics_path)
    annotation_by_source = {row["source_id"]: row for row in annotations}
    predictions = load_jsonl(prediction_path)
    for prediction in predictions:
        annotation = annotation_by_source.get(prediction.get("source_id"))
        prediction["gold_label"] = annotation.get("gold_label")
        for column in ANNOTATOR_LABEL_COLUMNS.values():
            prediction[column] = annotation.get(column)
        predicted = prediction.get("predicted_label")
        gold = prediction.get("gold_label")
        prediction["correct"] = (
            predicted == gold if predicted is not None and gold is not None else None
        )

    old_metrics = read_metrics(metrics_path)
    metrics = compute_metrics(predictions)
    for key, value in old_metrics.items():
        if key not in metrics:
            metrics[key] = value
    write_jsonl(prediction_path, predictions)
    write_csv(prediction_path.with_suffix(".csv"), predictions)
    write_json(metrics_path, metrics)
    return metrics


def write_report(
    output_dir: Path,
    experiments: list[tuple[str, str, dict[str, Any]]],
    data_stats: dict[str, Any],
) -> None:
    def pct(value: float | None) -> str:
        return "NA" if value is None else f"{value * 100:.2f}%"

    available = [(key, name, metrics) for key, name, metrics in experiments if metrics]
    lines = [
        "# Iwanami WSD Evaluation Results",
        "",
        "## Data",
        "",
        f"- Total rows: {data_stats['total_rows']}",
        f"- Rows with majority label: {data_stats['labeled_rows']}",
        f"- Gold label 0 rows: {data_stats['gold_zero_rows']}",
        f"- Gold nonzero rows: {data_stats['gold_nonzero_rows']}",
        f"- Candidate source status: {json.dumps(data_stats.get('candidate_source_status_counts', {}), ensure_ascii=False)}",
        f"- Invalid individual annotator labels excluded: {data_stats.get('invalid_individual_annotator_labels', 0)}",
        "",
        "## Main Metrics",
        "",
        "| Experiment | Covered rows | Covered accuracy | Accuracy nonzero gold | Gold 0 accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, name, metrics in available:
        lines.append(
            f"| {name} | {metrics['covered_rows']} | {pct(metrics['accuracy_all_labeled'])} | "
            f"{pct(metrics['accuracy_nonzero_gold'])} | {pct(metrics['accuracy_gold_zero'])} |"
        )

    lines.extend(
        [
            "",
            "## Macro Average over Individual Annotators",
            "",
            "| Experiment | annotator_1 | annotator_2 | annotator_3 | Macro accuracy | Macro nonzero accuracy |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, name, metrics in available:
        individual = metrics.get("individual_annotators", {})
        macro = metrics.get("macro_average_over_annotators", {})

        def annotator_cell(annotator: str) -> str:
            item = individual.get(annotator, {})
            return f"{pct(item.get('accuracy_all_labeled'))} (n={item.get('covered_rows', 0)})"

        lines.append(
            f"| {name} | {annotator_cell('annotator_1')} | {annotator_cell('annotator_2')} | "
            f"{annotator_cell('annotator_3')} | {pct(macro.get('accuracy_all_labeled'))} | "
            f"{pct(macro.get('accuracy_nonzero_gold'))} |"
        )

    lines.extend(
        [
            "",
            "## Corpus Breakdown",
            "",
            "| Experiment | Corpus | N | Accuracy |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for _, name, metrics in available:
        for corpus, item in sorted(metrics["by_corpus"].items()):
            lines.append(f"| {name} | {corpus} | {item['n']} | {pct(item['accuracy'])} |")
    lines.extend(
        [
            "",
            "## Experiment 2 Variants",
            "",
            "| Variant | Model | Usage embedding | Gloss embedding | Candidate space |",
            "| --- | --- | --- | --- | --- |",
            "| Previous Experiment 2 | `sbintuitions/sarashina-embedding-v2-1b` | SentenceTransformer embedding over the whole usage context | SentenceTransformer embedding over `sense_definition` plus examples | Iwanami candidate senses in the same `word_sheet` |",
            "| Revised Experiment 2 | `tohoku-nlp/bert-base-japanese-v3` | Mean pooling over target tokens between `<TARGET>` and `</TARGET>` | Mean pooling over `sense_definition` tokens only | Iwanami candidate senses in the same `word_sheet` |",
            "",
            "## Notes",
            "",
            "- Annotator macro accuracy is the unweighted arithmetic mean of the three annotator-specific accuracies, each computed on rows labeled by that annotator and covered by the model.",
            "- Individual annotator labels outside label 0 and the available Iwanami candidate labels are treated as invalid and excluded; details are stored in `individual_annotator_label_audit.json`.",
            "- Experiment 1 cannot predict label 0 or candidates without WordNet synsets because it selects among WordNet-linked Iwanami senses only.",
            "- Experiment 2 variants also select among Iwanami senses 1..n and therefore cannot predict label 0.",
        ]
    )
    (output_dir / "evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATION_CSV)
    parser.add_argument("--matched-jsonl", type=Path, default=DEFAULT_MATCHED_JSONL)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_GENERATED_DIR)
    parser.add_argument("--wordnet-db", type=Path, default=DEFAULT_WORDNET_DB)
    parser.add_argument("--best-model", type=Path, default=DEFAULT_BEST_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--tohoku-bert-model", default=DEFAULT_TOHOKU_BERT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bert-usage-batch-size", type=int, default=256)
    parser.add_argument("--bert-gloss-batch-size", type=int, default=512)
    parser.add_argument("--bert-usage-max-length", type=int, default=256)
    parser.add_argument("--bert-gloss-max-length", type=int, default=128)
    parser.add_argument("--bert-usage-max-chars", type=int, default=420)
    parser.add_argument("--embedding-batch-size", type=int, default=256)
    parser.add_argument("--embedding-max-length", type=int, default=256)
    parser.add_argument("--skip-experiment1", action="store_true")
    parser.add_argument("--skip-experiment2", action="store_true")
    parser.add_argument("--skip-experiment2-tohoku", action="store_true")
    parser.add_argument("--only-experiment2-tohoku", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.only_experiment2_tohoku:
        args.skip_experiment1 = True
        args.skip_experiment2 = True
        args.skip_experiment2_tohoku = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")

    annotations = load_annotations(args.annotations)
    annotation_word_sheets = {row["word_sheet"] for row in annotations}
    matched_senses = load_iwanami_senses(args.matched_jsonl)
    generated_senses = load_generated_senses(args.generated_dir, annotation_word_sheets)
    senses_by_sheet, candidate_audit = merge_sense_sources(
        matched_senses,
        generated_senses,
        annotation_word_sheets,
    )
    annotator_label_audit = filter_individual_annotator_labels(
        annotations,
        senses_by_sheet,
    )
    wordnet_glosses = fetch_wordnet_glosses(args.wordnet_db)

    labeled = [row for row in annotations if row["gold_label"] is not None]
    data_stats = {
        "total_rows": len(annotations),
        "labeled_rows": len(labeled),
        "gold_zero_rows": sum(1 for row in labeled if row["gold_label"] == 0),
        "gold_nonzero_rows": sum(1 for row in labeled if row["gold_label"] != 0),
        "word_sheets_in_annotations": len({row["word_sheet"] for row in annotations}),
        "word_sheets_with_iwanami_senses": len(senses_by_sheet),
        "candidate_source_status_counts": candidate_audit["status_counts"],
        "invalid_individual_annotator_labels": annotator_label_audit["invalid_label_count"],
    }
    write_json(args.output_dir / "data_stats.json", data_stats)
    write_json(args.output_dir / "candidate_audit.json", candidate_audit)
    write_json(
        args.output_dir / "individual_annotator_label_audit.json",
        annotator_label_audit,
    )

    metrics1: dict[str, Any] = read_metrics(args.output_dir / "experiment1_metrics.json")
    metrics2: dict[str, Any] = read_metrics(args.output_dir / "experiment2_metrics.json")
    metrics2_tohoku: dict[str, Any] = read_metrics(args.output_dir / "experiment2_tohoku_bert_metrics.json")
    if not args.skip_experiment1:
        predictions1, metrics1 = run_experiment1(
            annotations,
            senses_by_sheet,
            wordnet_glosses,
            args,
            device,
        )
        write_jsonl(args.output_dir / "experiment1_wordnet_predictions.jsonl", predictions1)
        write_csv(args.output_dir / "experiment1_wordnet_predictions.csv", predictions1)
        write_json(args.output_dir / "experiment1_metrics.json", metrics1)

    if not args.skip_experiment2:
        predictions2, metrics2 = run_experiment2(
            annotations,
            senses_by_sheet,
            args,
            device,
        )
        write_jsonl(args.output_dir / "experiment2_gloss_retrieval_predictions.jsonl", predictions2)
        write_csv(args.output_dir / "experiment2_gloss_retrieval_predictions.csv", predictions2)
        write_json(args.output_dir / "experiment2_metrics.json", metrics2)

    if not args.skip_experiment2_tohoku:
        predictions2_tohoku, metrics2_tohoku = run_experiment2_tohoku_bert(
            annotations,
            senses_by_sheet,
            args,
            device,
        )
        write_jsonl(args.output_dir / "experiment2_tohoku_bert_predictions.jsonl", predictions2_tohoku)
        write_csv(args.output_dir / "experiment2_tohoku_bert_predictions.csv", predictions2_tohoku)
        write_json(args.output_dir / "experiment2_tohoku_bert_metrics.json", metrics2_tohoku)

    metrics1 = refresh_existing_evaluation(
        args.output_dir / "experiment1_wordnet_predictions.jsonl",
        args.output_dir / "experiment1_metrics.json",
        annotations,
    )
    metrics2 = refresh_existing_evaluation(
        args.output_dir / "experiment2_gloss_retrieval_predictions.jsonl",
        args.output_dir / "experiment2_metrics.json",
        annotations,
    )
    metrics2_tohoku = refresh_existing_evaluation(
        args.output_dir / "experiment2_tohoku_bert_predictions.jsonl",
        args.output_dir / "experiment2_tohoku_bert_metrics.json",
        annotations,
    )

    write_report(
        args.output_dir,
        [
            ("experiment1", "Experiment 1: fine-tuned BERT WordNet cosine -> Iwanami", metrics1),
            ("experiment2", "Experiment 2 previous: Sarashina gloss retrieval baseline", metrics2),
            ("experiment2_tohoku", "Experiment 2 revised: Tohoku BERT target/gloss retrieval", metrics2_tohoku),
        ],
        data_stats,
    )
    print(
        json.dumps(
            {
                "data_stats": data_stats,
                "experiment1": metrics1,
                "experiment2": metrics2,
                "experiment2_tohoku_bert": metrics2_tohoku,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
