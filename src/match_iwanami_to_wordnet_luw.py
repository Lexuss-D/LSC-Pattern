#!/usr/bin/env python3
"""Match LUW Iwanami sense definitions to Japanese WordNet synsets."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / ".local" / "inputs" / "default_input"
DEFAULT_WORDNET_DB = REPO_ROOT / ".local" / "inputs" / "default_wordnet_db"
DEFAULT_OUTPUT = REPO_ROOT / ".local" / "inputs" / "default_output"
DEFAULT_MODEL = "sbintuitions/sarashina-embedding-v2-1b"
DEFAULT_DEVICE = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Match every recursive LUW Iwanami sense_definition to the most similar "
            "Japanese WordNet synset for the same lemma."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input Iwanami JSONL. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--wordnet-db",
        type=Path,
        default=DEFAULT_WORDNET_DB,
        help=f"Japanese WordNet SQLite DB. Default: {DEFAULT_WORDNET_DB}",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model id. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device passed to SentenceTransformer. Default: auto",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding. Default: 32",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=50,
        help="Maximum number of nodes while walking from a synset to the root.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for input and output files. Default: utf-8",
    )
    return parser.parse_args()


def import_numpy():
    import numpy as np  # type: ignore
    return np


def import_sentence_transformers():
    from sentence_transformers import SentenceTransformer  # type: ignore
    return SentenceTransformer


def import_torch():
    import torch  # type: ignore
    return torch


def resolve_device(device: str) -> str:
    torch = import_torch()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_jsonl(path: Path, encoding: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding=encoding) as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            items.append(obj)
    return items


def write_jsonl(path: Path, items: list[dict[str, Any]], encoding: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as outfile:
        for item in items:
            outfile.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalize_lemma(text: str) -> str:
    return text.split("-", 1)[0].strip()


def first_or_empty(values: list[str]) -> str:
    return values[0] if values else ""


def unique_nonempty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        stripped = str(value).strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


def build_hierarchy_dict(conn: sqlite3.Connection) -> dict[str, list[str]]:
    hierarchy: dict[str, list[str]] = defaultdict(list)
    cur = conn.execute("select synset1, synset2 from synlink where link='hypo'")
    for parent_synset, child_synset in cur:
        hierarchy[str(child_synset)].append(str(parent_synset))
    return dict(hierarchy)


def pick_depth_to_root_from_paths(paths: list[dict[str, Any]]) -> str | int:
    root_depths = [
        path.get("depth_to_root")
        for path in paths
        if isinstance(path, dict)
        and path.get("terminal_type") == "root"
        and isinstance(path.get("depth_to_root"), int)
    ]
    if root_depths:
        return min(root_depths)

    any_depths = [
        path.get("depth_to_root")
        for path in paths
        if isinstance(path, dict) and isinstance(path.get("depth_to_root"), int)
    ]
    return min(any_depths) if any_depths else ""


def format_paths(paths: tuple[tuple[str, ...], ...]) -> list[dict[str, Any]]:
    formatted_paths: list[dict[str, Any]] = []
    for raw_path in paths:
        terminal_type = "root"
        synset_path = list(raw_path)
        if synset_path and synset_path[-1] == "(max_depth reached)":
            terminal_type = "max_depth"
            synset_path = synset_path[:-1]
        elif synset_path and synset_path[-1] == "(cycle detected)":
            terminal_type = "cycle"
            synset_path = synset_path[:-1]

        formatted_paths.append(
            {
                "depth_to_root": max(len(synset_path) - 1, 0),
                "root_synset_id": synset_path[-1] if synset_path else "",
                "terminal_type": terminal_type,
                "synset_path": synset_path,
            }
        )
    return formatted_paths


def collect_sense_nodes(node: Any) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []

    def visit(current: Any) -> None:
        if not isinstance(current, dict):
            return
        nodes.append(current)
        child = current.get("sense")
        if isinstance(child, list):
            for item in child:
                visit(item)
        elif isinstance(child, dict):
            visit(child)

    visit(node)
    return nodes


def query_text(entry: dict[str, Any], sense: dict[str, Any]) -> str:
    lemma = normalize_lemma(str(entry.get("lemma", "")))
    orth = str(entry.get("orth", ""))
    hd = str(entry.get("hd", ""))
    lemma_reading = str(entry.get("lemma_reading", ""))
    sense_id = str(sense.get("@senseid", ""))
    gloss = str(sense.get("sense_definition", "")).strip()
    return (
        "task: 岩波国語辞典の語義に最も近い日本語WordNetの定義を検索してください。\n"
        f"query: 見出し語: {lemma}\n"
        f"表記: {orth}\n"
        f"読み: {hd or lemma_reading}\n"
        f"senseid: {sense_id}\n"
        f"語義: {gloss}"
    )


def candidate_text(lemma: str, synset: dict[str, Any]) -> str:
    defs_jpn = [str(x).strip() for x in synset.get("definitions_jpn", []) if str(x).strip()]
    defs_eng = [str(x).strip() for x in synset.get("definitions_eng", []) if str(x).strip()]
    definition = "；".join(defs_jpn) if defs_jpn else "；".join(defs_eng)
    if not definition:
        definition = str(synset.get("definition", "")).strip() or str(synset.get("name", ""))
    return (
        f"text: 見出し語: {lemma}\n"
        f"synset: {synset.get('synset_id', '')}\n"
        f"name: {synset.get('name', '')}\n"
        f"定義: {definition}"
    )


def update_sense_with_match(
    sense: dict[str, Any],
    matched_synset: dict[str, Any] | None,
    similarity: float | None,
) -> None:
    if not matched_synset:
        sense["id_synset"] = ""
        sense["sense_synset"] = ""
        sense["similarity"] = ""
        sense["depth_to_root"] = ""
        sense["num_layers"] = ""
        return

    depth_to_root = pick_depth_to_root_from_paths(matched_synset.get("paths_to_root", []))
    sense["id_synset"] = str(matched_synset.get("synset_id", ""))
    sense["sense_synset"] = str(matched_synset.get("definition", ""))
    sense["similarity"] = round(float(similarity), 6) if similarity is not None else ""
    sense["depth_to_root"] = depth_to_root
    sense["num_layers"] = depth_to_root


def main() -> None:
    args = parse_args()
    np = import_numpy()
    SentenceTransformer = import_sentence_transformers()
    device = resolve_device(args.device)

    items = load_jsonl(args.input, args.encoding)
    conn = sqlite3.connect(args.wordnet_db)
    hierarchy_dict = build_hierarchy_dict(conn)

    @lru_cache(maxsize=None)
    def get_synset_name(synset: str) -> str:
        cur = conn.execute("select name from synset where synset=?", (synset,))
        row = cur.fetchone()
        return str(row[0]) if row else synset

    @lru_cache(maxsize=None)
    def get_synset_defs(synset: str, lang: str) -> tuple[str, ...]:
        cur = conn.execute(
            "select def from synset_def where synset=? and lang=?",
            (synset, lang),
        )
        return tuple(unique_nonempty([row[0] for row in cur.fetchall()]))

    @lru_cache(maxsize=None)
    def get_synset_info(synset: str) -> dict[str, Any]:
        defs_jpn = list(get_synset_defs(synset, "jpn"))
        defs_eng = list(get_synset_defs(synset, "eng"))
        return {
            "synset_id": synset,
            "name": get_synset_name(synset),
            "definition": first_or_empty(defs_jpn) or first_or_empty(defs_eng),
            "definitions_jpn": defs_jpn,
            "definitions_eng": defs_eng,
        }

    @lru_cache(maxsize=None)
    def get_word_rows(lemma: str) -> tuple[tuple[Any, ...], ...]:
        cur = conn.execute(
            """
            select distinct w.wordid, w.lang, w.lemma, w.pron, w.pos
            from word as w
            where w.lang='jpn' and w.lemma=?
            order by w.wordid
            """,
            (lemma,),
        )
        return tuple(cur.fetchall())

    @lru_cache(maxsize=None)
    def get_synset_rows_for_lemma(lemma: str) -> tuple[tuple[Any, ...], ...]:
        cur = conn.execute(
            """
            select distinct se.synset, w.wordid, w.pos, w.pron
            from word as w
            join sense as se on se.wordid = w.wordid
            where w.lang='jpn' and se.lang='jpn' and w.lemma=?
            order by se.synset, w.wordid
            """,
            (lemma,),
        )
        return tuple(cur.fetchall())

    @lru_cache(maxsize=None)
    def enumerate_hypernym_paths(start_synset: str) -> tuple[tuple[str, ...], ...]:
        paths: list[tuple[str, ...]] = []
        stack: list[tuple[str, tuple[str, ...], frozenset[str]]] = [
            (start_synset, (start_synset,), frozenset([start_synset]))
        ]

        while stack:
            current_synset, path, visited = stack.pop()
            parents = hierarchy_dict.get(current_synset, [])
            if not parents:
                paths.append(path)
                continue

            if len(path) >= args.max_depth:
                paths.append(path + ("(max_depth reached)",))
                continue

            for parent in parents:
                if parent in visited:
                    paths.append(path + (parent, "(cycle detected)"))
                    continue
                stack.append((parent, path + (parent,), visited | {parent}))

        paths.sort()
        return tuple(paths)

    @lru_cache(maxsize=None)
    def build_synset_entries(lemma: str) -> tuple[dict[str, Any], ...]:
        rows_by_synset: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for synset_id, wordid, pos, pron in get_synset_rows_for_lemma(lemma):
            rows_by_synset[str(synset_id)].append((synset_id, wordid, pos, pron))

        entries: list[dict[str, Any]] = []
        for synset_id in sorted(rows_by_synset):
            rows = rows_by_synset[synset_id]
            paths = format_paths(enumerate_hypernym_paths(synset_id))
            synset_info = get_synset_info(synset_id)
            entries.append(
                {
                    "synset_id": synset_id,
                    "name": synset_info["name"],
                    "definition": synset_info["definition"],
                    "definitions_jpn": synset_info["definitions_jpn"],
                    "definitions_eng": synset_info["definitions_eng"],
                    "wordids": sorted({row[1] for row in rows}),
                    "poses": sorted({row[2] or "" for row in rows}),
                    "prons": sorted({row[3] or "" for row in rows}),
                    "depth_to_root": pick_depth_to_root_from_paths(paths),
                    "paths_to_root": paths,
                }
            )
        return tuple(entries)

    print(f"Loaded {len(items)} JSONL entries from {args.input}.")
    print(f"Using device: {device}")
    model = SentenceTransformer(args.model, device=device)

    candidate_texts: list[str] = []
    query_texts: list[str] = []
    item_states: list[dict[str, Any]] = []
    stats = {
        "sense_nodes": 0,
        "query_senses": 0,
        "matched_senses": 0,
        "unmatched_senses": 0,
        "entries_with_candidates": 0,
        "entries_without_candidates": 0,
    }

    for item in items:
        entry = item.get("data", {}).get("entry", {})
        entry = entry if isinstance(entry, dict) else {}
        lemma = normalize_lemma(str(entry.get("lemma", "")))
        word_rows = get_word_rows(lemma)
        synsets = list(build_synset_entries(lemma))

        if word_rows and synsets:
            stats["entries_with_candidates"] += 1
        else:
            stats["entries_without_candidates"] += 1

        sense_nodes = collect_sense_nodes(entry.get("sense"))
        stats["sense_nodes"] += len(sense_nodes)

        query_nodes: list[dict[str, Any]] = []
        query_vector_indices: list[int] = []
        candidate_vector_indices: list[int] = []

        nonempty_nodes = [
            sense
            for sense in sense_nodes
            if str(sense.get("sense_definition", "")).strip()
        ]

        if synsets and nonempty_nodes:
            for synset in synsets:
                candidate_vector_indices.append(len(candidate_texts))
                candidate_texts.append(candidate_text(lemma, synset))

        for sense in sense_nodes:
            sense_definition = str(sense.get("sense_definition", "")).strip()
            if not synsets or not sense_definition:
                update_sense_with_match(sense, None, None)
                if sense_definition:
                    stats["unmatched_senses"] += 1
                continue

            query_nodes.append(sense)
            query_vector_indices.append(len(query_texts))
            query_texts.append(query_text(entry, sense))
            stats["query_senses"] += 1

        item_states.append(
            {
                "synsets": synsets,
                "query_nodes": query_nodes,
                "query_vector_indices": query_vector_indices,
                "candidate_vector_indices": candidate_vector_indices,
            }
        )

    print(
        f"Encoding {len(candidate_texts)} WordNet candidate texts and "
        f"{len(query_texts)} Iwanami sense queries with model {args.model}."
    )

    candidate_embeddings = None
    if candidate_texts:
        candidate_embeddings = model.encode(
            candidate_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    query_embeddings = None
    if query_texts:
        query_embeddings = model.encode(
            query_texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    for state in item_states:
        synsets = state["synsets"]
        query_nodes = state["query_nodes"]
        query_vector_indices = state["query_vector_indices"]
        candidate_vector_indices = state["candidate_vector_indices"]

        if (
            not synsets
            or not query_nodes
            or candidate_embeddings is None
            or query_embeddings is None
        ):
            continue

        candidate_matrix = candidate_embeddings[candidate_vector_indices]
        query_matrix = query_embeddings[query_vector_indices]
        similarity_matrix = np.matmul(query_matrix, candidate_matrix.T)

        for row_idx, sense in enumerate(query_nodes):
            similarity_scores = similarity_matrix[row_idx]
            best_index = int(np.argmax(similarity_scores))
            best_score = float(similarity_scores[best_index])
            update_sense_with_match(sense, synsets[best_index], best_score)
            stats["matched_senses"] += 1

    write_jsonl(args.output, items, args.encoding)
    print(
        "Done. "
        f"matched_senses={stats['matched_senses']} "
        f"unmatched_senses={stats['unmatched_senses']} "
        f"sense_nodes={stats['sense_nodes']} "
        f"entries_with_candidates={stats['entries_with_candidates']} "
        f"entries_without_candidates={stats['entries_without_candidates']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
