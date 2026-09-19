"""Sense distributions and the three semantic dimensions used in the paper."""
from __future__ import annotations
import math
import re
from collections import Counter,defaultdict
from dataclasses import dataclass,field
from pathlib import Path
from typing import Any
CORPUS_ORDER = ('CHJ','SHC','BCCWJ')
FILENAME_RE = re.compile(r'^(BCCWJ_all|CHJ_all|SHC_all)_(.*)_n\d+\.jsonl$')
XML_ID_ARTIFACT_RE = re.compile(r'id\d+')

@dataclass
class SenseBinStats:
    n: int = 0
    valence_sum: float = 0.0
    valence_n: int = 0
    depth_sum: float = 0.0
    depth_n: int = 0
    metaphor_sum: int = 0
    metaphor_n: int = 0
    corpus_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class BinStats:
    n: int = 0
    valence_sum: float = 0.0
    valence_n: int = 0
    depth_sum: float = 0.0
    depth_n: int = 0
    metaphor_sum: int = 0
    metaphor_n: int = 0
    sense_counts: Counter[int] = field(default_factory=Counter)
    sense_stats: dict[int, SenseBinStats] = field(
        default_factory=lambda: defaultdict(SenseBinStats)
    )
    corpus_counts: Counter[str] = field(default_factory=Counter)


@dataclass
class WordStats:
    lemma: str
    lemma_reading: str
    bins: dict[int, BinStats] = field(default_factory=lambda: defaultdict(BinStats))
    rows_read: int = 0
    skipped_bad_time_bin: int = 0


def normalize_corpus(value: Any, path: Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        match = FILENAME_RE.match(path.name)
        raw = match.group(1) if match else ""
    if raw.endswith("_all"):
        raw = raw[: -len("_all")]
    if raw not in CORPUS_ORDER:
        return raw or "UNKNOWN"
    return raw


def parse_year(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_decade_label(value: Any) -> tuple[int, str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    match = re.fullmatch(r"(-?\d+)s", text)
    if not match:
        return None
    start = int(match.group(1))
    return start, f"{start}s"


def time_bin_from_record(record: dict[str, Any], time_bin: str) -> tuple[int, str] | None:
    year = parse_year(record.get("timestamp"))
    if time_bin == "year":
        if year is None:
            return None
        return year, str(year)

    if time_bin == "decade":
        parsed_decade = parse_decade_label(record.get("decade"))
        if parsed_decade is not None:
            return parsed_decade
        if year is None:
            return None
        decade_start = year // 10 * 10
        return decade_start, f"{decade_start}s"

    raise ValueError(f"Unsupported time bin: {time_bin}")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def format_optional_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.10g}"


def predicted_candidate(record: dict[str, Any]) -> dict[str, Any] | None:
    predicted_index = record.get("predicted_sense_index")
    if not isinstance(predicted_index, int) or predicted_index < 0:
        return None
    for candidate in record.get("wsd_candidate_synsets") or []:
        if isinstance(candidate, dict) and candidate.get("sense_index") == predicted_index:
            return candidate
    return None


def clean_definition_text(value: str) -> str:
    cleaned = XML_ID_ARTIFACT_RE.sub("", value)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def candidate_depth(candidate: dict[str, Any] | None) -> float | None:
    if candidate is None:
        return None
    raw = candidate.get("depth_to_root")
    if is_number(raw):
        return float(raw)
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def add_record(stats: WordStats, record: dict[str, Any], path: Path, time_bin: str) -> None:
    stats.rows_read += 1

    parsed_bin = time_bin_from_record(record, time_bin)
    if parsed_bin is None:
        stats.skipped_bad_time_bin += 1
        return
    bin_key, _bin_label = parsed_bin

    corpus = normalize_corpus(record.get("corpus"), path)
    bin_stats = stats.bins[bin_key]
    bin_stats.n += 1
    bin_stats.corpus_counts[corpus] += 1

    valence = record.get("valence")
    if is_number(valence):
        bin_stats.valence_sum += float(valence)
        bin_stats.valence_n += 1

    candidate = predicted_candidate(record)
    depth = candidate_depth(candidate)
    if depth is not None:
        bin_stats.depth_sum += depth
        bin_stats.depth_n += 1

    metaphor = record.get("metaphor_prediction")
    if metaphor in (0, 1):
        bin_stats.metaphor_sum += int(metaphor)
        bin_stats.metaphor_n += 1

    predicted_index = record.get("predicted_sense_index")
    if isinstance(predicted_index, int) and predicted_index >= 0:
        bin_stats.sense_counts[predicted_index] += 1
        sense_stats = bin_stats.sense_stats[predicted_index]
        sense_stats.n += 1
        sense_stats.corpus_counts[corpus] += 1
        if is_number(valence):
            sense_stats.valence_sum += float(valence)
            sense_stats.valence_n += 1
        if depth is not None:
            sense_stats.depth_sum += depth
            sense_stats.depth_n += 1
        if metaphor in (0, 1):
            sense_stats.metaphor_sum += int(metaphor)
            sense_stats.metaphor_n += 1

