#!/usr/bin/env python3
"""Evaluate JaSemChange pairwise relatedness against dictionary WSD labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATED_DIR = REPO_ROOT / ".local" / "inputs" / "default_annotated_dir"
DEFAULT_RUN_DIR = REPO_ROOT / ".local" / "inputs" / "default_run_dir"
DATASET_LABELS = {
    "chj_bccwj": "CHJ-BCCWJ",
    "shc_bccwj": "SHC-BCCWJ",
    "combined": "Combined",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same/different Iwanami WSD predictions with JaSemChange "
            "pairwise human relatedness judgments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--annotated-dir", type=Path, default=DEFAULT_ANNOTATED_DIR)
    parser.add_argument("--generated-dir", type=Path, default=DEFAULT_RUN_DIR / "generated")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RUN_DIR / "pairwise_wsd_evaluation",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Stratified bootstrap replicates used for the AUC confidence interval.",
    )
    parser.add_argument("--seed", type=int, default=20260713)
    return parser


def normalized_dataset_name(path: Path) -> str:
    for part in path.parts:
        normalized = part.strip().lower()
        if normalized in {"chj_bccwj", "shc_bccwj"}:
            return normalized
    raise ValueError(f"Could not infer dataset from path: {path}")


def usage_key(
    lemma: str,
    sample_id: Any,
    start_position: Any,
    sequence_number: Any,
) -> tuple[str, str, str, str]:
    return (
        str(lemma).strip(),
        str(sample_id).strip(),
        str(start_position).strip(),
        str(sequence_number).strip(),
    )


def load_wsd_index(generated_dir: Path) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    paths = sorted(generated_dir.glob("*.jsonl"))

    index: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                key = usage_key(
                    record["target_lemma"],
                    record["sample_id"],
                    record["start_position"],
                    record["serial"],
                )
                index[key] = record
    return index


def parse_valid_ratings(row: dict[str, str]) -> list[int]:
    ratings: list[int] = []
    for column, value in row.items():
        if not column.startswith("worker"):
            continue
        try:
            rating = int(value.strip())
        except (AttributeError, ValueError):
            continue
        if 1 <= rating <= 4:
            ratings.append(rating)
    return ratings


def load_pair_records(
    annotated_dir: Path,
    wsd_index: dict[tuple[str, str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    paths = sorted(annotated_dir.rglob("*_Compare.tsv"))

    records: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for path in paths:
        dataset = normalized_dataset_name(path)
        target_lemma = path.name.removesuffix("_Compare.tsv")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row_number, row in enumerate(reader, start=2):
                ratings = parse_valid_ratings(row)
                if not ratings:
                    exclusions["no_valid_human_rating"] += 1
                    continue

                key1 = usage_key(
                    target_lemma,
                    row["usage1_SampleID"],
                    row["usage1_CharacterStartingPosition"],
                    row["usage1_SequenceNumber"],
                )
                key2 = usage_key(
                    target_lemma,
                    row["usage2_SampleID"],
                    row["usage2_CharacterStartingPosition"],
                    row["usage2_SequenceNumber"],
                )
                usage1 = wsd_index.get(key1)
                usage2 = wsd_index.get(key2)
                if usage1 is None or usage2 is None:
                    exclusions["missing_wsd_usage"] += 1
                    continue
                sense1 = usage1.get("predicted_sense_index")
                sense2 = usage2.get("predicted_sense_index")
                if sense1 is None or sense2 is None:
                    exclusions["unparsed_wsd_prediction"] += 1
                    continue

                low_votes = sum(rating <= 2 for rating in ratings)
                high_votes = sum(rating >= 3 for rating in ratings)
                human_low: int | None
                if low_votes == high_votes:
                    human_low = None
                    exclusions["tied_low_high_votes_for_auc"] += 1
                else:
                    human_low = int(low_votes > high_votes)

                records.append(
                    {
                        "dataset": dataset,
                        "target_lemma": target_lemma,
                        "source_file": str(path),
                        "source_row": row_number,
                        "usage1_id": usage1["usage_id"],
                        "usage2_id": usage2["usage_id"],
                        "usage1_sample_id": key1[1],
                        "usage1_start_position": key1[2],
                        "usage1_sequence_number": key1[3],
                        "usage2_sample_id": key2[1],
                        "usage2_start_position": key2[2],
                        "usage2_sequence_number": key2[3],
                        "predicted_sense1": int(sense1),
                        "predicted_sense2": int(sense2),
                        "predicted_same": int(sense1 == sense2),
                        "predicted_different": int(sense1 != sense2),
                        "human_ratings": ",".join(map(str, ratings)),
                        "human_rating_count": len(ratings),
                        "human_relatedness": sum(ratings) / len(ratings),
                        "human_exact_same": int(all(rating == 4 for rating in ratings)),
                        "human_related_same_3plus": int(
                            sum(ratings) / len(ratings) >= 3.0
                        ),
                        "human_low_votes": low_votes,
                        "human_high_votes": high_votes,
                        "human_low": human_low,
                    }
                )
    return records, exclusions


def stratified_auc_bootstrap(
    labels: np.ndarray,
    scores: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    negative_scores = scores[labels == 0]
    positive_scores = scores[labels == 1]
    # With a binary prediction score, AUC is (sensitivity + specificity) / 2.
    # Stratified case resampling is therefore equivalent to these binomial draws.
    bootstrap_true_positives = rng.binomial(
        len(positive_scores), positive_scores.mean(), size=samples
    )
    bootstrap_true_negatives = rng.binomial(
        len(negative_scores), 1.0 - negative_scores.mean(), size=samples
    )
    aucs = 0.5 * (
        bootstrap_true_positives / len(positive_scores)
        + bootstrap_true_negatives / len(negative_scores)
    )
    low, high = np.quantile(aucs, [0.025, 0.975])
    return float(low), float(high)


def evaluate_group(
    records: list[dict[str, Any]],
    dataset: str,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    selected = records if dataset == "combined" else [r for r in records if r["dataset"] == dataset]
    same_scores = np.array(
        [r["human_relatedness"] for r in selected if r["predicted_same"]], dtype=float
    )
    different_scores = np.array(
        [r["human_relatedness"] for r in selected if not r["predicted_same"]], dtype=float
    )

    welch = stats.ttest_ind(same_scores, different_scores, equal_var=False, alternative="greater")
    mann_whitney = stats.mannwhitneyu(
        same_scores, different_scores, alternative="greater", method="asymptotic"
    )
    same_indicator = np.array([r["predicted_same"] for r in selected], dtype=int)
    relatedness = np.array([r["human_relatedness"] for r in selected], dtype=float)
    point_biserial = stats.pointbiserialr(same_indicator, relatedness)

    auc_records = [r for r in selected if r["human_low"] is not None]
    auc_labels = np.array([r["human_low"] for r in auc_records], dtype=int)
    auc_scores = np.array([r["predicted_different"] for r in auc_records], dtype=int)
    auc = float(roc_auc_score(auc_labels, auc_scores))
    auc_low, auc_high = stratified_auc_bootstrap(
        auc_labels, auc_scores, bootstrap_samples, seed
    )

    exact_tp = sum(r["predicted_same"] and r["human_exact_same"] for r in selected)
    exact_tn = sum(
        not r["predicted_same"] and not r["human_exact_same"] for r in selected
    )
    exact_fp = sum(r["predicted_same"] and not r["human_exact_same"] for r in selected)
    exact_fn = sum(
        not r["predicted_same"] and r["human_exact_same"] for r in selected
    )
    exact_agreement_count = exact_tp + exact_tn
    for record in selected:
        record["exact_match_agreement"] = int(
            record["predicted_same"] == record["human_exact_same"]
        )

    related_tp = sum(
        r["predicted_same"] and r["human_related_same_3plus"] for r in selected
    )
    related_tn = sum(
        not r["predicted_same"] and not r["human_related_same_3plus"]
        for r in selected
    )
    related_fp = sum(
        r["predicted_same"] and not r["human_related_same_3plus"] for r in selected
    )
    related_fn = sum(
        not r["predicted_same"] and r["human_related_same_3plus"]
        for r in selected
    )
    related_agreement_count = related_tp + related_tn
    for record in selected:
        record["related_3plus_agreement"] = int(
            record["predicted_same"] == record["human_related_same_3plus"]
        )

    return {
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "n_pairs": len(selected),
        "n_same": len(same_scores),
        "n_different": len(different_scores),
        "same_mean": float(same_scores.mean()),
        "same_sd": float(same_scores.std(ddof=1)),
        "different_mean": float(different_scores.mean()),
        "different_sd": float(different_scores.std(ddof=1)),
        "mean_difference": float(same_scores.mean() - different_scores.mean()),
        "welch_t": float(welch.statistic),
        "welch_p_one_sided": float(welch.pvalue),
        "mann_whitney_u": float(mann_whitney.statistic),
        "mann_whitney_p_one_sided": float(mann_whitney.pvalue),
        "point_biserial_r": float(point_biserial.statistic),
        "point_biserial_p_two_sided": float(point_biserial.pvalue),
        "auc_n": len(auc_records),
        "auc_n_low": int(auc_labels.sum()),
        "auc_n_high": int(len(auc_labels) - auc_labels.sum()),
        "auc_excluded_ties": len(selected) - len(auc_records),
        "auc": auc,
        "auc_ci95_low": auc_low,
        "auc_ci95_high": auc_high,
        "exact_human_same_n": exact_tp + exact_fn,
        "exact_human_different_n": exact_tn + exact_fp,
        "exact_agreement_count": exact_agreement_count,
        "exact_agreement_rate": exact_agreement_count / len(selected),
        "exact_tp": exact_tp,
        "exact_tn": exact_tn,
        "exact_fp": exact_fp,
        "exact_fn": exact_fn,
        "related_3plus_human_same_n": related_tp + related_fn,
        "related_3plus_human_different_n": related_tn + related_fp,
        "related_3plus_agreement_count": related_agreement_count,
        "related_3plus_agreement_rate": related_agreement_count / len(selected),
        "related_3plus_tp": related_tp,
        "related_3plus_tn": related_tn,
        "related_3plus_fp": related_fp,
        "related_3plus_fn": related_fn,
    }


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_p(value: float) -> str:
    return f"{value:.3e}" if value < 0.001 else f"{value:.4f}"


def write_report(
    path: Path,
    metrics: list[dict[str, Any]],
    records: list[dict[str, Any]],
    exclusions: Counter[str],
    annotated_dir: Path,
    generated_dir: Path,
    bootstrap_samples: int,
    seed: int,
) -> None:
    lines = [
        "# JaSemChange Usage Pair WSD 评价报告",
        "",
        "## 评价设置",
        "",
        f"- 人工标注：`{annotated_dir}`，只读取 `*_Compare.tsv`。",
        f"- WSD 结果：`{generated_dir}`。",
        "- 人工 relatedness 是每个 pair 所有有效 `1`-`4` 标注的算术平均；"
        "忽略 `0` 和非数值备注。",
        "- 仅当两个 usage 的 `predicted_sense_index` 完全相同时，"
        "`predicted_same = 1`。",
        "- 对 AUC，人工评分 `1`-`2` 投票为 low-relatedness（different sense），"
        "`3`-`4` 投票为 high-relatedness；两类票数相同的 pair 只从 AUC 中排除。",
        "- 均值比较使用单侧 Welch t 检验，备择假设为 "
        "`mean(same) > mean(different)`；同时报告单侧 Mann-Whitney U 检验作为"
        "适用于序数评分的稳健性检查。",
        "- Point-biserial correlation 将 WSD same-sense 编码为 `1`、"
        "different-sense 编码为 `0`，p 值为双侧。",
        "- AUC 以人工 low-relatedness 为正类，以 WSD different-sense (`0/1`) "
        "为预测分数。",
        "- 完全同义一致率评价将所有有效人工评分均为 `4` 的 pair 视为 "
        "human same-sense；只要存在低于 `4` 的有效评分，就视为 human "
        "different-sense。",
        "- 放宽后的二元一致率评价将人工平均 relatedness `>=3` 的 pair 视为 "
        "human same-sense，将平均分 `<3` 的 pair 视为 human different-sense。",
        f"- AUC 的 95% CI 使用 {bootstrap_samples:,} 次按类别分层的 bootstrap "
        f"（seed `{seed}`）。",
        "",
        "## 结果",
        "",
        "| 数据集 | n | Same n / 均值 | Different n / 均值 | 均值差 | Welch p（单侧） | Point-biserial r | p（双侧） | AUC | 95% CI | AUC n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['dataset_label']} | {row['n_pairs']} | "
            f"{row['n_same']} / {row['same_mean']:.4f} | "
            f"{row['n_different']} / {row['different_mean']:.4f} | "
            f"{row['mean_difference']:.4f} | {format_p(row['welch_p_one_sided'])} | "
            f"{row['point_biserial_r']:.4f} | "
            f"{format_p(row['point_biserial_p_two_sided'])} | {row['auc']:.4f} | "
            f"[{row['auc_ci95_low']:.4f}, {row['auc_ci95_high']:.4f}] | "
            f"{row['auc_n']} |"
        )

    lines.extend(["", "## 显著性与解释", ""])
    for row in metrics:
        significant = row["welch_p_one_sided"] < 0.05 and row["mean_difference"] > 0
        verdict = "是" if significant else "否"
        lines.append(
            f"- **{row['dataset_label']}**：在 alpha=0.05 下，same-sense pair 的"
            f"平均 relatedness {'显著高于' if significant else '没有显著高于'} "
            f"different-sense pair（{verdict}；Welch `t={row['welch_t']:.4f}`, "
            f"`p={format_p(row['welch_p_one_sided'])}`）。Mann-Whitney 稳健性检查为 "
            f"`U={row['mann_whitney_u']:.1f}`, "
            f"`p={format_p(row['mann_whitney_p_one_sided'])}`。"
        )
    combined = next(row for row in metrics if row["dataset"] == "combined")
    lines.extend(
        [
            "",
            "在合并数据中，正的 point-biserial correlation 表明：被 WSD 判为同一辞典"
            f"词义的 pair 往往获得更高的人工 relatedness（`r_pb="
            f"{combined['point_biserial_r']:.4f}`）。AUC `{combined['auc']:.4f}` "
            "衡量二元 WSD 判断区分人工 low-relatedness 与 high-relatedness pair 的"
            "能力，其中 `0.5` 为随机水平。",
            "",
            "## 完全同义二元一致率",
            "",
            "| 数据集 | Human same | Human different | 判断一致 | 一致率 | TP | TN | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['dataset_label']} | {row['exact_human_same_n']} | "
            f"{row['exact_human_different_n']} | {row['exact_agreement_count']} / "
            f"{row['n_pairs']} | {row['exact_agreement_rate']:.4f} | "
            f"{row['exact_tp']} | {row['exact_tn']} | {row['exact_fp']} | "
            f"{row['exact_fn']} |"
        )
    lines.extend(
        [
            "",
            "其中正类为 human same-sense：TP 表示人工与 WSD 都判断为同义，TN "
            "表示二者都判断为不同义；FP 表示 WSD 判断同义但人工未达到完全同义，"
            "FN 表示人工完全同义但 WSD 判断为不同义。",
            "",
            "## 将评分 3 和 4 视为同词义",
            "",
            "| 数据集 | Human same | Human different | 判断一致 | 一致率 | TP | TN | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['dataset_label']} | {row['related_3plus_human_same_n']} | "
            f"{row['related_3plus_human_different_n']} | "
            f"{row['related_3plus_agreement_count']} / {row['n_pairs']} | "
            f"{row['related_3plus_agreement_rate']:.4f} | "
            f"{row['related_3plus_tp']} | {row['related_3plus_tn']} | "
            f"{row['related_3plus_fp']} | {row['related_3plus_fn']} |"
        )
    lines.extend(
        [
            "",
            "这里 human same-sense 表示 pair 的人工平均 relatedness `>=3`；"
            "TP/TN/FP/FN 的正类均为 human same-sense。",
            "",
            "## 数据覆盖",
            "",
            f"- 用于均值和相关分析的 pair：{len(records)}。",
            f"- 没有有效数值人工评分而排除的 pair："
            f"{exclusions['no_valid_human_rating']}。",
            f"- 缺少 WSD usage 的 pair：{exclusions['missing_wsd_usage']}。",
            f"- 包含未解析 WSD 结果的 pair："
            f"{exclusions['unparsed_wsd_prediction']}。",
            f"- low/high 人工票数相同、仅从 AUC 排除的 pair："
            f"{exclusions['tied_low_high_votes_for_auc']}。",
            "",
            "## 注意事项",
            "",
            "上述检验将 pair 视为独立观测。由于同一 usage 和 target word 可能出现在"
            "多个 pair 中，p 值没有建模这种聚类依赖；每个 pair 的效应估计和 AUC 仍可"
            "直接解释，但解读推断性 p 值时应注意这一限制。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wsd_index = load_wsd_index(args.generated_dir)
    records, exclusions = load_pair_records(args.annotated_dir, wsd_index)

    metrics = [
        evaluate_group(records, dataset, args.bootstrap_samples, args.seed + offset)
        for offset, dataset in enumerate(("chj_bccwj", "shc_bccwj", "combined"))
    ]
    pair_fields = list(records[0])
    metric_fields = list(metrics[0])
    write_csv(args.output_dir / "pairwise_predictions.csv", records, pair_fields)
    write_csv(args.output_dir / "evaluation_metrics.csv", metrics, metric_fields)
    write_report(
        args.output_dir / "pairwise_wsd_evaluation_report.md",
        metrics,
        records,
        exclusions,
        args.annotated_dir,
        args.generated_dir,
        args.bootstrap_samples,
        args.seed,
    )

    print(f"WSD usages indexed: {len(wsd_index)}")
    print(f"Pair rows retained: {len(records)}")
    print(f"Exclusions: {dict(exclusions)}")
    for row in metrics:
        print(
            f"{row['dataset_label']}: n={row['n_pairs']}, "
            f"same_mean={row['same_mean']:.4f}, "
            f"different_mean={row['different_mean']:.4f}, "
            f"r_pb={row['point_biserial_r']:.4f}, AUC={row['auc']:.4f}"
        )
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
