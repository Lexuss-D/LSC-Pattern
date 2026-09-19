# Experiment-to-command map

All stages use `bash pipeline/run.sh --config .local/config.env --stage STAGE`.
`--mode normalized` is the default. Outputs are below `.local/runs/normalized/`;
`--mode paper` defaults to a separate `.local/runs/paper/` directory.

| Paper result | Stage(s) | Principal outputs |
|---|---|---|
| Table 1, dataset statistics | export-annotations; aggregate | `data/annotations.csv`; `analysis/counts.json`, `entry_counts.csv` |
| Table 2, original corpus scale | corpus-stats (optional independent scan) | `corpus_statistics.json`: token records, valid/missing years; paper rounded counts: CHJ 12.3M, SHC 31.1M, BCCWJ 4.3M |
| WSD, Table 7 parameters | prepare; wsd | `prompts/`, `generated/`, model logs |
| Three semantic dimensions | dimensions; aggregate | `merged/`, `analysis/metrics_by_decade.csv`, `sense_metrics_by_decade.csv` |
| Table 3, Qwen + five baselines | restore; baselines; aggregate; evaluate | `baselines/`, `evaluation/evaluation.json` |
| Table 4, 7/20-word JaSemChange | jasemchange | `jasemchange/correlations/correlations.json` |
| Table 11, human agreement | evaluate | `evaluation/evaluation.json` (word-level alpha and pairwise kappa) |
| Table 12, Sense 0 | aggregate | `analysis/counts.json`, `sense_distribution_by_decade.csv` |
| Table 13, pairwise relatedness | jasemchange | `jasemchange/pairwise/` |
| Tables 5, 14–16; Appendix A.9 | aggregate; trends | `trends/slopes.csv`, `top_trends.csv`, `robustness.csv`, `summary.json` |
| Figure 2, case studies | figures | `figures/case_studies.png` and `.pdf`; numbered sense legends |
| Table 6, representative text | restore licensed contexts locally | Text is intentionally absent from public artifacts; no script republishes the paper's quotations |

The original inference used Qwen3-8B, temperature 0, thinking enabled, batch 64,
maximum generated tokens 16,384, memory utilization 0.92, and four 96-GB RTX PRO
6000 Blackwell Max-Q GPUs. The paper reports 10h 22m 6s; this is historical runtime,
not a promise for a different environment. The example config uses one GPU;
set `GPU_IDS=0,1,2,3` and `GPU_WORKERS=4` for the original parallel layout.

## Reuse existing predictions

Set `EXISTING_GENERATED`, `EXISTING_BASELINES`, and `EXISTING_JASEM_GENERATED` to
private directories with the original schemas. Run `aggregate`, `evaluate`,
`jasemchange`, `trends`, and `figures` individually. `all` always requests the full
pipeline, including model execution; setting reuse variables does not silently
turn it into a partial experiment.

For `paper` trend reproduction, set `HISTORICAL_METRICS` to a JSON list of the two
original numeric metric-table paths. The pipeline filters them to the fixed 150
entries. Use the original tables with one row per entry and decade.
Normalized analysis always rebuilds metric tables from predictions.

Each run writes its resource paths and settings to `inputs.json`. Use `--run-dir`
to choose an output directory and `--stage` to select where to start.
