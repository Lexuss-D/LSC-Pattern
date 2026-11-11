# LSC-Pattern

**English** · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

Code and annotations for **LSC-Pattern: Interpretable Lexical Semantic Change
Analysis via Diachronic Sense Distributions** (Ling and Komachi, EMNLP 2026).

The pipeline assigns Japanese historical usages to dictionary senses and tracks
three measures over time: context valence, semantic scope and metaphoricity.
The main experiment uses 150 words. The annotation CSV contains 976 usages of
50 words, each labeled by three anonymous annotators.

## Request for full annotated data

If you have bought the Iwanami ver.5 and CHJ & SHC lisence, you may contact us for the annotated data with sense IDs and usages.

## Install

Use Python 3.12. Full inference requires NVIDIA GPUs and a compatible CUDA environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
```

For statistics from existing predictions, install `requirements.txt` instead.

## Prepare the resources

Corpus text, dictionary entries, benchmark data and model weights are not included.
Supply the prepared CHJ, SHC and BCCWJ CSVs, Iwanami and WordNet files, polarity and
WLSP resources, JaSemChange data, Qwen3-8B and the baseline/metaphor checkpoints.
File formats and download links are in [RESOURCES.md](docs/RESOURCES.md).

```bash
mkdir -p .local
cp config/example.env .local/config.env
# Set the resource paths and GPU options in .local/config.env.
```

`PROMPT_TEMPLATE` points to the original prompt with its private example. That
example is a separate input; corpus and dictionary files alone do not reconstruct
it. [RESOURCES.md](docs/RESOURCES.md#prompt-template) explains how to prepare it.

## Run

```bash
bash pipeline/run.sh --config .local/config.env --stage all --mode normalized
```

The stages run in this order. Replace `all` with a stage name to run it separately.

| Stage | What it does |
|---|---|
| `prepare` | Build the sense inventory, align WordNet, extract usages and create prompts |
| `restore` | Recover the annotation contexts from local corpus files |
| `wsd` | Run Qwen3-8B sense prediction |
| `dimensions` | Compute valence and metaphoricity; merge predictions |
| `aggregate` | Build decade-level sense distributions and dimension scores |
| `baselines` | Run WordNet BERT, usage–gloss and target–gloss retrieval |
| `evaluate` | Score Qwen and the five baselines, including Random and MFS; compute annotator agreement |
| `jasemchange` | Evaluate the 7-word and 20-word sets and usage pairs |
| `trends` | Run OLS, Theil–Sen, Mann–Kendall and BH correction |
| `figures` | Save the three case-study plots as PNG and PDF |

Results go to `.local/runs/normalized/`. Use `--run-dir .local/runs/NAME` for another
output directory. Start at the stage you need; intermediate files come from earlier
stages. `corpus-stats` separately counts corpus tokens and years.

The paper used temperature 0, thinking enabled, batch 64, up to 16,384 generated
tokens and GPU memory utilization 0.92. The example uses one GPU. Set
`GPU_IDS=0,1,2,3` and `GPU_WORKERS=4` for four workers.

## Use existing predictions

Set `EXISTING_GENERATED`, `EXISTING_BASELINES` and `EXISTING_JASEM_GENERATED` in the
config, then run the analysis stages:

```bash
for stage in aggregate evaluate jasemchange trends figures; do
  bash pipeline/run.sh --config .local/config.env --stage "$stage"
done
```

`normalized` maps human labels to model senses using confirmed correspondences and
excludes unresolved labels. `--mode paper` uses the historical scoring rules. For
historical trend tables, also set `HISTORICAL_METRICS` to the two original CSV paths.
The evaluation sets differ: Qwen scores 776/924 in normalized mode and 783/961 in
paper mode. This difference does not indicate a model improvement.

## Data and documentation

[annotations.csv](data/annotations.csv) contains corpus locations and all three
labels, without passages or dictionary definitions. It includes 308 CHJ, 503 SHC
and 165 BCCWJ usages. Sense 0 is a valid fallback label; an empty value is missing.

- [DATA.md](docs/DATA.md): CSV fields, label rules and context restoration.
- [EXPERIMENTS.md](docs/EXPERIMENTS.md): paper tables, commands and output files.
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): historical differences and reference results.

Valence describes context sentiment, WordNet depth approximates specificity, and
metaphoricity is a model prediction. Interpret trends alongside the sense distributions.

## License and citation

Original code and documentation: [MIT](LICENSE). Original human labels:
[CC BY 4.0](LICENSE-DATA.md). Third-party resources retain their own terms.

```bibtex
@inproceedings{ling-komachi-2026-lsc-pattern,
  title = {LSC-Pattern: Interpretable Lexical Semantic Change Analysis via Diachronic Sense Distributions},
  author = {Ling, Zhidong and Komachi, Mamoru},
  year = {2026},
  booktitle = {Proceedings of EMNLP}
}
```

See also [CITATION.cff](CITATION.cff).
