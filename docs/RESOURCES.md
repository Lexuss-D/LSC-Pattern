# Resources

The pipeline uses local corpus, dictionary and model files. These resources are
not distributed with the code. Use the prepared formats below; the original corpus
filenames and sizes are listed in `data/resources.json`.

| Configuration | Required input / acquisition |
|---|---|
| `CHJ_CSV` | Enriched CHJ LUW CSV, `CHJ_with_bibliography.csv`; [provider](https://clrd.ninjal.ac.jp/chj/) |
| `SHC_CSV` | Enriched SHC LUW CSV, `SHC_with_bibliography.csv`; [provider](https://clrd.ninjal.ac.jp/shc/index.html) |
| `BCCWJ_CSV` | Magazine-register enriched LUW CSV, `PM_with_bibliography.csv`; [provider and application](https://clrd.ninjal.ac.jp/bccwj/en/qa.html) |
| `IWANAMI_TARGETS` | Original enriched Iwanami JSONL with recursive `data.entry.sense`, `@senseid`, `sense_definition`, optional `eg_definition`, WLSP and WordNet fields; obtain the licensed resource through [GSK](https://www.gsk.or.jp/), then retain the original preprocessing schema |
| `IWANAMI_FULL` | Full `iwanami_corpus.jsonl`, including monosemous entries, for JaSemChange |
| `WORDNET_DB` | Japanese WordNet SQLite `wnjpn.db`; [Japanese WordNet](https://bond-lab.github.io/wnja/) |
| `PN_DICT` | Original `pn.csv.m3.120408.trim`; [Japanese Sentiment Polarity Dictionary](https://www.cl.ecei.tohoku.ac.jp/Open_Resources-Japanese_Sentiment_Polarity_Dictionary.html) |
| `BUNRUI_WORD`, `BUNRUI_DATABASE` | Original WLSP-derived `bunnrui_data.json` and `bunnrui_database.json`; [WLSP provider](https://clrd.ninjal.ac.jp/goihyo.html) |
| `QWEN_CACHE` | Local Hugging Face cache for [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B), with `refs/main`, `snapshots/` and `blobs/` |
| `EMBEDDING_MODEL` | Model ID or local path for [Sarashina-Embedding-v2-1B](https://huggingface.co/sbintuitions/sarashina-embedding-v2-1b) |
| `BERT_MODEL` | Model ID or local path for [Tohoku Japanese BERT v3](https://huggingface.co/tohoku-nlp/bert-base-japanese-v3) |
| `BERT_CHECKPOINT` | The original fine-tuned WordNet WSD `best_model` directory. Base BERT is not an equivalent replacement. Training is outside this inference/evaluation release. |
| `METAPHOR_PACKAGE`, `METAPHOR_CHECKPOINT` | Original Zhu et al. (2025) inference package and `fold_4_best/model.pt`. No redistributable license was located in the supplied package; obtain these components from the model provider. The pipeline copies Python files into its private run directory before execution. |
| `JASEM_ROOT` | Original `JaSemChange_full` tree containing `samples`, `samples_shc_bccwj`, and `rawdata`; see the [dataset paper](https://www.jstage.jst.go.jp/article/jnlp/31/4/31_1487/_article) |
| `JASEM_ANNOTATED` | Annotated pair TSVs under `chj_bccwj` / `shc_bccwj`, including `*_Compare.tsv` |
| `JASEM_CHJ_SCORES`, `JASEM_SHC_SCORES` | Historical word-level tables with first column = word and `Compare` = mean relatedness. Both CSV and TSV delimiters are detected. |
| `PROMPT_TEMPLATE` | Exact original template with private demonstration text, reconstructed as below |

## Corpus columns

CHJ and SHC require `サンプルID`, `開始文字位置`, `終了文字位置`,
`長単位出現書字形`, `長単位語彙素`, `長単位語彙素読み`, and `成立年`.
The metaphor adapter additionally reads `長単位品詞` for CHJ and `corpus_col_32`
for the exact SHC enriched export used by the paper.

BCCWJ requires `サンプルID`, `出現形開始位置`, `出現形終了位置`,
`書字形出現形`, `語彙素`, `語彙素読み`, `出版年`, and `品詞`.

The release starts from these **already prepared CSV/JSONL formats**. It does not
claim a generic converter from arbitrary new corpus editions or arbitrary
Iwanami XML. Keep the original ordered records and tokenization. The historical
window traverses that token stream and does not reset at document boundaries.
The release deliberately preserves this behavior for exact reconstruction.

## Prompt template

`config/prompt_skeleton.txt` contains only task instructions and placeholders.
Prepare a private UTF-8 JSON object with keys `demo_usage` and `demo_senses`, using
the exact licensed demonstration from the original experiment. Then run:

```bash
python src/prepare_prompt.py --example-json .local/private/prompt_example.json \
  --output .local/private/prompt_template.txt
```

The maintainer can instead use `--source-script /path/to/original/build_lscp_wsd_prompts.py`.
The tool parses the literal with Python AST; it never imports or executes that
private source file.
The original example is a separate input for reproducing the paper. Using a
different example changes the prompt and may change the predictions.

## Derived inputs and weights

`DICTIONARY_ALREADY_ALIGNED=1` reuses an original enriched dictionary with the
recorded WordNet alignment. Otherwise the pipeline reruns Sarashina alignment.
The released target and sense-ID manifests fix the 150 entries and their 501
candidate indices without distributing definitions, examples, depth values, or
WordNet mappings.

The metaphor preprocessing path reproduces the original run's fallback setting:
it supplies the WLSP lookup and does not add a BCCWJ representative-sentence
corpus. Providing that additional corpus would define a different experiment.
No model weights or BCCWJ-derived representative sentences are in the release.

The original example cannot be reconstructed from the corpus and dictionary files
alone. Supply the original template, the original prompt script, or a private JSON
file containing the example.
