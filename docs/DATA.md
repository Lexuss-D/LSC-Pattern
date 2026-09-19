# Annotation dataset

`data/annotations.csv` is the only usage-level human annotation table. It contains
976 rows, 50 entries, and three independent labels per usage. Counts: CHJ 308,
SHC 503, BCCWJ 165. Text is absent; all 976 locations were verified by recovering
the full original window from the private corpus files.

CSV encoding is UTF-8, with a header. Empty strings indicate missing/unavailable
values. Do not conflate an empty cell with integer 0. Read IDs and positions as
strings so leading zeros survive spreadsheet imports.

| Fields | Meaning |
|---|---|
| `usage_id` | `lscp_<corpus>_<corpus_record_no>`, for example `lscp_chj_12345`; corpus name is lowercase |
| `entry_id`, `lemma`, `lemma_reading` | Research entry identity; readings disambiguate homographs |
| `corpus`, `resource_id`, `corpus_file_id` | Corpus identity, snapshot reference, and configuration key |
| `sample_id` | Original corpus sample ID, copied as a string |
| `target_start`, `target_end`, `position_system` | Original native coordinate values and the original column names; no coordinate conversion |
| `year`, `decade` | Original usage year and decade bucket |
| `corpus_record_no` | 1-based CSV data-record ordinal, excluding the header; not a physical line number |
| `source_id`, `source_usage_row` | Legacy extracted-file identifier and 1-based JSONL line number for provenance; these are not corpus sample IDs |
| `label_annotator_N_raw` | Original spreadsheet value rendered as text; full-width values remain visible |
| `label_annotator_N` | Integer after digit/bracket/Excel numeric normalization; even invalid integer 22 is retained |
| `label_annotator_N_valid` | 1 if the value is in that entry's workbook inventory, otherwise 0 |
| `majority_label` | At least two valid annotators agree, in the **annotation** inventory; blank on no majority |
| `n_valid_labels`, `agreement_count`, `consensus_status` | Valid annotation count, largest vote count, and majority/no-majority status |

`N` is 1, 2 or 3. Names, worker filenames, working-time sheets and the private
name-to-number mapping are absent from this repository. Numbering remains fixed
across exports through a private maintainer mapping file.

## Related manifests

- `targets.csv`: the fixed 150 research entries and expected output filenames.
- `sense_ids.csv`: 501 model-candidate indices and dictionary IDs, without glosses.
- `annotation_inventory.json`: the actual workbook choices for the 50 entries.
- `annotation_sense_alignment.csv`: exact, verified correspondence between the
  annotation and inference inventories; unresolved mappings stay empty.
- `resources.json`: corpus filenames, byte sizes and record-numbering conventions.
  Official release versions were not recorded in the original project.

The original annotation label is never overwritten by the model inventory index.
Normalized model evaluation uses the alignment table and recomputes eligible
majorities; therefore its denominator differs from the CSV's 961 annotation
majorities. See [the result notes](REPRODUCIBILITY.md).

## Local restoration

Set the three corpus paths in `.local/config.env`, then run:

```bash
bash pipeline/run.sh --config .local/config.env --stage restore
```

The output `.local/runs/normalized/restored_annotations.jsonl` contains the original
contexts. Restoration reads each target's CSV record number and takes the surrounding
30 tokens on each side. It requires the original corpus record order and tokenization.
The historical extraction window does not reset at document boundaries.

## Export annotations

Create a local JSON mapping `annotator_1`, `annotator_2`, `annotator_3` to the three
workbooks. Set `ANNOTATOR_MAP` and the corpus paths in the config, then run
`--stage export-annotations`. The exporter combines labels by `source_id` and uses
the target's occurrence number to recover native corpus positions. It writes
`data/annotations.csv`, `data/annotation_inventory.json` and `data/resources.json`.

The metadata-only format avoids distributing corpus passages; it is not a blanket
license grant for the referenced resources. The providers' applicable conditions
remain separate from the license on the original human labels.

## Rebuild the verified sense correspondence

With the three private workbooks and a private candidate inventory emitted by
`aggregate`, rebuild the sense mapping with:

```bash
python src/align_annotation_senses.py --annotator-map .local/private/annotators.json \
  --model-inventory .local/runs/normalized/analysis/inventory.jsonl
```

The script matches cleaned definitions exactly. Missing or ambiguous matches stay
unresolved. Its default output is `.local/reports/annotation_sense_alignment.csv`;
use `--output data/annotation_sense_alignment.csv` to update the public mapping.

`data/jasemchange_targets.csv` lists the 20 benchmark words and the seven-word
overlap subset. Neither this list nor the sense mapping contains dictionary text.
