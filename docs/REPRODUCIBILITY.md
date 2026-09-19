# Reproduction notes

The camera-ready paper is the reference, but source artifacts take precedence over
invented corrections. This release preserves original annotations and exposes
verified differences. `normalized` is the default; `paper` retains historical
numeric-label evaluation and accepts historical metric tables. It is not a claim
that every printed number can be reconstructed exactly from the available files.

## Annotation and sense inventories

All three source workbooks contain the same 976 usages and 50 entries. All 976
native corpus locations and reconstructed token windows were verified against
the workbooks. There are 961 original majority-vote usages and 15 with no majority.
One label is 22 for an entry whose allowed annotation labels are 0–8. It is retained
and marked invalid. Two full-width numerical labels are preserved in raw columns.

The actual workbooks contain **145 nonzero sense definitions**, not the 162 stated
in the paper. The inference candidate lists for those 50 entries contain 150 senses.
The entire main cohort has 501 candidates, 454 with a WordNet alignment. These are
different inventories and are reported separately.

The public alignment manifest contains 141 exact definition correspondences and
four unresolved senses, plus 50 fallback-zero entries. Exact comparison trims
whitespace and removes the original XML-ID formatting artifacts; it does not use
semantic similarity or guess correspondence. For 道具_ドウグ, annotation sense 2
maps to inference sense 6. Unresolved: 結構_ケッコウ:5, 迚も_トテモ:1 and :2,
and 次元_ジゲン:2. No restricted definitions are included in this report.

For normalized WSD evaluation, valid human labels are mapped into the fixed model
inventory. Unresolved mappings are excluded before majority voting. This leaves
**924 majority references**, with **776 Qwen matches (0.839827)**; this is a different
evaluation set from the historical **783/961 (0.814776)** and must not be described
as a model improvement. Individual normalized denominators are 938, 939, and 940.
IAA remains an agreement measurement in the original annotation inventory and
excludes only invalid labels; it does not discard senses merely because their
model alignment is unresolved.

## Intrinsic evaluation

| Method | Camera-ready majority | Recomputed historical labels | Normalized aligned labels |
|---|---:|---:|---:|
| Random | 0.282 | 0.279740 | 0.279398 |
| MFS | 0.699 | 0.699272 | 0.695887 |
| WordNet BERT → Iwanami | 0.435 | 0.434964 | 0.445887 |
| Target token–gloss | 0.591 | 0.591051 | 0.595238 |
| Usage–gloss | 0.681 | 0.680541 | 0.694805 |
| Qwen3-8B | 0.815 | 0.814776 | 0.839827 |

All regenerated baseline scores use the same available prediction files. Uncovered
WordNet predictions count as incorrect in majority accuracy. The three individual
historical retrieval scores use model-covered rows, matching the original report;
the release also records full-set coverage so that the denominator is visible.
MFS is an optimistic baseline fitted separately to each reference's evaluation
labels. Random is an analytical expectation over the actual displayed candidates
(including fallback 0), not a sampled run.

The archived random summary reports 0.281848 majority accuracy and 0.281478 against
each annotator. Its generating script/configuration was not found, and neither
the workbook inventory nor the stored inference inventory reproduces that value.
The release computes the documented expectation and explicitly reports the
remaining discrepancy instead of adjusting candidate counts to force agreement.

Historical word-level alpha is 0.688586; normalized alpha is 0.690024.
Historical mean pairwise kappa is 0.798625, calculated on bare numeric sense labels.
The paper describes entry–sense labels; implementing that description and excluding
invalid labels gives normalized kappa 0.866863. The printed 0.798 is kept as a paper
reference, not silently substituted for the recomputed value.

## Main usage counts and missing data

The actual 150-file prediction set contains 213,945 records. Of these, 269 have a
blank timestamp. Removing those leaves 213,676 date-valid records. Among these,
26 predictions have no parsed sense. Thus the sense-distribution total is 213,650,
matching the paper. There are 3,037 Sense-0 predictions. Sense 0 is a valid fallback,
not a parse failure and not a claim that a novel sense was discovered.

Valence averages finite usage scores and includes Sense-0 usages; the target token
is excluded from the sentiment context. No matched polarity token gives a missing
score, not an invented neutral score. Metaphoricity includes only predictions 0/1;
error value -1 is omitted. Scope averages valid aligned-sense depths and excludes
Sense 0 and missing alignments. A wholly unaligned entry remains missing in the
implementation, although the paper describes a score of zero for that case.
Each numeric table includes its own valid-score denominator. Resolved sense
probabilities are renormalized over resolved assignments in that period.

## Extrinsic evaluation

The overlap experiment uses the seven overlapping entries' **main-corpus usage
predictions**, grouped by corpus identity. The full experiment uses the separate
20-word JaSemChange inference run, with the historical year split <=1925,
1926–1999, and >=2000. Computing the overlap subset from the full experiment is a
different experiment (and produces a different SHC correlation).

Both original relatedness and `4 - relatedness` change degree are reported. JSD uses
base-2 logarithms; it is divergence, not the square-root Jensen–Shannon distance.
Recomputed change-degree correlations:

| Cohort | CHJ–BCCWJ | SHC–BCCWJ |
|---|---:|---:|
| Overlap (7) | 0.750000 | 0.821429 |
| Full (20) | 0.366933 | 0.580256 |

The original full-set report's negative signs refer to relatedness. Pairwise
analysis retains 798 pairs (399 per corpus comparison). Strict means the highest
relatedness category; relaxed includes the highest two. These are agreements with
binarized relatedness, not word-sense classification accuracy.

## Trend analysis

The 150-entry cohort has 130 entries observed in at least ten decades. Recomputing
from the final predictions yields 366 eligible word–dimension series: 130 valence,
106 scope, and 130 metaphoricity. The historical tables used in the paper yield
365: the additional current scope series is 結構_ケッコウ. The default reports the
current result; `paper` accepts the two original per-run metric tables to reproduce
the historical cohort. Both identify 20 globally BH-significant monotonic trends
(2 valence, 6 scope, 12 metaphoricity). Constant series have undefined R² in this
release, slope 0 and Mann–Kendall p=1. Global BH uses all eligible series together.
No slope ranking alone establishes a historical semantic-change pattern.

## Scope

The reference results were recomputed from the supplied annotations and existing
model predictions. Full GPU inference and training were not rerun during packaging.
The original metaphor package, checkpoints and prompt example are separate local
inputs; see [RESOURCES.md](RESOURCES.md). The repository contains the experimental
pipeline and analysis code.
