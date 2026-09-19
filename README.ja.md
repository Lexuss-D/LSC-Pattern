# LSC-Pattern

[English](README.md) · **日本語** · [简体中文](README.zh-CN.md)

**LSC-Pattern: Interpretable Lexical Semantic Change Analysis via Diachronic
Sense Distributions**（Ling・Komachi、EMNLP 2026）のコードと人手アノテーションです。

日本語の歴史的用例に辞書語義を割り当て、文脈の感情価、意味範囲、比喩性を年代ごとに
分析します。主実験は 150 語、人手アノテーションは 50 語・976 用例です。
各用例に 3 名の匿名アノテータによるラベルがあります。

## インストール

Python 3.12 を使用します。全量推論には NVIDIA GPU と対応する CUDA 環境が必要です。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
```

既存予測から統計量を計算する場合は、`requirements.txt` のみをインストールします。

## 資源の準備

コーパス本文、辞書項目、ベンチマークデータ、モデル重みは含みません。
前処理済みの CHJ・SHC・BCCWJ CSV、岩波辞典、WordNet、感情極性辞書、WLSP、
JaSemChange、Qwen3-8B、およびベースライン・比喩モデルの重みを用意してください。
形式と入手先は [RESOURCES.md](docs/RESOURCES.md) にあります。

```bash
mkdir -p .local
cp config/example.env .local/config.env
# .local/config.env に資源のパスと GPU 設定を記入します。
```

`PROMPT_TEMPLATE` には元の例を含むプロンプトファイルを指定します。この例は別途必要で、
コーパスと辞書だけからは自動復元できません。準備方法は
[RESOURCES.md](docs/RESOURCES.md#prompt-template) を参照してください。

## 実行

```bash
bash pipeline/run.sh --config .local/config.env --stage all --mode normalized
```

以下の順に実行します。`all` を段階名に置き換えると個別に実行できます。

| 段階 | 処理 |
|---|---|
| `prepare` | 語義一覧の再構築、WordNet 対応付け、用例抽出、プロンプト作成 |
| `restore` | ローカルのコーパスから人手アノテーションの文脈を復元 |
| `wsd` | Qwen3-8B による語義予測 |
| `dimensions` | 文脈の感情価と比喩性の計算、予測の結合 |
| `aggregate` | 年代ごとの語義分布と 3 指標の集計 |
| `baselines` | WordNet BERT、用例–語釈、対象語–語釈の検索ベースライン |
| `evaluate` | Qwen と Random・MFS を含む 5 ベースラインの評価、アノテータ間一致度 |
| `jasemchange` | 7 語・20 語の評価と用例対の分析 |
| `trends` | OLS、Theil–Sen、Mann–Kendall、BH 補正 |
| `figures` | 3 事例の図を PNG・PDF で出力 |

既定の出力先は `.local/runs/normalized/` です。`--run-dir .local/runs/NAME` で変更できます。
個別実行では、それ以前の段階で作る中間ファイルが必要です。
`corpus-stats` でコーパスの token 数と年代も集計できます。

論文では temperature 0、thinking 有効、batch 64、最大生成長 16,384 tokens、
GPU メモリ使用率 0.92 を使用しました。設定例は GPU 1 台です。4 台で実行する場合は
`GPU_IDS=0,1,2,3` と `GPU_WORKERS=4` を指定します。

## 既存予測の利用

設定ファイルに `EXISTING_GENERATED`、`EXISTING_BASELINES`、`EXISTING_JASEM_GENERATED`
を指定し、以下を実行します。

```bash
for stage in aggregate evaluate jasemchange trends figures; do
  bash pipeline/run.sh --config .local/config.env --stage "$stage"
done
```

既定の `normalized` は確認済みの人手–モデル語義対応を使い、未確定のラベルを除外します。
`--mode paper` は旧計算規則を使います。旧傾向表の再現には `HISTORICAL_METRICS` に
元の 2 CSV のパスを JSON 配列で指定します。評価集合が異なり、Qwen の正解数は
それぞれ 776/924 と 783/961 です。この差はモデル改善を意味しません。

## データと説明

[annotations.csv](data/annotations.csv) はコーパス内の位置と 3 名分のラベルを収録し、
本文や辞書語釈は含みません。内訳は CHJ 308 件、SHC 503 件、BCCWJ 165 件です。
Sense 0 は該当語義なし等を表す有効なラベルで、空欄は欠損値です。

- [DATA.md](docs/DATA.md)：各フィールド、ラベル規則、文脈の復元。
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)：論文の表・図とコマンド・出力の対応。
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)：旧結果との差異と参照値。

感情価は周辺文脈の評価傾向、WordNet 深度は語義の具体性の近似、比喩性はモデルの予測です。
傾向は語義分布と合わせて解釈してください。

## ライセンスと引用

著者のコード・文書は [MIT](LICENSE)、独自の人手ラベルは [CC BY 4.0](LICENSE-DATA.md)
です。第三者の資源には各提供元の条件が適用されます。

Ling・Komachi（2026）の LSC-Pattern 論文を引用してください。BibTeX は
[英語 README](README.md#license-and-citation)、引用情報は [CITATION.cff](CITATION.cff) にあります。
