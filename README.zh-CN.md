# LSC-Pattern

[English](README.md) · [日本語](README.ja.md) · **简体中文**

论文 **LSC-Pattern: Interpretable Lexical Semantic Change Analysis via Diachronic
Sense Distributions**（Ling 与 Komachi，EMNLP 2026）的代码和人工标注。

流水线先为日语历史用例分配辞典语义，再按年代分析上下文情感、语义范围和隐喻性。
主实验包含 150 个词；人工标注包含 50 个词的 976 条用例，每条有三位匿名标注者的标签。

## 安装

使用 Python 3.12。完整推理需要 NVIDIA GPU 和兼容的 CUDA 环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gpu.txt
```

如果只对已有预测计算统计结果，安装 `requirements.txt` 即可。

## 准备资源

仓库不包含语料原文、辞典条目、benchmark 数据或模型权重。请自行准备处理好的
CHJ、SHC、BCCWJ CSV，岩波辞典、WordNet、情感辞典、WLSP、JaSemChange，
以及 Qwen3-8B 和基线／隐喻模型的权重。文件格式和获取入口见 [RESOURCES.md](docs/RESOURCES.md)。

```bash
mkdir -p .local
cp config/example.env .local/config.env
# 编辑 .local/config.env，填写资源路径和 GPU 参数。
```

`PROMPT_TEMPLATE` 指向包含原示例的提示文件。该示例需要单独准备，不能仅凭语料
和辞典自动恢复。准备方法见 [RESOURCES.md](docs/RESOURCES.md#prompt-template)。

## 运行

```bash
bash pipeline/run.sh --config .local/config.env --stage all --mode normalized
```

流水线依次执行下表中的阶段。将 `all` 换成阶段名即可单独运行。

| 阶段 | 功能 |
|---|---|
| `prepare` | 重建语义清单、对齐 WordNet、提取用例并生成提示 |
| `restore` | 从本地语料恢复人工标注的上下文 |
| `wsd` | 用 Qwen3-8B 预测语义 |
| `dimensions` | 计算上下文情感和隐喻性，合并预测 |
| `aggregate` | 按年代汇总语义分布和三个维度的分数 |
| `baselines` | 运行 WordNet BERT、用例—释义和目标词—释义检索基线 |
| `evaluate` | 评估 Qwen 和含 Random、MFS 在内的五种基线，计算标注者一致性 |
| `jasemchange` | 评估 7 词、20 词及用例对 |
| `trends` | 计算 OLS、Theil–Sen、Mann–Kendall 和 BH 校正 |
| `figures` | 输出三个案例的 PNG 和 PDF 图 |

默认结果目录为 `.local/runs/normalized/`。可用 `--run-dir .local/runs/NAME` 指定
其他目录。单独运行某阶段时，需要先准备前面阶段的中间文件。
另可用 `corpus-stats` 统计语料 token 数和年份。

论文参数为 temperature 0、开启 thinking、batch 64、最多生成 16,384 tokens、
显存利用率 0.92。示例配置使用一张 GPU；四卡运行设置 `GPU_IDS=0,1,2,3` 和 `GPU_WORKERS=4`。

## 使用已有预测

在配置中设置 `EXISTING_GENERATED`、`EXISTING_BASELINES` 和 `EXISTING_JASEM_GENERATED`，然后运行：

```bash
for stage in aggregate evaluate jasemchange trends figures; do
  bash pipeline/run.sh --config .local/config.env --stage "$stage"
done
```

默认 `normalized` 模式使用已确认的人工—模型语义对应关系，排除无法确定的标签。
`--mode paper` 使用历史计算规则；复现历史趋势还需将 `HISTORICAL_METRICS` 设置为两个原始 CSV 的路径列表。
两种模式的评估集合不同：Qwen 分别为 776/924 和 783/961，不能据此声称模型提升。

## 数据与说明

[annotations.csv](data/annotations.csv) 保存语料位置和三人的标签，不含原文及辞典释义。
其中 CHJ 308 条、SHC 503 条、BCCWJ 165 条。Sense 0 是有效的兜底标签，空值表示缺失。

- [DATA.md](docs/DATA.md)：字段、标签规则和上下文恢复。
- [EXPERIMENTS.md](docs/EXPERIMENTS.md)：论文表图对应的命令和输出。
- [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)：历史差异与参考结果。

上下文情感反映周围文本的评价倾向，WordNet 深度近似表示语义具体程度，隐喻性来自模型预测。
趋势需要结合语义分布解释。

## 许可与引用

原创代码与文档采用 [MIT](LICENSE)，原创人工标签采用 [CC BY 4.0](LICENSE-DATA.md)。
第三方资源遵循各自的使用条件。

请引用 Ling 与 Komachi（2026）的 LSC-Pattern 论文。BibTeX 见
[英语 README](README.md#license-and-citation)，引用信息另见 [CITATION.cff](CITATION.cff)。
