# MultiHop RAG（Oracle Evidence 逐跳基线）

当前数据与生成入口已切换为原始数据的完整支持段落：`data/raw/` 保存 ModelScope 原始文件，`data/processed/` 保存统一 JSONL。B 方式每步加入一个段落，同时保留已有段落和模型响应。

三个数据集均保留原始全量，共 300,237 条。HotpotQA、2Wiki 沿用支持事实标题顺序；MuSiQue 沿用官方分解顺序。所有与旧 FlashRAG 重合的样本已核对文档顺序；2Wiki train 其余样本采用同一规则。样本数量和分组统计见 [raw 数据说明](data/raw/README.md)。

新流程说明见 [数据格式](data/FORMAT.md) 和 [原始数据运行指南](src/redeep/原始数据运行指南.md)。后文含旧版 FlashRAG/检索流程的历史说明；当前实验按新指南执行。

三个数据集的真实 Qwen 小规模检查可直接运行下列命令：自动生成 train/dev、重新 fit、evaluate，
检查共享标签、完整证据、ReDeEP 分数和 token IDs，并在输出目录保存 `smoke_report.json`。

```bash
python -m scripts.smoke_test_redeep --train-records 64 --dev-records 16 \
  --generation-batch-size 4 --score-batch-size 2
```

当前数据运行只需要 `data/raw/` 和 `data/processed/`。旧 `data/FlashRAG_Data/` 已移除，后文引用该路径的历史命令不适用于当前目录；历史备份保存在 `data/backups/`。

```bash
python -m scripts.validate_processed_datasets
python -m src.run_oracle \
  --input data/processed/musique/dev.jsonl \
  --dataset musique --model model/Qwen3-4B-Instruct-2507 \
  --max-records 5 --batch-size 1 --progress-every 1 \
  --output outputs/original_b/smoke/musique_dev_oracle.jsonl
```

当前正式实验只研究“检索正确时的生成幻觉”。不运行 Wiki18、BM25 或其他检索器，直接使用
数据集提供的支持证据，并按 hop 逐步生成中间结果和最终答案。这样可以把检索失败与生成
幻觉分离。`configs/flashrag_fixed_hop.yaml` 和 `src.run` 保留为旧的检索式兼容入口；正式
实验使用 `src.run_oracle` 和 `configs/oracle_iterative.yaml`。

直接使用 FlashRAG 原生的 `Config`、`get_dataset`、`get_retriever`、`get_generator` 和 `BasicPipeline`，只在 Pipeline 层增加固定 hop 控制与 trace 记录。FlashRAG 官方已预处理 HotpotQA、2WikiMultiHopQA、MuSiQue；MultiHop-RAG 若不在当前 FlashRAG 数据包中，需要按同样的 JSONL 格式放入 `data_dir/multihop-rag/`。

## 安装

```bash
pip install -r requirements.txt
```

注意：这里使用 GitHub 官方 FlashRAG 仓库，不使用 PyPI 上同名的非官方/不完整包。
raw/processed 数据文件使用 Git LFS；clone 本仓库前需安装 Git LFS，clone 后可执行
`git lfs pull` 确认数据已完整下载。

## 运行：正式 oracle 逐跳流程

先用少量样本检查证据适配和 Qwen 生成：

```bash
PYTHONPATH=. python -m src.run_oracle \
  --input data/processed/musique/dev.jsonl \
  --dataset musique \
  --model model/Qwen3-4B-Instruct-2507 \
  --max-records 5 \
  --output outputs/musique_dev_oracle.jsonl
```

输出中的每条记录包含 `hop_records`、每一跳的 prompt/response/token ids、最终答案 F1 和
`hallucination_label`。每一跳只读取该 hop 的 gold supporting evidence，后一跳会接收前一跳
的中间结果。正式 ReDeeP 评分仍通过 `src.run_redeep` 对生成的最终记录进行校准和评估。

严格 oracle 模式要求数据文件中的支持标题确实存在于上下文中；如果缺失，程序会明确报错，
不会把候选上下文伪装成 gold evidence。`--allow-context-fallback` 只用于 smoke test，产生
的记录会标记为启发式 fallback，不应作为正式实验结果。

当前仓库的 FlashRAG HotpotQA 导出在部分样本中缺少原始支持页面。可用官方 HotpotQA
文件补齐后再运行正式实验（脚本不会覆盖原始 JSONL）：

```bash
PYTHONPATH=. python scripts/prepare_hotpot_gold.py --split train \
  --input data/FlashRAG_Data/hotpotqa/train.jsonl \
  --output data/FlashRAG_Data/hotpotqa/train_gold.jsonl
PYTHONPATH=. python scripts/prepare_hotpot_gold.py --split dev \
  --input data/FlashRAG_Data/hotpotqa/dev.jsonl \
  --output data/FlashRAG_Data/hotpotqa/dev_gold.jsonl
```

脚本默认从 HotpotQA 官方 GitHub 下载原始文件，也可以用 `--source /path/to/file.json`
指定已下载文件。匹配依据是规范化后的 question+answer，并会检查每条样本都成功匹配。

## 旧版检索式兼容流程

```bash
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml \
  --dataset_name hotpotqa --split dev \
  --output outputs/hotpotqa_dev_fixed_hop.json
```

FlashRAG 官方数据集需要先下载到本地数据目录。FlashRAG 的标准结构是：

```text
data/FlashRAG_Data/hotpotqa/{train,dev}.jsonl
data/FlashRAG_Data/musique/{train,dev}.jsonl
data/FlashRAG_Data/2wikimultihopqa/{train,dev}.jsonl
data/wiki_corpus.jsonl
```

这三个官方预处理数据集没有带答案的 `test.jsonl`，默认使用 `dev`。切换数据集：

```bash
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml \
  --dataset_name musique --split dev \
  --output outputs/musique_dev_fixed_hop.json
```

旧版检索式流程中，问答数据与检索语料是两类文件。运行 BM25 前还需准备配置中 `corpus_path` 指向的
`data/wiki_corpus.jsonl`，并建立与配置中 `index_path` 一致的 BM25s 索引：

```bash
python -m flashrag.retriever.index_builder \
  --retrieval_method bm25 \
  --corpus_path data/wiki_corpus.jsonl \
  --bm25_backend bm25s \
  --save_dir data/index
```

该命令生成 `data/index/bm25/`。语料不存在或索引路径错误时，运行入口会在加载模型前给出明确错误。

支持 `hotpotqa`、`musique`、`2wikimultihopqa`、`multihop-rag`。每行输出包含问题、答案、hop 数、每跳 query、检索文档及分数，供后续幻觉检测使用。

## 共享幻觉弱标签

三个数据集没有统一的人工幻觉标签，因此统一使用最终答案 EM/token-F1 构造弱标签。
预测与全部标准答案使用相同基础规范化：小写、Unicode/空白归一化，保留冠词和有效标点；
多个原始答案别名取最高分。完整规则见 [答案标签说明](data/ANSWER_LABELS.md)。

```text
answer_em = 1 或 answer_f1 >= 0.30 -> hallucination_label = 0
否则                              -> hallucination_label = 1
```

标准答案为 yes/no 时，对原始回答进行完整词匹配：只出现一种就提取并精确比较；两种同时
出现或两种都未出现均标为幻觉。原始 `prediction` 与 token IDs 保留，提取结果另存
`answer_for_label`。普通实体问题不启用此提取规则。

阈值可在 `src.run_oracle`/`src.run_redeep` 中通过 `--f1-threshold` 指定；旧固定 hop 配置通过
`hallucination_f1_threshold` 配置。固定 hop pipeline 的输出会直接保存
`answer_em`、`answer_f1`、标签、判定原因和规则版本，ReDeEP 与后续自研方法共用
`src/hallucination_labels.py`，不再分别实现标签规则。

这是衡量最终答案正确性的弱标签，不是对回答中每个 span 的人工幻觉标注。如果启用
retrieval-aware 评估，支持文档未被检索到的样本保留
`retrieval_sufficient: false`，最终标签设为 `null`，不进入校准和指标计算。

## Qwen ReDeEP 幻觉检测

当前 ReDeEP 适配使用本地 `model/Qwen3-4B-Instruct-2507` 提取内部状态。模型通过 ModelScope
下载，不使用 Hugging Face 在线下载，也不使用原始
`ReDEeP-ICLR` 中的 LLaMA 实现，也不包含 AARF 干预。检测器计算完整的 ECS + PKS
联合分数，支持 token-level 和 chunk-level 两种粒度。
本适配只输出完整 ECS + PKS 联合 ReDeEP；不实现 AARF，也不生成 ECS-only、PKS-only
或其他对照组结果。

先运行多跳 pipeline，分别生成包含 `fixed_hop_trace`、`answer_prompt` 和
`redeep_records` 的 train/dev 输出：

```bash
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml \
  --dataset_name hotpotqa --split train \
  --output outputs/hotpotqa_train_fixed_hop.json

PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml \
  --dataset_name hotpotqa --split dev \
  --output outputs/hotpotqa_dev_fixed_hop.json
```

然后使用训练集校准 ReDeEP 的 attention heads、FFN layers、归一化范围和联合权重，再在
dev 集固定校准参数评估：

```bash
# 训练集校准，并输出训练集分数和 calibration 文件
PYTHONPATH=. python -m src.run_redeep fit \
  --input outputs/hotpotqa_train_fixed_hop.json \
  --output outputs/redeep_hotpotqa_train.jsonl \
  --calibration outputs/redeep_hotpotqa.json

# dev 集只评估，不重新选择 feature
PYTHONPATH=. python -m src.run_redeep evaluate \
  --input outputs/hotpotqa_dev_fixed_hop.json \
  --output outputs/redeep_hotpotqa_dev.jsonl \
  --calibration outputs/redeep_hotpotqa.json
```

chunk-level 需要额外指定 BGE embedding，例如：

```bash
PYTHONPATH=. python -m src.run_redeep fit \
  --input outputs/hotpotqa_train_fixed_hop.json \
  --output outputs/redeep_hotpotqa_chunk_train.jsonl \
  --calibration outputs/redeep_hotpotqa_chunk.json \
  --granularity chunk \
  --embedding-model model/bge-base-en-v1.5
```

三个数据集分别运行时，将输入文件和 calibration 文件按数据集命名即可。默认
`--label-mode f1_answer --f1-threshold 0.30`；若要排除支持文档未被检索到的样本，使用
`--label-mode f1_retrieval_aware`。旧名称 `weak_answer` 和 `retrieval_aware` 仅作为兼容
别名；这些参数名保持兼容，实际都使用共享 EM/F1 与 yes/no 规则。
标签规则已更新，旧校准文件不能直接用于新版 evaluate，需要重新 fit。

Qwen ReDeEP 每条样本需要对完整 prompt + response 做一次全序列 forward，并返回所有
attention 矩阵和 FFN residual，因此实际运行需要 GPU、足够显存和 `requirements.txt`
中的 `torch`、`transformers`、`flashrag` 依赖。模型权重、索引和输出不会提交；三个问答
数据集使用 Git LFS 管理并随仓库上传。

生成 prompt 默认限制为 3840 tokens，并为最多 256 个回答 token 预留空间；ReDeEP 的
完整 question + evidence + response 默认上限为 4096 tokens。生成前若 evidence 超长，会
同时保留最早和最后部分，并保存实际使用的 `prompt_parts` 以及 Qwen 实际生成的
`response_token_ids`。ReDeEP 使用这两部分复现原生成序列，不会在检测阶段静默改写 prompt；
旧输出没有 token IDs 时会退回文本分词。完整序列若仍超过 `--max-input-tokens`，该样本会
明确标为不可评分，需要降低 `generator_max_input_len` 后重新生成。需要更长上下文时可同时
调整两个上限，但 eager attention 的显存开销随序列长度近似二次增长。
HotpotQA 训练集约 9 万条，不建议第一次运行就处理全部数据；可以先加
`--max-records 500` 建立包含两类标签的 calibration 子集，最终 dev 评估时去掉该参数。
正式处理完整数据集建议增加 `--no-token-scores`，保留聚合 ECS/PKS 和最终分数，仅省略
体积很大的逐 token 特征数组。
