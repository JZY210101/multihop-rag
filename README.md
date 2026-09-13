# MultiHop RAG（FlashRAG 基线）

直接使用 FlashRAG 原生的 `Config`、`get_dataset`、`get_retriever`、`get_generator` 和 `BasicPipeline`，只在 Pipeline 层增加固定 hop 控制与 trace 记录。FlashRAG 官方已预处理 HotpotQA、2WikiMultiHopQA、MuSiQue；MultiHop-RAG 若不在当前 FlashRAG 数据包中，需要按同样的 JSONL 格式放入 `data_dir/multihop-rag/`。

## 安装

```bash
pip install -r requirements.txt
```

注意：这里使用 GitHub 官方 FlashRAG 仓库，不使用 PyPI 上同名的非官方/不完整包。

## 运行

```bash
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml
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
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml --dataset_name musique
```

问答数据与检索语料是两类文件。运行 BM25 前还需准备配置中 `corpus_path` 指向的
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

三个数据集没有统一的人工幻觉标签，因此数据处理阶段统一使用答案 token-F1 生成答案级
弱标签。预测答案会经过小写化、去标点、去英文冠词和空白归一化；存在多个 gold answer
别名时取最高 F1：

```text
answer_f1 = max(token_f1(prediction, gold_answer_i))
answer_f1 <= 0.30  -> hallucination_label = 1
answer_f1 >  0.30  -> hallucination_label = 0
```

阈值可在 `configs/flashrag_fixed_hop.yaml` 中通过
`hallucination_f1_threshold` 配置。固定 hop pipeline 的输出会直接保存
`answer_f1`、`hallucination_label`、`hallucination_label_method`、
`hallucination_f1_threshold` 和 `retrieval_sufficient`，ReDeEP 与后续自研方法共用
`src/hallucination_labels.py`，不再分别实现标签规则。

这是衡量最终答案正确性的弱标签，不是对回答中每个 span 的人工幻觉标注。如果启用
retrieval-aware 评估，支持文档未被检索到的样本保留
`retrieval_sufficient: false`，最终标签设为 `null`，不进入校准和指标计算。

## Qwen ReDeEP 幻觉检测

当前 ReDeEP 适配使用 `Qwen/Qwen2.5-7B-Instruct` 提取内部状态，不使用原始
`ReDEeP-ICLR` 中的 LLaMA 实现，也不包含 AARF 干预。检测器计算完整的 ECS + PKS
联合分数，支持 token-level 和 chunk-level 两种粒度。
本适配只输出完整 ECS + PKS 联合 ReDeEP；不实现 AARF，也不生成 ECS-only、PKS-only
或其他对照组结果。

先运行多跳 pipeline，生成包含 `fixed_hop_trace`、`answer_prompt` 和 `redeep_records`
的输出。然后使用训练集校准 ReDeEP 的 attention heads、FFN layers、归一化范围和
联合权重，再在 dev 集固定校准参数评估：

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
  --embedding-model BAAI/bge-base-en-v1.5
```

三个数据集分别运行时，将输入文件和 calibration 文件按数据集命名即可。默认
`--label-mode f1_answer --f1-threshold 0.30`；若要排除支持文档未被检索到的样本，使用
`--label-mode f1_retrieval_aware`。旧名称 `weak_answer` 和 `retrieval_aware` 仅作为兼容
别名，计算仍使用共享 token-F1 规则，不再使用 Exact Match。

Qwen ReDeEP 每条样本需要对完整 prompt + response 做一次全序列 forward，并返回所有
attention 矩阵和 FFN residual，因此实际运行需要 GPU、足够显存和 `requirements.txt`
中的 `torch`、`transformers`、`flashrag` 依赖。模型权重、索引、数据和输出均已在
`.gitignore` 中排除。

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
