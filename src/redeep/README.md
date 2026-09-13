# Qwen ReDeEP 多跳适配

## 1. 目标与范围

原始 ReDeEP 面向单跳 RAG，输入可概括为：

```text
question -> retrieved context -> response
```

当前版本适配固定 hop 的多跳 RAG：

```text
question
  -> hop 1 query + documents
  -> hop 2 query + documents
  -> ...
  -> final response
  -> ECS + PKS 联合 ReDeEP 分数
```

支持 HotpotQA、2WikiMultiHopQA 和 MuSiQue。使用
`Qwen/Qwen2.5-7B-Instruct` 提取内部状态，不使用 LLaMA，不包含 AARF，也不输出
ECS-only、PKS-only 等对照结果。

## 2. 多跳修改

- `FixedHopPipeline` 保存每一跳的 query、检索文档、hop 编号和文档 id。
- 所有 hop 的证据按原顺序合并，格式为 `[hop=1 doc_id=...] text`。
- 检测时使用 pipeline 保存的 `prompt_parts` 和实际生成的 `response_token_ids`，避免重建
  prompt 或重新分词回答造成 token 偏移；旧输出没有 token IDs 时才退回文本分词。
- ECS 只在多跳检索证据区域内寻找高注意力 token/chunk，不把问题模板当作证据。
- PKS 对最终回答的生成位置计算各 FFN 层前后词表分布的差异。
- 在 train 输出上选择 attention heads 和 FFN layers、拟合联合分数，再固定参数评估 dev。

示例：问题需要先找到人物，再查人物出生地。两个 hop 的文档都会进入 context；当 Qwen
生成最终答案时，ECS 检查答案是否依赖这些外部证据，PKS 检查 FFN 参数知识是否与当前
生成倾向冲突，二者共同给出幻觉分数。

## 3. 实现目录

```text
src/redeep/
├── io.py               # 读取 trace、构造多跳 prompt、接入共享弱标签
├── qwen_extractor.py   # Qwen attention、hidden state 和 FFN residual
├── scores.py           # token-level ECS 与 PKS
├── chunk_scores.py     # chunk-level ECS 与 PKS
├── calibration.py      # 特征选择、归一化和联合校准
└── detector.py         # 端到端检测及 JSONL 输出

src/run_redeep.py       # fit/evaluate 命令行入口
src/hallucination_labels.py # 全框架共享的答案 F1 与检索充分性规则
configs/redeep.yaml     # 默认检测配置
```

## 4. Qwen 内部状态与分数

模型以 `attn_implementation="eager"` 加载，因为 ECS 必须取得真实 attention 矩阵。
前向只调用 Qwen backbone，不计算完整 CausalLM logits，从而减少显存峰值。

Qwen2.5 decoder 是 pre-norm 结构：

- 在 `post_attention_layernorm` 输入处记录 FFN 前 residual；
- 在 decoder layer 输出处记录 FFN 后 hidden state；
- 使用最终 RMSNorm 和 `lm_head` 做 Logit Lens 投影；
- 最终层 hidden state 用于 ECS 余弦相似度。

因果模型位置 `p` 用来预测 token `p+1`。因此回答 token 的 ECS/PKS 使用其前一位置的
attention 和 hidden state，避免把待检测 token 本身泄漏给检测特征。

token-level 计算：

- ECS：每个候选 head 对证据区域取 attention 最高的 10% token，将其 hidden state
  平均后与当前生成状态计算余弦相似度；ECS 越低，幻觉风险通常越高。
- PKS：对 FFN 前后状态分别进行 RMSNorm 和 `lm_head` 投影，计算两个词表分布的 JSD；
  PKS 越高，幻觉风险通常越高。

chunk-level 先用 attention 选择最相关的证据 chunk，再比较该证据 chunk 与回答 chunk。
建议配置 `BAAI/bge-base-en-v1.5`，与论文的句段语义比较更一致；不配置 embedding 模型时，
代码会回退到 Qwen hidden-state cosine，输出中的 `embedding_backend` 会标明实际后端。

## 5. 标签与校准

`fit` 必须使用已经完成多跳检索和答案生成的 train 输出，不能直接输入下载后的原始
`train.jsonl`。校准数据必须同时包含标签 0 和 1。

ReDeEP 与其他幻觉检测方法共用 `src/hallucination_labels.py`。预测和 gold answer 先进行
小写化、去标点、去英文冠词和空白归一化；多个 gold 别名取最大 token-F1：

```text
answer_f1 <= 0.30 -> hallucination_label = 1
answer_f1 >  0.30 -> hallucination_label = 0
```

- `f1_answer`：直接使用上述答案级弱标签。
- `f1_retrieval_aware`：支持文档未检索到时最终标签为 `null`，不参与校准和指标；否则
  使用同一 F1 标签。证据充分性按 supporting title/doc id 匹配，不是 supporting
  sentence 级判断。

旧名称 `weak_answer`、`retrieval_aware` 仅保留为兼容别名，内部不再使用 Exact Match。
该标签反映最终答案正确性，不等同于人工标注的回答 span 级幻觉。

校准流程为：按训练集 AUC 选择 top heads/layers，对 ECS 和 PKS 做 min-max 归一化，
再在 train 上选择联合权重 `alpha`，最终分数保持原始 ReDeEP 的组合形式：

```text
ReDeEP = normalized_PKS - alpha * normalized_ECS
```

dev 只加载 train 保存的 heads、layers、归一化范围、`alpha` 和分类阈值，不重新选择参数。
校准文件同时保存模型、token/chunk 粒度、embedding 模型和标签设置；evaluate 配置不一致
时会在加载 Qwen 前停止，避免得到不可比较的分数。

## 6. 配置

默认配置位于 `configs/redeep.yaml`，命令行参数优先于 YAML。

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `model_name` | `Qwen/Qwen2.5-7B-Instruct` | 内部状态提取模型 |
| `device_map` | `auto` | Accelerate 模型放置；单卡可设 `none` 并指定 `device` |
| `torch_dtype` | `auto` | 可设 `float16`、`bfloat16` 或 `float32` |
| `granularity` | `token` | `token` 或 `chunk` |
| `label_mode` | `f1_answer` | `f1_answer` 或 `f1_retrieval_aware` |
| `f1_threshold` | `0.30` | F1 小于或等于该值时标为幻觉 |
| `top_heads` | `8` | train 上保留的 ECS heads 数量 |
| `top_layers` | `8` | train 上保留的 PKS layers 数量 |
| `top_fraction` | `0.10` | token ECS 选取的高注意力证据比例 |
| `pks_batch_size` | `8` | Logit Lens 投影批大小 |
| `chunk_size` | `400` | chunk 字符数上限 |
| `embedding_model` | `null` | chunk ECS 的 BGE 模型；空值时使用 Qwen hidden state |
| `max_input_tokens` | `4096` | question + evidence + response 的 token 上限 |
| `max_records` | `null` | 可选的样本前缀限制 |

eager attention 的显存开销随序列长度近似二次增长。超长 evidence 应由生成 pipeline 在
生成前同时保留首尾并截断；ReDeEP 不会修改已用于生成的 prompt。保存的 prompt + response
超过 `max_input_tokens` 时，该样本输出 `redeep_score: null` 和明确的 `redeep_error`。显存不足
时，应同时降低生成配置中的 `generator_max_input_len` 和这里的上限，再重新生成 trace。

## 7. 运行

先安装依赖并准备 `data/wiki_corpus.jsonl`：

```bash
pip install -r requirements.txt
```

分别生成 train 和 dev 的多跳 trace。下面以 HotpotQA 为例；另外两个数据集将名称替换为
`2wikimultihopqa` 或 `musique`：

```bash
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml \
  --dataset_name hotpotqa --split train
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml \
  --dataset_name hotpotqa --split dev
```

输出位置由 FlashRAG 的 `save_dir` 决定，以终端显示和实际生成文件为准。然后独立完成
train 校准和 dev 评估：

```bash
PYTHONPATH=. python -m src.run_redeep fit \
  --config configs/redeep.yaml \
  --input outputs/hotpotqa_train_fixed_hop.json \
  --output outputs/redeep_hotpotqa_train.jsonl \
  --calibration outputs/redeep_hotpotqa.calibration.json

PYTHONPATH=. python -m src.run_redeep evaluate \
  --config configs/redeep.yaml \
  --input outputs/hotpotqa_dev_fixed_hop.json \
  --output outputs/redeep_hotpotqa_dev.jsonl \
  --calibration outputs/redeep_hotpotqa.calibration.json
```

chunk-level 推荐增加：

```bash
--granularity chunk --embedding-model BAAI/bge-base-en-v1.5
```

第一次运行可增加 `--max-records 500` 建立校准子集；若其中只有一种标签，需要扩大样本数。
正式 dev 评估应移除 `--max-records`。三个数据集建议分别使用各自 train calibration，不能
用 dev 重新选择 heads/layers。完整数据集建议增加 `--no-token-scores`：`ecs`、`pks` 的
head/layer 聚合特征和最终分数仍会保留，只省略体积很大的逐 token 数组。

每条输出主要包含 `redeep_score`、`redeep_prediction`、`ecs`、`pks`、`answer_f1`、
`hallucination_label`、`base_hallucination_label`、`hallucination_label_method`、
`hallucination_f1_threshold`、`retrieval_sufficient`、`response_token_ids`、token span 和多跳 trace。终端汇总
输出 ROC-AUC、F1、precision、recall、PCC 及可计算时的按 hop AUC。
