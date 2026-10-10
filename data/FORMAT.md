# 数据格式：完整支持段落与 B 输入策略

当前阶段：六个统一输入文件已生成；B 输入策略已接入 `src.run_oracle`；已完成离线测试，真实 Qwen 生成仍需在 GPU 环境验证。

## 项目与目录

本轮指定的项目为 `/Users/cousin7/Desktop/multihop-rag`。

```text
data/
  raw/                         原始文件，保留镜像字节与所有原始标注
    hotpotqa/
    2wikimultihopqa/
    musique/
  processed/                   仅含正确支持段落的统一输入
    hotpotqa/{train,dev}.jsonl
    2wikimultihopqa/{train,dev}.jsonl
    musique/{train,dev}.jsonl
  backups/flashrag_20261009/    当前项目旧数据的副本
```

`raw/` 不覆盖、不裁剪。`processed/` 每行一个完整问题，使用 JSONL。
旧生成响应、检测结果和校准文件保留在原输出目录；新结果后续应使用独立目录。`data/FlashRAG_Data/` 和 `data/source_manifests/` 已移除，历史副本仍保存在 backups 中。

## 一条样本

下面以 MuSiQue 的真实 ID、问题和段落索引示意结构；`text` 位置仅为说明文字，正式导出必须是完整原始支持段落。

```json
{
  "id": "2hop__460946_294723",
  "dataset": "musique",
  "split": "dev",
  "question": "Who is the spouse of the Green performer?",
  "gold_answers": ["Miquette Giraudy"],
  "support_documents": [
    {
      "doc_id": "p10",
      "source_index": 10,
      "title": "Green (Steve Hillage album)",
      "text": "该支持段落的全部原始文本"
    },
    {
      "doc_id": "p5",
      "source_index": 5,
      "title": "Miquette Giraudy",
      "text": "该支持段落的全部原始文本"
    }
  ],
  "num_steps": 2,
  "dataset_hop_count": 2,
  "order_source": "official_decomposition",
  "metadata": {
    "source_file": "data/raw/musique/musique_ans_v1.0_dev.jsonl",
    "question_type": "2hop"
  }
}
```

## 字段约定

| 字段 | 含义 |
|---|---|
| `id` | 原始样本 ID，保留原值，不改成 `train_0` |
| `dataset`、`split` | 数据集与划分 |
| `question` | 原始问题，不改写 |
| `gold_answers` | 主答案和原始答案别名；保留原文，主答案在前 |
| `support_documents` | 仅含正确支持段落，数组顺序为执行顺序 |
| `doc_id` | 样本内段落标识，如 `p10`；这是程序标识，不冒充官方文档 ID |
| `source_index` | 该段落在原始样本中的索引，便于回查 |
| `title`、`text` | 原始标题和完整支持段落；保留全部句子 |
| `num_steps` | 实际生成次数，等于支持段落数组长度 |
| `dataset_hop_count` | 有明确数据集依据的推理跳数；没有可靠来源时为 `null` |
| `order_source` | 顺序来自官方分解、构造的依赖顺序或未经验证的标注顺序 |
| `metadata` | 原始文件位置、问题类型、支持事实、关系三元组、原始分解等溯源标注 |

`metadata` 中保留源数据具有的 `supporting_facts`、`evidences`、`question_decomposition` 等字段。原始中间标准答案与子问题用于离线核查、支持段落选择及排序，不输入模型；三个数据集统一使用原问题和模型自己的历史响应。

HotpotQA、2Wiki 的段落额外保留 `sentences` 数组，逐句原文可回查。`text` 仅将全部句子的首尾空白整理后用空格拼接；不删句、不改写句子。

文件包含 `schema_version: 1`；`metadata.source_row` 是原始文件中从 0 开始的记录位置。新输出保留 `num_steps` 与 `dataset_hop_count`，兼容字段 `hop_num` 的定义为执行步数。

统一输入文件不包含模型预测、生成响应、ReDeEP 分数或依据生成答案计算的幻觉标签；这些属于运行输出。
最终答案标签使用 [共享 EM/F1/yes-no 规则](ANSWER_LABELS.md)，原始生成文本与标签提取结果分别保存。

## 三个数据集的抽取规则

- HotpotQA：以 `supporting_facts` 标题定位 `context` 中的支持段落；保留全部句子。按标题首次出现的顺序输入，已逐条核对与旧 FlashRAG 的 `metadata.supporting_facts.title` 顺序一致。
- 2Wiki：同样定位完整支持段落，保留 `_id`、问题类型与 `evidences` 关系三元组。train 全量为 167,454 条，dev 为 12,576 条；此前排除的 dev 样本已恢复。旧 FlashRAG 的 15,000 条 train 是原始文件前 15,000 条，已逐条核对问题、支持标注和文档顺序。
- MuSiQue：使用 Ans 的 train/dev，核对 `answerable=true`；按官方分解中的 `paragraph_support_idx` 定位段落，并核对 `is_supporting=true`。原始候选段落列表保留在 raw 中，processed 仅输出对应的正确支持段落。

文档排序规则：

- MuSiQue：`official_decomposition`，使用原始步骤顺序。
- HotpotQA、2Wiki：`flashrag_supporting_facts_order`，固定使用支持事实标题首次出现的顺序；不再构造关系依赖图或使用 LLM 重排。
- HotpotQA train/dev、2Wiki dev、MuSiQue train/dev 已与旧 FlashRAG 全量核对；2Wiki train 核对了重合的 15,000 条，其余 152,454 条采用相同标题排序规则，没有声称逐条存在 FlashRAG 参考。
- HotpotQA、2Wiki 的标注列表顺序是实验输入约定，不声称是官方推理链，因此 `is_official_reasoning_order=false`。当前 `needs_order_review=false` 表示输入规则已固定。

2Wiki dev 的 `7a2249120bb011ebab90acde48001122` 已恢复，采用 FlashRAG 的支持标题顺序，关系标注完整保留。当前不排除任何样本，实验输入总计 300,237 条。

FlashRAG 用于核对文档顺序；段落正文和标准答案仍取原始数据。旧 FlashRAG HotpotQA dev 的候选上下文与原始数据多数不一致；2Wiki dev 的部分 `golden_answers` 另外包含别名，当前排序核对不将这些别名写入新数据。

HotpotQA、2Wiki 的原始文件没有逐样本显式跳数，因此 `dataset_hop_count` 保存为 `null`，使用 `num_steps` 表示实际执行次数。MuSiQue 的跳数取原始 ID 并与分解步骤数核对。2Wiki 的关系三元组数量可能与段落数量不同，不能直接据此填写推理跳数。

## 已确认的 B 输入策略

```text
第 1 步：原问题 + D1                         → 模型响应 R1
第 2 步：原问题 + D1、D2 + R1                → 模型响应 R2
第 3 步：原问题 + D1、D2、D3 + R1、R2        → 最终答案
```

每一步只增加一份新支持段落。已有段落和模型自己的响应保留，未来段落不可见。ReDeEP 的证据范围对应当步全部已出现的段落。

数据导出不裁支持句。生成阶段默认遇到超长完整证据就报错；显式设置 `--allow-evidence-truncation` 才允许截断，输出记录 `context_truncated=true`、`evidence_complete=false`。正式完整证据实验应保持默认行为。

## 核查与下一步

`scripts.prepare_original_datasets` 生成六个 JSONL 和 `preparation_report.json`；`scripts.validate_processed_datasets` 逐条对照全部原始样本并生成 `validation_report.json`。验证包括原始 ID、问题答案、完整支持段落、官方分解和固定输入顺序。

`python -m scripts.validate_flashrag_order` 是可选历史核查，默认读取 `data/backups/flashrag_20261009/FlashRAG_Data/`，输出 `data/processed/flashrag_order_report.json`。日常运行和原始数据核查不需要 FlashRAG 参考文件。

raw 的三个数据集目录只保留六个实验数据文件。下载记录、MuSiQue-Full 和下载元数据保存在 `data/backups/raw_cleanup_20261009/`。数量与分组统计见 `data/raw/README.md`。

下一步先在 GPU 环境对三个数据集运行小规模 B 生成，确认提示词、显存和长度限制，再用新 train/dev trace 重新 fit/evaluate。旧校准文件来自旧提示词和证据设置，应与新实验分开。
