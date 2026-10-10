# 原始数据与样本统计

三个数据集均保留原始全量 train/dev，共 **300,237 条**。正式实验使用 MuSiQue-Ans；本目录的三个数据集目录只保留使用的 JSON/JSONL 数据文件。

| 数据集 | train | dev | 合计 |
|---|---:|---:|---:|
| HotpotQA | 90,447 | 7,405 | 97,852 |
| 2WikiMultiHopQA | 167,454 | 12,576 | 180,030 |
| MuSiQue-Ans | 19,938 | 2,417 | 22,355 |
| 合计 | 277,839 | 22,398 | 300,237 |

## 跳数与输入步数分布

| 数据集 | 分组 | train | dev |
|---|---|---:|---:|
| HotpotQA | 2 个支持段落／2 步 | 90,447 | 7,405 |
| 2WikiMultiHopQA | 2 个支持段落／2 步 | 132,823 | 9,825 |
| 2WikiMultiHopQA | 4 个支持段落／4 步 | 34,631 | 2,751 |
| MuSiQue-Ans | 2 跳／2 步 | 14,376 | 1,252 |
| MuSiQue-Ans | 3 跳／3 步 | 4,387 | 760 |
| MuSiQue-Ans | 4 跳／4 步 | 1,175 | 405 |

MuSiQue 跳数来自原始 ID，并与官方分解核对。HotpotQA、2Wiki 的分组按正确支持段落数统计，表示程序输入步数；不能仅据此给 2Wiki 样本添加官方 4-hop 标签。

## 文档顺序与使用

HotpotQA、2Wiki 按 `supporting_facts` 中支持标题首次出现的顺序输入；MuSiQue 按官方分解顺序输入。与旧 FlashRAG 重合的样本已逐条核对顺序。2Wiki train 的 FlashRAG 参考只有 15,000 条，其余 152,454 条采用同一标题排序规则。

模型读取 `data/processed/<dataset>/{train,dev}.jsonl`，仅输入正确支持段落的完整正文。三个数据集的原始问题、答案、ID 和全部样本均保留；此前排除的 2Wiki dev 样本已恢复。

来源为 ModelScope：`OpenDataLab/HotpotQA`、`voidful/2WikiMultihopQA`、`voidful/MuSiQue`。来源版本保存在下载脚本中；下载时仍检查文件大小和校验值，默认不另建下载记录目录。历史数据和下载元数据保存在 `data/backups/`，当前运行不需要这些备份。

完整格式见 [data/FORMAT.md](../FORMAT.md)，数量与顺序核查结果见 `data/processed/preparation_report.json`、`validation_report.json`、`flashrag_order_report.json`。
