# 最终答案的共享标签规则

实现：`src/hallucination_labels.py`。ReDeEP 与后续方法共用，不调用外部评判模型。
规则版本：`em_or_f1_basic_bool_v2`。标签 `0` 表示答案通过，`1` 表示答案未通过。
这是最终答案正确性的弱标签，不判定中间推理或整段解释是否忠实于证据。

## 普通答案

预测与全部标准答案/原始别名使用相同规范化。分别取最高 EM 和 token-F1：

```text
EM = 1 或 token-F1 >= 0.30 → 标签 0
否则                       → 标签 1
```

阈值边界包含 `0.30`。空预测为 `1`；缺失标准答案为 `null`，不进入 fit/指标。
token-F1 使用按空白切分后的词频重叠，不是模型 tokenizer 或字符 F1。

基础规范化：

- Unicode NFC、小写化、合并空白；保留 `a/an/the` 与重音字符。
- 统一弯引号、直引号以及等价连字符/负号；去除包裹非空完整答案的成对引号。
- 合法千位逗号归一化：`3,677` → `3677`；不改 `12,34`。
- 仅去除普通词或完整数字后的一个句末句号；保留单字母缩写、常见称谓/公司缩写和多点版本号。
- 保留其余标点、小数点、正负号、百分号、货币符号、斜线和范围；不做单位转换或推测别名。

带外层引号和句号的 `"Paris."`、`"Paris".` 都规范化为 `paris`，重复规范化结果保持一致。
多段分别引用的标题不作为一个整体去引号；格式错误的数字不进行部分千位分隔符转换。

例如：`The The`、`A`、`!!!`、`19.22%`、`−9 °F` 都保留有效信息。
这是自定义标签用 EM/F1，不称作三个数据集的官方评分器。

## yes/no 答案

仅当所有有效标准答案规范化后都属于 `yes/no` 时启用。对原始生成文本做忽略大小写的完整词匹配：

```python
matching_text = unicodedata.normalize("NFC", response).lower()
re.findall(r"(?<!\w)(yes|no)(?!\w)", matching_text)
```

字母、数字、下划线等词字符内部不能匹配；空白和标点可以作为边界。
不使用句式模板、不要求关键词出现在句首，也不采用语义评判。

标准答案从数据集的 `gold_answers`/`golden_answers`/`gold_answer` 等字段读取，
不根据问题句式分类。空答案列表可回退到有效主答案；多个有效别名全部为 yes/no 才启用本规则。

| 匹配情况 | 用于标签的答案 | 判定 |
|---|---|---|
| 只有 `yes`，可重复 | `yes` | 与标准答案精确比较 |
| 只有 `no`，可重复 | `no` | 与标准答案精确比较 |
| `yes`、`no` 同时出现 | 空字符串 | 强制标签 `1`，原因 `boolean_conflict` |
| 两者都没有 | 空字符串 | 标签 `1`，原因 `boolean_missing` |

标准答案为 `yes` 时：

| 原始回答 | 提取结果 | 标签 |
|---|---|---|
| `Yes, it is.` | `yes` | `0` |
| `Yes, nobody objects.` | `yes` | `0` |
| `No, it isn't.` | `no` | `1` |
| `Yes, there is no difference.` | 冲突 | `1` |
| `Yesterday nobody came.` | 未提取 | `1` |

这一规则较保守：包含 `no doubt`、`no difference` 等措辞的肯定回答也会被判为冲突。
它同样可能接受仅出现 `yes` 但解释错误的回答；本协议只使用提取出的最终答案构造标签。

## 输出与旧结果

- `prediction`、`response_token_ids`：保留原始生成回答，用于 ReDeEP 内部状态提取。
- `answer_for_label`、`normalized_answer_for_label`、`normalized_gold_answers`：标签用答案及规范化结果。
- `answer_em`、`answer_f1`：标签用答案与全部标准答案比较后的最高分；冲突/未提取时为 `0`，无标准答案时为 `null`。
- `boolean_answer_status`、`boolean_answer_matches`：提取状态与去重关键词，按首次出现顺序保存。
- `hallucination_label`、`hallucination_label_reason`、`hallucination_label_version`：标签、原因和版本。

原始生成回答不被替换为 `yes/no`，以免与生成 token IDs、ReDeEP 分析内容不一致。
现有 `--label-mode f1_answer` 参数名为兼容旧命令保留，实际执行本页完整规则。

fit/evaluate 会从原始预测重新计算标签，无需只因标签变化就重新生成模型回答。
必须重新 fit，再用新校准文件 evaluate；旧校准文件缺少新规则版本，会被拒绝加载。
旧输出文件本身不会被自动覆盖，更新后的标签保存在新 fit/evaluate 输出中。
