# Qwen ReDeeP 多跳适配

当前正式实验只研究 gold evidence 已正确提供时的生成幻觉。不运行 Wiki18、BM25 或外部检索器。模型使用 ModelScope 下载的 `Qwen/Qwen3-4B-Instruct-2507`，不使用原始 LLaMA2-7B，也不实现 AARF。

## 逐跳流程

```text
Hop 1 gold evidence -> 中间结果 1
Hop 2 gold evidence + 中间结果 1 -> 中间结果 2
...
最后一跳 -> final answer
每一跳保存 prompt、evidence、response、token ids
```

Qwen 使用模型自带的 chat template，保存的 prompt 与实际生成 prompt 完全一致。MuSiQue 还会输入数据集提供的当前子问题，但不会向模型泄露 gold intermediate answer。

`src/oracle_hop_pipeline.py` 完成逐跳生成，`src/run_oracle.py` 是生成入口；`src/redeep/` 内部模块负责 Qwen attention、FFN residual、token-level ECS、PKS、校准和联合 ReDeeP 分数。正式实验不实现或运行 chunk 模式。

## 运行

```bash
PYTHONPATH=. python -m src.run_oracle \
  --input data/FlashRAG_Data/musique/dev.jsonl \
  --dataset musique --model model/Qwen3-4B-Instruct-2507 \
  --max-records 5 --output outputs/musique_dev_oracle.jsonl
```

HotpotQA 使用 `data/FlashRAG_Data/hotpotqa/train_gold.jsonl` 和 `data/FlashRAG_Data/hotpotqa/dev_gold.jsonl`。先用 train 输出执行 `src.run_redeep fit` 校准，再用同一 calibration 对 dev 执行 `evaluate`。三个数据集分别校准。

小规模 train trace 要使用 `--sampling-strategy random --seed 42`，避免数据文件按 hop 排序造成子集只有 2-hop。`src.run_redeep fit` 会再从这些 trace 中按标签分层留出 20% validation；head/layer 数量和联合系数只在 validation 上选择。

## Token ReDeEP 校准

当前仅运行 token-level ReDeEP：

```text
ECS：每个生成 token 取 evidence context 中 attention 最高的 10% token，计算隐藏状态余弦相似度
PKS：FFN 前后 residual 经 LayerNorm + lm_head 后，计算标准数学 JSD
联合分数：normalized PKS - alpha * normalized ECS
```

候选池包含 Qwen 实际存在的所有 attention heads 和 FFN layers。按照论文协议搜索：

```text
K_head  = 1 .. min(32, 实际候选 head 数)
K_layer = 1 .. min(32, 实际 layer 数)
alpha   = 0.1 .. 1.9，步长 0.1
```

这里的 32 是“最终选择多少个特征”的搜索上限，不是把 Qwen 限制为前 32 层。标准 JSD 使用 `0.5 * KL(p||m) + 0.5 * KL(q||m)`；不使用原仓库反向 KL 写法和额外 `10e5` 缩放。

## 标签

最终答案使用共享 token-F1 弱标签：

```text
answer_f1 <= 0.30  -> hallucination_label = 1
answer_f1 >  0.30  -> hallucination_label = 0
```

数据集没有统一的中间 hop gold answer，因此 hop-level ReDeeP 结果只保存为诊断信息；正式检测指标以最终答案标签为准。

## 重要文件

```text
src/redeep/qwen_extractor.py  Qwen attention/hidden state/FFN 提取
src/redeep/scores.py          token ECS 与 PKS
src/redeep/calibration.py     train 校准
src/redeep/detector.py        ReDeeP 检测
src/run_redeep.py             fit/evaluate
```

运行检查：

```bash
PYTHONPATH=. python -m compileall -q src scripts
PYTHONPATH=. pytest -q
```
