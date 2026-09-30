# MultiHop RAG：Gold Evidence 逐跳 ReDeeP 基线

本项目只研究“证据已经正确提供时，生成模型是否产生幻觉”。正式实验不运行 BM25、Wiki18 或其他检索器：每条样本直接使用数据集提供的 supporting evidence，并按 hop 逐步生成中间结果和最终答案。

## 正式流程

```text
数据集样本
  -> 第 1 跳 gold evidence -> 中间结果 1
  -> 第 2 跳 gold evidence + 中间结果 1 -> 中间结果 2
  -> ... -> 最终答案
  -> Qwen 内部状态 -> ReDeeP ECS + PKS
  -> 最终答案 token-F1 标签
```

当前模型是本地的 `Qwen/Qwen3-4B-Instruct-2507`，模型下载使用 ModelScope；不使用原始 ReDeeP 的 LLaMA2-7B，也不实现 AARF。正式入口是 `src.run_oracle`，检测入口是 `src.run_redeep`。

生成时使用 Qwen 模型自带的 chat template，并把格式化后的实际 prompt 保存给 ReDeeP。MuSiQue 若提供每跳子问题，会将当前子问题与该跳 gold evidence 一起输入；HotpotQA 和 2Wiki 没有子问题字段时，仅依据原问题、当前跳证据和已有中间结果逐跳生成。

## 数据文件

```text
data/FlashRAG_Data/hotpotqa/train_gold.jsonl
data/FlashRAG_Data/hotpotqa/dev_gold.jsonl
data/FlashRAG_Data/2wikimultihopqa/{train,dev}.jsonl
data/FlashRAG_Data/musique/{train,dev}.jsonl
```

HotpotQA 的 `*_gold.jsonl` 由 Hugging Face `hotpotqa/hotpot_qa` distractor 配置生成，包含完整 `context` 和 `supporting_facts`。原始 Parquet 分片保存在 `data/original_hotpotqa/hf_distractor/`，转换脚本为 `scripts/prepare_hotpot_from_parquet.py`。

仓库不再需要并且不再包含：

```text
data/wiki_corpus.jsonl
data/index/bm25/
```

## 安装

建议 Python 3.10/3.11，并在 GPU 环境中执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

模型下载：

```bash
mkdir -p model/Qwen3-4B-Instruct-2507
modelscope download --model Qwen/Qwen3-4B-Instruct-2507 --local_dir model/Qwen3-4B-Instruct-2507
```

## 逐跳生成 smoke test

```bash
PYTHONPATH=. python -m src.run_oracle \
  --input data/FlashRAG_Data/musique/dev.jsonl \
  --dataset musique --model model/Qwen3-4B-Instruct-2507 \
  --max-records 5 --output outputs/musique_dev_oracle.jsonl
```

HotpotQA 使用补齐后的 gold 文件：

```bash
PYTHONPATH=. python -m src.run_oracle \
  --input data/FlashRAG_Data/hotpotqa/dev_gold.jsonl \
  --dataset hotpotqa --model model/Qwen3-4B-Instruct-2507 \
  --max-records 5 --output outputs/hotpotqa_dev_oracle.jsonl
```

每条输出包含 `hop_records`、每一跳 prompt/evidence/response/token ids、最终预测、答案 F1 和幻觉标签。

## ReDeeP 校准与评估

先用 train 输出拟合 attention heads、FFN layers 和联合权重，再用同一 calibration 在 dev 上评估：

```bash
PYTHONPATH=. python -m src.run_redeep fit \
  --input outputs/musique_train_oracle.jsonl \
  --output outputs/redeep_musique_train.jsonl \
  --calibration outputs/redeep_musique.json

PYTHONPATH=. python -m src.run_redeep evaluate \
  --input outputs/musique_dev_oracle.jsonl \
  --output outputs/redeep_musique_dev.jsonl \
  --calibration outputs/redeep_musique.json
```

三个数据集分别使用各自的 train calibration。第一次校准可加 `--max-records 500`；完整输出建议加 `--no-token-scores` 以节省磁盘。

数据文件可能按 hop 排序。校准子集不要直接取前 500 条，应在生成 train trace 时增加 `--sampling-strategy random --seed 42`；只做 5 条连通性检查时继续使用默认的 prefix 即可。

## 标签规则

统一使用最终答案 token-F1：

```text
answer_f1 <= 0.30  -> hallucination_label = 1
answer_f1 >  0.30  -> hallucination_label = 0
```

中间 hop 没有稳定的统一 gold answer，因此 hop-level ReDeeP 分数只作为诊断 trace 保存，不把最终答案标签错误地当成中间 hop 标签。

## 代码结构

```text
src/data_adapters.py                    三数据集 gold evidence 统一适配
src/oracle_hop_pipeline.py              无检索器的逐 hop Qwen 生成
src/run_oracle.py                       生成入口
src/redeep/                             Qwen ReDeeP ECS/PKS、校准和检测
src/run_redeep.py                       fit/evaluate 入口
scripts/prepare_hotpot_from_parquet.py  HotpotQA parquet 转 gold JSONL
configs/oracle_iterative.yaml           正式实验配置
```

运行测试：

```bash
PYTHONPATH=. pytest -q
```

原始 `ReDEeP-ICLR/` 目录仅作为论文参考，按项目约定不上传到远程仓库。
