# MultiHop RAG（FlashRAG 基线）

直接使用 FlashRAG 原生的 `Config`、`get_dataset`、`get_retriever`、`get_generator` 和 `BasicPipeline`，只在 Pipeline 层增加固定 hop 控制与 trace 记录。FlashRAG 官方已预处理 HotpotQA、2WikiMultiHopQA、MuSiQue；MultiHop-RAG 若不在当前 FlashRAG 数据包中，需要按同样的 JSONL 格式放入 `data_dir/multihop-rag/`。

## 安装

```bash
pip install -r requirements.txt
```

注意：这里使用 GitHub 官方 FlashRAG 仓库，不使用 PyPI 上同名的非官方/不完整包。
```

## 运行

```bash
PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml
```

FlashRAG 官方数据集需要先下载到本地数据目录。FlashRAG 的标准结构是：

```text
data/FlashRAG_Data/hotpotqa/{train,dev,test}.jsonl
data/FlashRAG_Data/musique/{train,dev,test}.jsonl
data/FlashRAG_Data/2wikimultihopqa/{train,dev,test}.jsonl
data/FlashRAG_Data/multihop-rag/{train,dev,test}.jsonl
```

切换数据集：`PYTHONPATH=. python -m src.run --config configs/flashrag_fixed_hop.yaml --dataset_name musique`

支持 `hotpotqa`、`musique`、`2wikimultihopqa`、`multihop-rag`。每行输出包含问题、答案、hop 数、每跳 query、检索文档及分数，供后续幻觉检测使用。
