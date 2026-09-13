"""使用 FlashRAG 原生 Config/Dataset/Retriever/Generator 运行固定 hop Pipeline。"""
import argparse
from pathlib import Path


def _validate_runtime_paths(config):
    corpus_path = Path(config["corpus_path"])
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}. Prepare the FlashRAG wiki corpus before running retrieval."
        )
    index_path = Path(config["index_path"])
    if not index_path.is_dir():
        raise FileNotFoundError(
            f"BM25 index not found: {index_path}. Build it with flashrag.retriever.index_builder first."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flashrag_fixed_hop.yaml")
    parser.add_argument("--dataset_name")
    parser.add_argument("--split")
    args = parser.parse_args()
    overrides = {}
    if args.dataset_name:
        overrides["dataset_name"] = args.dataset_name
    if args.split:
        overrides["split"] = [args.split]
    try:
        from flashrag.config import Config
        from flashrag.utils import get_dataset
    except ImportError as exc:
        raise ImportError("Running the multi-hop pipeline requires FlashRAG; install requirements.txt first") from exc

    from .fixed_hop_pipeline import FixedHopPipeline

    config = Config(config_file_path=args.config, config_dict=overrides)
    _validate_runtime_paths(config)
    all_split = get_dataset(config)
    for split in config["split"]:
        if all_split.get(split) is None:
            raise FileNotFoundError(f"Dataset split not found: {config['dataset_path']}/{split}.jsonl")
        print(f"Running fixed-hop FlashRAG pipeline on {split} ({len(all_split[split])} samples)")
        output = FixedHopPipeline(config).run(all_split[split], do_eval=True)
        output.save(f"{config['save_dir']}/{config['dataset_name']}_{split}_fixed_hop.json")


if __name__ == "__main__":
    main()
