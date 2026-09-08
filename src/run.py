"""使用 FlashRAG 原生 Config/Dataset/Retriever/Generator 运行固定 hop Pipeline。"""
import argparse
from flashrag.config import Config
from flashrag.utils import get_dataset
from .fixed_hop_pipeline import FixedHopPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flashrag_fixed_hop.yaml")
    parser.add_argument("--dataset_name")
    parser.add_argument("--split")
    args = parser.parse_args(); overrides = {}
    if args.dataset_name: overrides["dataset_name"] = args.dataset_name
    if args.split: overrides["split"] = [args.split]
    config = Config(config_file_path=args.config, config_dict=overrides)
    all_split = get_dataset(config)
    for split in config["split"]:
        if all_split.get(split) is None: continue
        print(f"Running fixed-hop FlashRAG pipeline on {split} ({len(all_split[split])} samples)")
        output = FixedHopPipeline(config).run(all_split[split], do_eval=True)
        output.save(f"{config['save_dir']}/{config['dataset_name']}_{split}_fixed_hop.json")

if __name__ == "__main__": main()
