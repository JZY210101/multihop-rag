"""Download unmodified source data from the selected ModelScope mirrors.

Only the six labeled train/dev files used by the current experiment are kept
in raw. Download bookkeeping lives in a temporary directory; provenance files
are optional. This script does not convert records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote

import requests
from modelscope.hub.snapshot_download import dataset_snapshot_download


SOURCES = {
    "2wikimultihopqa": {
        "repository": "voidful/2WikiMultihopQA",
        "revision": "75d023a1becf3e2e73401639536eae3d0bc82d2c",
        "files": ["train.json", "dev.json"],
    },
    "musique": {
        "repository": "voidful/MuSiQue",
        "revision": "aa92b9e2a0bb07120885f620e6f9116ea4f62376",
        "files": [
            "musique_ans_v1.0_train.jsonl",
            "musique_ans_v1.0_dev.jsonl",
        ],
    },
    "hotpotqa": {
        "repository": "OpenDataLab/HotpotQA",
        "revision": "8f4804ba8c47c8d911bdf8d5fb6dabb23535f544",
        "files": ["raw/hotpot_train_v1.1.json", "raw/hotpot_dev_distractor_v1.json"],
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(name: str, root: Path, workers: int, write_manifest: bool = False) -> None:
    source = SOURCES[name]
    destination = root / name
    destination.mkdir(parents=True, exist_ok=True)
    expected = {}
    for parent in sorted({str(Path(item).parent) for item in source["files"]}):
        response = requests.get(
            f"https://www.modelscope.cn/api/v1/datasets/{source['repository']}/repo/tree",
            params={"Revision": source["revision"], "Root": "" if parent == "." else parent},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("Code") != 200:
            raise RuntimeError(f"Could not list {name}: {payload.get('Message')}")
        expected.update({row["Path"]: row for row in payload["Data"]["Files"]})
    for filename in source["files"]:
        if filename not in expected or not expected[filename].get("Sha256"):
            raise RuntimeError(f"Missing upstream checksum for {name}/{filename}")

    print(f"Downloading {source['repository']} at {source['revision']}", flush=True)
    checked = []
    with tempfile.TemporaryDirectory(prefix=f".{name}-download-", dir=root) as staging:
        dataset_snapshot_download(
            dataset_id=source["repository"],
            revision=source["revision"],
            local_dir=staging,
            allow_patterns=source["files"],
            max_workers=workers,
        )
        for filename in source["files"]:
            path = Path(staging) / filename
            row = expected[filename]
            actual = sha256(path)
            if path.stat().st_size != row["Size"] or actual != row["Sha256"]:
                raise RuntimeError(f"Downloaded file failed integrity check: {path}")
            target = destination / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
            checked.append({
                "path": filename, "size": target.stat().st_size, "sha256": actual,
                "url": (
                    f"https://www.modelscope.cn/api/v1/datasets/{source['repository']}/repo"
                    f"?Revision={source['revision']}&FilePath={quote(filename, safe='')}"
                ),
            })
            print(f"Verified {target}: {actual}", flush=True)
    if not write_manifest:
        return
    manifest = {
        "repository": source["repository"],
        "revision": source["revision"],
        "format": "unmodified upstream files",
        "files": checked,
    }
    manifest_dir = root.parent / "source_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--write-manifest", action="store_true",
                        help="Optionally create source_manifests; omitted by default")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in args.datasets:
        download(name, args.output_dir, args.workers, write_manifest=args.write_manifest)


if __name__ == "__main__":
    main()
