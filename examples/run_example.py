from __future__ import annotations

from pathlib import Path
import os

from src.benchmark_runner import main


def run():
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"

    # Current code uses many relative paths; run from ./src to keep them consistent.
    os.chdir(src_dir)

    cfg_path = repo_root / "examples" / "quickstart_config.yaml"
    main(str(cfg_path))


if __name__ == "__main__":
    run()
