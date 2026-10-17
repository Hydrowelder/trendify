"""Times TrendifyPipeline.run (generate + render) over scripts/example_data/models with 1..N worker processes."""

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

from trendify.examples import example_record_generator
from trendify.pipeline import TrendifyPipeline

MODELS_DIR = Path(__file__).resolve().parent / "example_data" / "models"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-procs", type=int, default=10)
    args = parser.parse_args()

    run_dirs = [p for p in MODELS_DIR.glob("*") if p.is_dir()]
    if not run_dirs:
        sys.exit(
            f"No run directories in {MODELS_DIR}. Create them with: "
            f"trendify example-data -w {MODELS_DIR.parent} -n 500 --n-cols 300"
        )

    print(f"TrendifyPipeline.run, {len(run_dirs)} runs from {MODELS_DIR}")
    print(f"{'procs':>5} {'seconds':>8}")
    with tempfile.TemporaryDirectory() as tmp:
        for n in range(1, args.max_procs + 1):
            pipeline = TrendifyPipeline(output_dir=Path(tmp) / f"n{n}", n_procs=n)
            start = time.perf_counter()
            pipeline.run(example_record_generator, run_dirs)
            print(f"{n:>5} {time.perf_counter() - start:>8.2f}", flush=True)
            shutil.rmtree(pipeline.output_dir)


if __name__ == "__main__":
    main()
