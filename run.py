"""Recompute manuscript Figures 3--5 and complete Tables III--IV."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "1"
os.environ.update(TF_USE_LEGACY_KERAS="1", TF_ENABLE_ONEDNN_OPTS="0",
                  TF_CPP_MIN_LOG_LEVEL="3", CUDA_VISIBLE_DEVICES="-1",
                  PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
sys.dont_write_bytecode = True

from experiments import ROOT, make_jobs, simulate
from odte_lyco.config import load_config


def run_jobs(jobs, workers):
    results = []
    print(f"Simulations: {len(jobs)}; workers: {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(simulate, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            result = future.result()
            results.append(result)
            print(f"[{index}/{len(jobs)}] {job.experiment} {job.scene} "
                  f"{job.method} seed={job.seed}: {result['wall_seconds']:.1f}s", flush=True)
    return sorted(results, key=lambda r: (r["experiment"], r["scene"], r["method"], r["seed"]))


def main():
    config = load_config(ROOT / "config.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "main", "energy"), default="all")
    parser.add_argument("--seeds", type=int, default=int(config.simulation["seeds"]))
    parser.add_argument("--slots", type=int, default=int(config.simulation["slots"]),
                        help="Original-scenario slots; energy runs continue for 5/3 of this horizon.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "paper_outputs")
    args = parser.parse_args()
    if args.seeds < 1 or args.workers < 1:
        parser.error("Seeds and workers must be positive integers.")
    if args.slots < config.slow_layer_period_slots or args.slots % config.slow_layer_period_slots:
        parser.error("Slots must be a positive multiple of 60, with at least 60 slots.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".runtime_", dir=output) as workspace:
        os.environ["MPLCONFIGDIR"] = workspace
        results = run_jobs(make_jobs(args.experiment, args.seeds, args.slots), args.workers)
        from tables import write_tables
        from plots import write_figures
        write_tables(results, output)
        write_figures(results, config, output)
    print(f"Paper figures and complete tables: {output}", flush=True)


if __name__ == "__main__":
    main()
