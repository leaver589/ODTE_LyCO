"""Run current manuscript experiments and generate Tables III--V / Figs. 3--6."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ[name] = "1"
os.environ.update(TF_USE_LEGACY_KERAS="1", TF_ENABLE_ONEDNN_OPTS="0",
                  TF_CPP_MIN_LOG_LEVEL="3", CUDA_VISIBLE_DEVICES="-1",
                  PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
sys.dont_write_bytecode = True

from experiments import ROOT, SCENARIOS, METHODS, make_jobs, simulate
from odte_lyco.config import load_config


def simulation_hash():
    # Plot styling and table formatting do not invalidate expensive simulations.
    files = [ROOT / "config.yaml", ROOT / "experiments.py", ROOT / "requirements.txt",
             *sorted((ROOT / "odte_lyco").glob("*.py")),
             *sorted((ROOT / "baselines").glob("*.py"))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_result(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def run_jobs(jobs, cache, workers, plot_only):
    missing = [job for job in jobs if not (cache / job.filename).is_file()]
    if plot_only:
        return [load_result(cache / job.filename) for job in jobs]
    print(f"Runs: {len(jobs)}; cached: {len(jobs) - len(missing)}; to simulate: {len(missing)}", flush=True)
    if missing:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(simulate, job): job for job in missing}
            for index, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                result = future.result()
                with gzip.open(cache / job.filename, "wt", encoding="utf-8", compresslevel=4) as stream:
                    json.dump(result, stream, separators=(",", ":"))
                print(f"[{index}/{len(missing)}] {job.experiment} {job.scene} "
                      f"{job.method} seed={job.seed}: {result['wall_seconds']:.1f}s", flush=True)
    return [load_result(cache / job.filename) for job in jobs]


def main():
    config = load_config(ROOT / "config.yaml")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "main", "energy"), default="all")
    parser.add_argument("--seeds", type=int, default=int(config.simulation["seeds"]))
    parser.add_argument("--slots", type=int, default=int(config.simulation["slots"]),
                        help="Original-scenario slots; energy runs continue for 5/3 of this horizon.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT / "paper_outputs")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate tables/figures from these runs' saved data.")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fingerprint = simulation_hash()
    cache = output / "data" / fingerprint[:12] / f"slots_{args.slots}"
    cache.mkdir(parents=True, exist_ok=True)
    jobs = make_jobs(args.experiment, args.seeds, args.slots)
    with TemporaryDirectory(prefix=".runtime_", dir=output) as workspace:
        os.environ["MPLCONFIGDIR"] = workspace
        results = run_jobs(jobs, cache, args.workers, args.plot_only)
        from tables import write_tables
        from plots import write_figures
        write_tables(results, output)
        write_figures(results, config, output)
    manifest = {
        "simulation_source_sha256": fingerprint,
        "experiment": args.experiment, "scenarios": SCENARIOS,
        "methods": {k: v[2] for k, v in METHODS.items()},
        "seeds": list(range(args.seeds)), "sensors": 8,
        "slot_seconds": config.slot_seconds, "original_slots": args.slots,
        "energy_slots": round(args.slots * 5 / 3), "energy_arrival_pressure_k": 2,
        "weight_evolution": "150-min Full energy runs; 60-s executed-action means",
        "cdf_vertical_range": [0, 0.75],
        "confidence_intervals": "two-sided Student t 95%; seed is the statistical unit; seed-wise macro means",
        "board_table": "archived RK3566 measurements in data/rk3566_published_measurements.json",
        "simulation_cache": cache.relative_to(output).as_posix(),
        "python": platform.python_version(), "platform": platform.system(),
        "packages": {p: version(p) for p in ("numpy", "PyYAML", "matplotlib", "scipy", "torch", "tensorflow", "tf-keras")},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Tables and vector figures: {output}", flush=True)


if __name__ == "__main__":
    main()
