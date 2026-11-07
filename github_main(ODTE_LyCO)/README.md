# ODTE-LyCO: manuscript reproduction

This standalone package recomputes **Tables III–IV and Figures 3–5** of the current manuscript layout, exports the standalone energy-debt panels used in the combined Fig. 3, and regenerates **Table V** from the included RK3566 measurement record. No files from the parent project or author-supplied saved simulation results are needed.

## Install and run

Tested on Windows x86-64 with Python **3.12.14**, using CPU simulation. On Windows, create the local environment and run:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -B run.py
```

All dependencies are pinned. After setup, `run.bat` runs the same entry point and forwards command-line arguments. On Linux/macOS, create `.venv` with `python3.12 -m venv .venv` and use `.venv/bin/python`. The remaining commands below assume the environment is activated.

The full protocol has **600 runs**, with eight sensors, 0.1-s slots, 20 paired seeds (0–19), and four workers using one numerical-computation thread each:

- 480 runs: three scenarios × eight methods × 20 seeds, each lasting 54,000 slots (90 min). Methods: All-fast, All-slow, Fixed-E, Base, Base+DW, ODTE-LyCO, LyDROO, and QECO.
- 120 runs: three scenarios × Full / No-Y × 20 seeds, each lasting 90,000 slots (150 min). The original scenario clock remains at 54,000 slots. The live state continues after 90 min, with three randomly scheduled 1–3-min diagnostic/regional bursts. Only the added arrival pressure is doubled (`k=2`); the workload increments retain their k=1 amplitude.

Full simulation is the default. Completed runs are saved under `paper_outputs/data/`. Re-running the same command reuses completed runs and executes the remainder. A hash of numerical code, configuration, and dependency requirements separates incompatible caches. Figure/table styling changes preserve the simulation cache.

## Paper outputs

| Manuscript | Files under `paper_outputs/` |
|---|---|
| Table III, combined ablations and learning baselines | `tables/table_3.csv`, `tables/table_3.tex` |
| Table IV, 150-min energy-feedback ablation | `tables/table_4.csv`, `tables/table_4.tex` |
| Table V, RK3566 measurements | `tables/table_5.csv`, `tables/table_5.tex` |
| Figure 3, 150-min dynamic weights | `figures/weight_evolution.pdf` |
| Figure 4, migration trajectories | `figures/migration_comparison.pdf` |
| Figure 5, five-method event-window CDFs | `figures/event_odte_cdf_compact.pdf` |
| Figure 6, energy-deficit trajectories | `figures/y_feedback_three_debt.pdf` |
| Combined weights/debt layout figure | `figures/weight_y_feedback_combined.pdf` |

Only these five figure files and three tables are exported, plus reusable run data and `run_manifest.json`. PDFs are compact vector figures. CDF ordinates span **0–0.75**, with x spanning 0–100% ODTE. The `.tex` tables use `booktabs` and can replace the corresponding manuscript floats. Figures 1–2 are explanatory artwork; Tables I–II describe the parameters/scenarios already given in `config.yaml`.

## Statistical definitions

Every metric, including percentiles, is calculated per seed before averaging. The reported ODTE, P05, E05, and Ratio entries in Table III use two-sided **95% Student-t intervals**, shown as mean ± half-width in one cell. Macro intervals first average the three scenarios within each seed. Intervals are not truncated or rounded before aggregation.

CDFs are calculated within each run's original event interval, then averaged equally across seeds; snapshots without mature requests remain included. The weight evolution uses 60-s means from the 150-min Full runs, followed by equal seed means. Migration plots use seed 0, exact start/switch times, 30-s pressure maxima, and a nonuniform time scale. Energy curves use 10-min checkpoints, pointwise 95% seed-level intervals, and later-event occupancy bands.

The No-Y policy removes accumulated energy feedback from source scoring, result transmission, recovery weighting, and placement. It retains Q/ODTE feedback, instantaneous resource costs, physical energy consumption, and the real per-sensor energy ledger. Later-burst on-time rates count requests arriving in those bursts; they are not ODTE snapshot statistics.

Both simulation learning baselines learn online from seeded initialization in the same physical environment. Their network implementations and upstream MIT licenses remain in `baselines/`.

## RK3566 table

`data/rk3566_published_measurements.json` contains the archived results and provenance. Throughput, latency, and parameter counts come from the original board run summaries. Power values are the author-reported total-board readings in manuscript Table V; they are not present in the timing CSVs. This package regenerates the published table from those records, rather than measuring board performance on the reviewer's computer.

The board protocol used S1, seed 0, 54,000 slots, a 200-slot warm-up, one numerical thread, CPU control for Base / Base+DW / ODTE-LyCO, and frozen FP16 NPU inference plus CPU logic for LyDROO / QECO. New board measurements require the corresponding RK3566/NPU setup and power equipment.

## Short check and replotting

```sh
python -B run.py --seeds 1 --slots 540 --workers 2 --output paper_outputs/smoke
python -B run.py --seeds 1 --slots 540 --output paper_outputs/smoke --plot-only
python -B -m unittest discover -s tests
```

Short runs scale scenario and late-burst times together and show the actual shorter time horizon. They check execution, not manuscript performance. A single seed has no estimable confidence interval, so its intervals/bands are omitted. Use the unmodified default command for the paper protocol.

For one experimental block, add `--experiment main` or `--experiment energy`. Once the default run is complete, `python -B run.py --plot-only` regenerates all paper outputs without simulation. All commands and paths work from a standalone copy of this directory.
