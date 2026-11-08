# ODTE-LyCO: reproducible hydraulic-fracturing digital twin experiments

This package implements the sensor-edge-cloud model and ODTE-LyCO controller in Sections II and III of *Coordinated Task Scheduling for Entanglement Quality and Energy Consumption in Hydraulic Fracturing Microseismic Digital Twins*. Its default command computes Figures 3-5 and complete Tables III-IV from seeded simulations.

## Environment and installation

The reference environment uses **Python 3.12.14**, Windows x86-64, and CPU execution. `requirements.txt` pins NumPy 2.5.1, SciPy 1.18.1, Matplotlib 3.11.1, PyYAML 6.0.3, PyTorch 2.9.1, TensorFlow 2.21.0, and tf-keras 2.21.0. NumPy and SciPy support numerical evaluation and confidence intervals; Matplotlib renders vector figures; PyYAML reads the experiment settings; PyTorch runs LyDROO; TensorFlow and tf-keras run QECO through the TensorFlow v1 compatibility interface.

Open a terminal in this directory and create an isolated environment:

```bat
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -B run.py
```

`run.bat` uses the same local environment and forwards command arguments:

```bat
run.bat --workers 4
```

For a Python 3.12 environment on Linux, the corresponding commands are:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B run.py --workers 4
```

For subsequent `python` commands, use the environment's Python executable or activate the environment with `.venv\Scripts\activate.bat` in Windows Command Prompt, `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in Linux.

The runner sets one numerical thread per worker, selects CPU execution, and enables the tf-keras compatibility interface before loading the learning libraries. Four workers are the default; `--workers` controls concurrent simulations and memory use. Installation retrieves the pinned packages; experiment execution uses the source files included in this directory.

## Paper protocol

The default run executes **600 simulations** with eight logical sensors, 0.1-second slots, and seeds **0-19**:

- **Table III and Figures 3 and 5:** three scenarios, eight methods, and 20 seeds, for 480 runs of 54,000 slots (90 minutes). The methods are All-fast, All-slow, Fixed-E, Base, Base+DW, ODTE-LyCO, LyDROO, and QECO.
- **Table IV and Figure 4:** three scenarios, two energy-feedback policies, and 20 paired seeds, for 120 runs of 90,000 slots (150 minutes). Each run continues its original 90-minute state through three later diagnostic/regional bursts.

Base combines task-queue, energy-deficit, and ODTE feedback with fixed weights and additive scoring. Base+DW activates dynamic weights, and ODTE-LyCO adds smooth aggregation. All-fast uses Base scoring with per-slot placement; All-slow uses Base scoring with action templates held over 60-slot windows; Fixed-E initializes and maintains all twins at the edge. The coordinated methods use per-slot task actions and 60-slot placement evaluations.

The energy-feedback ablation uses the physical energy ledger and instantaneous resource costs while setting energy-deficit contributions in source scoring, result-upload scoring, recovery gating, and placement to zero. The full method uses the live energy-deficit values in these four control paths.

Task arrivals, sizes, and network observations use separate seeded random streams. Transmission outcomes use keys containing the seed, slot, channel, and task or service identifier. Methods sharing a scenario and seed therefore receive paired external conditions.

The later-burst schedule uses a separate seeded generator. Burst start times lie within 92-107, 112-127, and 132-143 minutes, and each burst lasts 1-3 minutes. A shared amplitude fraction is drawn uniformly from 0.25-0.50 for each burst. Arrival multipliers are `1 + 2 * fraction * (scenario_peak - 1)`, while workload multipliers are `1 + fraction * (scenario_peak - 1)`. This implements the paper's `k=2` arrival-pressure experiment. The scenario clock remains anchored to the first 54,000 slots throughout the extension.

## Generated figures and tables

A full run writes seven files under `paper_outputs/`:

- `figures/migration_comparison.pdf` is **Figure 3**, showing seed-0 migration starts, location switches, migration intervals, 30-second pressure maxima, and service-path outages for S1-S3.
- `figures/weight_y_feedback_combined.pdf` is **Figure 4**, combining the 150-minute weight evolution and energy-deficit trajectories.
- `figures/event_odte_cdf_compact.pdf` is **Figure 5**, showing the event-window ODTE distributions of five methods, with a cumulative-probability axis from 0 to 0.75.
- `tables/table_3.csv` and `tables/table_3.tex` contain **complete Table III**: all twelve metrics for eight methods in S1-S3 and the equal-scenario macro average, giving 32 result rows.
- `tables/table_4.csv` and `tables/table_4.tex` contain **Table IV**: event-period E05, later-burst on-time rate, total 150-minute energy, and final mean energy debt, with six scenario-policy rows.

CSV files use UTF-8 encoding with a byte-order mark for spreadsheet compatibility. LaTeX tables use `booktabs`; the manuscript already loads that package. Each invocation recomputes its selected simulations and replaces its corresponding figures and tables. Plotting caches reside in a temporary runtime directory that is cleared when execution finishes. Simulation records stay in memory during the run.

## Core source files

- **`run.py`** is the execution entry point. It configures numerical threads, schedules worker processes, reports progress, and calls the figure and table exporters.
- **`config.yaml`** is the complete parameter and scenario specification. Its `simulation`, `resources`, `energy`, `controller`, and `placement` blocks define the clocks, capacities, budgets, control weights, and migration rules. Its `service_classes` block defines the status, diagnostic, and regional DAGs.
- **`experiments.py`** is the experiment coordinator. `make_jobs` constructs the paper's scenario-method-seed combinations. `run_main` evaluates the 90-minute comparison. `EnergyEnvironment`, `late_episodes`, and `run_energy` implement the continuous 150-minute energy-feedback experiment.
- **`odte_lyco/config.py`** is the typed YAML loader, including service parameters, DAG templates, and scenario definitions.
- **`odte_lyco/models.py`** defines task-node states, network observations, execution routes, controller actions, migration records, and the paper's ablation settings.
- **`odte_lyco/environment.py`** is the shared discrete-event simulator. It admits requests, releases dependency-ready DAG nodes, serves CPUs and links, accounts for actual sensor energy, advances migration transfers, and submits final results to the current authoritative twin.
- **`odte_lyco/controller.py`** is the fast-layer controller. It builds feasible actions, estimates latency and quality, computes dynamic weights and capacity gates, evaluates additive or smooth action scores, ranks source tasks, and selects result-upload powers.
- **`odte_lyco/placement.py`** is the placement controller. It accumulates network observations, estimates class-specific placement quality and backlog, evaluates migration-adjusted utility, and enforces migration timing and concurrency rules.
- **`odte_lyco/queues.py`** defines physical task queues, dedicated migration queues, and the shared virtual-queue update.
- **`odte_lyco/odte.py`** evaluates deadline-matured request cohorts, computes timeliness/reliability/availability and their ODTE product, and updates the three component-deficit queues.
- **`odte_lyco/metrics.py`** computes the paper's per-run request, ODTE, latency, backlog, energy, and migration statistics.
- **`odte_lyco/external_baselines.py`** connects the copied LyDROO and QECO learners to the common three-tier simulator, action constraints, energy accounting, and final-submission criteria.
- **`tables.py`** formats seed-level statistics and complete manuscript tables. **`plots.py`** renders the three experimental figures.
- **`tests/test_reproduction.py`** verifies controller calculations, cohort settlement, route locking, dependency release, energy accounting, migration observations, source priorities, statistical units, and complete Table III export.

## Scenario simulation

Scenario simulations generate synthetic request DAGs and time-varying resource/link states for evaluating orchestration decisions. The `scenarios` block of `config.yaml` contains the entire scenario definitions. `SCENARIOS` in `experiments.py` maps their descriptive names to the paper labels:

- **S1**, `hf_new06_velocity_model_drift_3d_imaging_tuned`, represents 3-D velocity drift and joint imaging. Staged arrivals and task sizes increase during 20-75% of the run, while resource windows alter edge/cloud capacity during 25-75%. Its event evaluation interval is 15-75%.
- **S2**, `hf_final02_parent_child_frac_hit_casing`, represents parent-child frac-hit and casing-integrity diagnosis. Pumping bursts occur during 20-68%; backhaul and edge capacity change during 20-34%; edge computing availability reaches 0.22 during 39-55%, accompanied by cloud expansion. Its event evaluation interval is 18-68%.
- **S3**, `hf_multiwell_multistage_zipper_switching`, represents four-well multistage zipper fracturing. Four consecutive phases span 12-26%, 26-40%, 40-54%, and 54-68%. Sensor groups provide heterogeneous wireless loss and energy budgets. The regional twin starts in the cloud, and the other twins start at the edge. Its event evaluation interval is 12-68%.

`burst_windows` define arrival and workload multipliers; `resource_windows` define time-varying capacity, link quality, and node availability; outage-window fields define forced interruptions; `sensor_groups` define sensor-specific conditions. Window boundaries use the original scenario length. `LycoEnvironment._network_state`, `_arrival_multiplier`, `_workload_multiplier`, and `_generate_arrivals` convert these settings into the live simulation.

Status requests contain one node, diagnostic requests use a three-node chain, and regional requests use a four-node fork-join DAG. The root's first service locks the execution location for the request. Successors enter service in the slot after their predecessors complete. Completion is recorded upon successful final-result submission to the current authoritative twin; requests continue receiving service after their deadlines.

## Control timing

ODTE is settled every six slots. Coordinated placement uses a 60-slot observation window and schedules migration for the next slot. All-fast evaluates placement between migrations, using observations accumulated since the preceding evaluation. Observation collection continues throughout state transfer and the 20-slot pause, and each executed evaluation refreshes the observation accumulator. Same-class migration starts are separated by 240 slots in coordinated/All-slow modes and 24 slots in All-fast mode. State transfer uses dedicated queues with priority over task traffic on the shared backhaul, followed by a 20-slot pause. Results wait for service and are forwarded to the current authoritative location when necessary.

Candidate evaluation considers classes between state transfers and applies the utility, quality, and minimum-start-interval conditions. Equal-score source tasks favor earlier deadlines.

## Copied learning baselines and sources

Both learners are included in `baselines/`, together with their upstream MIT license texts. The included source supports local execution from this package.

**LyDROO:** [upstream GitHub repository](https://github.com/revenol/LyDROO), specifically [memory.py](https://github.com/revenol/LyDROO/blob/main/memory.py). `baselines/lydroo.py` contains the adapted PyTorch memory DNN, online imitation updates, and OPN candidate generation. `baselines/LICENSE-LyDROO` preserves the upstream license and copyright. The adapter uses a 24-256-128-8 network for eight sensors, a replay capacity of 1,024, batch size 128, and training interval 20. Binary local/offload proposals are evaluated through the common discrete CPU/power choices and the edge/cloud routes.

**QECO:** [upstream GitHub repository](https://github.com/ImanRHT/QECO), specifically [D3QN.py](https://github.com/ImanRHT/QECO/blob/main/D3QN.py). `baselines/qeco.py` contains the adapted dueling double-DQN/LSTM learner. `baselines/LICENSE-QECO` preserves the upstream license and copyright. Each sensor has an agent with six observation features, a two-feature load history over ten steps, 20 LSTM units, two 20-unit dense layers, and three route actions. Replay capacity is 500, batch size is 32, target replacement occurs every 200 learning updates, and training occurs every ten slots after the first 200 slots. Replay entries retain each decision's actual current/next histories, and target updates synchronize the full network, including the LSTM. Exploration samples the Edge and Cloud actions, following the upstream convention; the learned greedy step evaluates all three route actions.

Both baselines learn online from seeded initialization and retain their scenario's initial twin placement. Their three-tier adapters share the physical task/DAG model, link outcomes, execution constraints, and final-result submission rule with ODTE-LyCO. QECO rewards reflect timely completion, delay, and sensor energy; LyDROO selects and learns from candidate actions. The source preserves the upstream learning operations used in this experiment while keeping the experiment-specific adaptation in `odte_lyco/external_baselines.py`.

## License

The ODTE-LyCO source is released under the MIT License in [`LICENSE`](LICENSE). The copied LyDROO and QECO source files retain their upstream MIT terms and copyright statements in `baselines/LICENSE-LyDROO` and `baselines/LICENSE-QECO`.

## Statistics

Each seed is the statistical unit. ODTE mean, P05, event-window E05, minimum-component W05, latency statistics, on-time rate, time-averaged queue workload, energy, terminal energy debt, and migration count are first calculated within each run. Means then weight seeds equally. Macro values average scenarios within each seed before estimating their means and uncertainty.

Table III reports two-sided **95% Student-t confidence intervals** as mean plus/minus half-width for ODTE, P05, E05, and energy Ratio. Intervals retain their full mathematical range. Quality statistics are percentages; latency uses slots; backlog uses normalized service slots; energy uses joules; terminal energy debt uses joules per sensor. Complete Table III contains 384 metric cells across its 32 result rows.

The ODTE cohort at each window contains requests whose deadlines have matured and whose quality is being settled for the first time. Timeliness averages freshness over submitted requests; reliability divides on-time submissions by the entire matured cohort. Empty cohorts use timeliness and reliability equal to one. Availability measures the current authoritative service chain, including migration interruptions. Every class-window snapshot participates in whole-run statistics, and event-window statistics select snapshots ending within the scenario evaluation interval.

Figure 3 uses exact seed-0 migration times in seconds. Its display scale expands closely spaced migration events and compresses long quiet periods; each tick retains its actual simulation time. Hatching marks interruptions of the gateway-backhaul-cloud service path.

Figure 4 uses per-run 60-second means of executed-source-action weights followed by equal seed means. Each weight is divided by its fixed base value. Energy-debt curves show ten-minute checkpoints, equal seed means, and pointwise 95% Student-t intervals. The lower occupancy band averages the later-burst schedules. Table IV's later-burst on-time rate uses diagnostic/regional requests arriving during those added bursts. Figure 5 averages within-run empirical event-window CDFs equally over seeds.

## Verification and selected experiments

Run the regression checks in the activated environment:

```sh
python -B -m unittest discover -s tests
```

The completed reference run executed all 600 simulations with 12 workers on a Windows host with 16 logical processors and 32 GB RAM. The ODTE-LyCO macro results are **97.35 +/- 0.05% mean ODTE**, **85.81 +/- 0.63% P05**, **69.49 +/- 1.75% E05**, **1202.640 J** total sensor energy, and **0.960 +/- 0.0017** energy Ratio. All-fast has macro P05 **8.62%** and mean migration count **80.83**. The validation covers 21 regression checks, 128 independently reconstructed Table III confidence intervals, and 60 matching 90-minute prefixes between the main and extended experiments. The exported figures and tables provide the experimental reference for the reviewed implementation.

A compact end-to-end execution uses a 5,400-slot original clock and one seed:

```sh
python -B run.py --seeds 1 --slots 5400 --workers 4 --output verification_outputs
```

This command evaluates the entire pipeline at a reduced horizon. Its table cells contain per-run measurements, and its single-seed output displays means. The default command supplies the paper's 54,000-slot horizon, 20 paired seeds, and confidence intervals.

The compact command also passed in a separately copied local Windows environment with all seven pinned package versions. This environment completed 30 simulations and 21 regression checks using local installed packages, and produced the complete seven-file output set.

Select an experimental block through:

```sh
python -B run.py --experiment main --workers 4
python -B run.py --experiment energy --workers 4
```

The first command computes Table III and Figures 3 and 5. The second computes Table IV and Figure 4. Running these two commands with the default horizon and seed count completes the full paper protocol.
