"""The 90-min comparison and the paired 150-min, k=2 energy ablation."""

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np

from odte_lyco.config import load_config
from odte_lyco.environment import EnvironmentOverrides, LycoEnvironment
from odte_lyco.external_baselines import (
    BASELINE_STRATEGY, LYDROO_STRATEGY, LyDROO3TierController, QECOController,
)
from odte_lyco.models import STRATEGIES

ROOT = Path(__file__).resolve().parent
SCENARIOS = {
    "S1": "hf_new06_velocity_model_drift_3d_imaging_tuned",
    "S2": "hf_final02_parent_child_frac_hit_casing",
    "S3": "hf_multiwell_multistage_zipper_switching",
}
METHODS = {
    "all_fast": ("base", "all_fast", "All-fast"),
    "all_slow": ("base", "all_slow", "All-slow"),
    "fixed_edge": ("base", "fixed_edge", "Fixed-E"),
    "base": ("base", "coordinated", "Base"),
    "dynamic": ("dynamic", "coordinated", "Base+DW"),
    "odte_lyco": ("odte_lyco", "coordinated", "ODTE-LyCO"),
    "lydroo": ("lydroo", "coordinated", "LyDROO"),
    "qeco": ("qeco", "coordinated", "QECO"),
}
CDF_METHODS = ("odte_lyco", "dynamic", "base", "lydroo", "qeco")
CDF_X = np.linspace(0.0, 1.0, 2001)
ENERGY_METHODS = ("odte_lyco", "no_y")
FOLLOWUP_CLASSES = ("diagnostic", "regional")


@dataclass(frozen=True)
class Job:
    experiment: str
    scene: str
    method: str
    seed: int
    slots: int


def make_jobs(experiment, seeds, slots):
    jobs = []
    for name, methods in (("main", METHODS), ("energy", ENERGY_METHODS)):
        if experiment in ("all", name):
            jobs.extend(Job(name, scene, method, seed, slots)
                        for scene in SCENARIOS for method in methods
                        for seed in range(seeds))
    return jobs


@dataclass(frozen=True)
class Episode:
    start: int
    end: int
    arrival: dict
    workload: dict


def late_episodes(config, scene, seed, slots):
    """The fixed paper protocol; its RNG is separate from the simulator RNGs."""
    rng = np.random.default_rng(np.random.SeedSequence(
        [20261105, 150, tuple(SCENARIOS).index(scene), seed]))
    windows = config.scenarios[SCENARIOS[scene]].burst_windows
    peaks = {name: (
        max([1.0] + [float(w.get("multiplier", 1.0)) for w in windows
                     if w.get("service_class") == name]),
        max([1.0] + [float(w.get("workload_multiplier", 1.0)) for w in windows
                     if w.get("service_class") == name]),
    ) for name in FOLLOWUP_CLASSES}
    scale = slots / 54000
    episodes = []
    for band in ((92.0, 107.0), (112.0, 127.0), (132.0, 143.0)):
        # Round on the original 0.1-s clock before scaling a short smoke run.
        start = round(float(rng.uniform(*band)) * 600)
        duration = round(float(rng.uniform(60.0, 180.0)) / 0.1)
        fraction = float(rng.uniform(0.25, 0.50))
        episodes.append(Episode(
            round(start * scale), round((start + duration) * scale),
            {name: 1 + 2 * fraction * (peaks[name][0] - 1)
             for name in FOLLOWUP_CLASSES},
            {name: 1 + fraction * (peaks[name][1] - 1)
             for name in FOLLOWUP_CLASSES},
        ))
    return episodes


class EnergyEnvironment(LycoEnvironment):
    """Continue the live 90-min state; add only the specified late arrivals."""

    def __init__(self, *args, episodes, **kwargs):
        self.episodes = episodes
        self.late_admitted = 0
        self.late_timely = 0
        super().__init__(*args, **kwargs)

    def episode_at(self, slot):
        return next((e for e in self.episodes if e.start <= slot < e.end), None)

    def _arrival_multiplier(self, service_class, slot):
        episode = self.episode_at(slot)
        return super()._arrival_multiplier(service_class, slot) * (
            episode.arrival.get(service_class, 1.0) if episode else 1.0)

    def _workload_multiplier(self, service_class, slot):
        episode = self.episode_at(slot)
        return super()._workload_multiplier(service_class, slot) * (
            episode.workload.get(service_class, 1.0) if episode else 1.0)

    def _create_dag_request(self, *, sensor_id, service_class, slot,
                            input_bits, cpu_cycles, output_bits):
        if service_class in FOLLOWUP_CLASSES and self.episode_at(slot):
            self.late_admitted += 1
        return super()._create_dag_request(
            sensor_id=sensor_id, service_class=service_class, slot=slot,
            input_bits=input_bits, cpu_cycles=cpu_cycles, output_bits=output_bits)

    def _commit(self, task, slot):
        if task.service_class in FOLLOWUP_CLASSES and self.episode_at(task.arrival_slot):
            self.late_timely += int(slot <= task.deadline_slot)
        return super()._commit(task, slot)


def run_main(job, config):
    policy, mode, _ = METHODS[job.method]
    strategy = {"lydroo": LYDROO_STRATEGY,
                "qeco": BASELINE_STRATEGY}.get(policy)
    if strategy is None:
        strategy = STRATEGIES[policy]
    capture_timeline = job.seed == 0 and job.method in ("all_fast", "all_slow", "base")
    env = LycoEnvironment(
        config, config.scenarios[SCENARIOS[job.scene]], strategy, job.seed,
        EnvironmentOverrides(slots=job.slots, sensors=8), layer_mode=mode,
        collect_timeline=capture_timeline,
    )
    if policy == "lydroo":
        env.controller = LyDROO3TierController(config, sensors=8, seed=job.seed)
    elif policy == "qeco":
        env.controller = QECOController(config, sensors=8, seed=job.seed)
        env.learning_feedback = env.controller
    result = {"metrics": env.run()}
    if job.method in CDF_METHODS:
        values = np.sort([s.odte for s in env.metrics.event_snapshots])
        result["cdf"] = (np.searchsorted(values, CDF_X, side="right") / len(values)).tolist()
    if capture_timeline:
        result["timeline"] = env.timeline_trace()
    if policy == "qeco":
        env.controller.close()
    return result


def run_energy(job, config):
    episodes = late_episodes(config, job.scene, job.seed, job.slots)
    env = EnergyEnvironment(
        config, config.scenarios[SCENARIOS[job.scene]], STRATEGIES[job.method],
        job.seed, EnvironmentOverrides(slots=job.slots, sensors=8), episodes=episodes,
        weight_trace_window_slots=600 if job.method == "odte_lyco" else None,
    )
    total_slots = round(job.slots * 5 / 3)
    # Y(0)=0 is known, not an estimated trajectory point.
    checkpoints = [[0.0, 0.0]]
    checkpoint_slots = {round(job.slots * n / 9) for n in range(1, 16)}
    for slot in range(total_slots):
        env.step(slot)
        completed = slot + 1
        if completed in checkpoint_slots:
            checkpoints.append([completed * config.slot_seconds / 60,
                                sum(q.value for q in env.energy_queues) / env.sensors])
        if completed == job.slots:
            prefix = env.summarize(completed)
    end = env.summarize(total_slots)
    return {
        "metrics": {
            "event_p05_odte": prefix["event_p05_odte"],
            "ontime_p": env.late_timely / env.late_admitted if env.late_admitted else None,
            "energy_150": env.metrics.total_energy,
            "debt_final": end["final_energy_debt_per_sensor"],
        },
        "checkpoints": checkpoints,
        "episodes": [[e.start * config.slot_seconds / 60,
                      e.end * config.slot_seconds / 60] for e in episodes],
        "weight_slots": total_slots,
        **({"weights": env.weight_trace_rows(total_slots)}
           if job.method == "odte_lyco" else {}),
    }


def simulate(job):
    start = time.perf_counter()
    config = load_config(ROOT / "config.yaml")
    result = (run_main if job.experiment == "main" else run_energy)(job, config)
    return {"experiment": job.experiment, "scene": job.scene, "method": job.method,
            "seed": job.seed, "slots": job.slots, "wall_seconds": time.perf_counter() - start,
            **result}
