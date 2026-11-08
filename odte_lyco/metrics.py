"""Only the request, ODTE, queue and sensor-energy metrics used in the paper."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RunMetrics:
    sensors: int
    slots: int
    energy_budget_per_slot: float
    evaluation_event_start_slot: int
    evaluation_event_end_slot: int
    admitted: int = 0
    completed: int = 0
    timely_completed: int = 0
    total_energy: float = 0.0
    queue_sum: float = 0.0
    queue_samples: int = 0
    migrations: int = 0
    delays: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)
    event_snapshots: list = field(default_factory=list)


    def record_admission(self):
        self.admitted += 1

    def record_commit(self, task):
        delay = task.delay()
        self.completed += 1
        self.timely_completed += int(delay <= task.deadline_slots)
        self.delays.append(delay)

    def record_energy(self, value):
        self.total_energy += float(value)

    def record_queue(self, total):
        self.queue_sum += float(total)
        self.queue_samples += 1

    def record_migration(self):
        self.migrations += 1

    def add_snapshots(self, snapshots, *, slot):
        self.snapshots.extend(snapshots)
        if self.evaluation_event_start_slot <= slot < self.evaluation_event_end_slot:
            self.event_snapshots.extend(snapshots)

    def finalize(self, energy_debt):
        odte = np.asarray([item.odte for item in self.snapshots])
        event = np.asarray([item.odte for item in self.event_snapshots])
        minimum = np.asarray(
            [
                min(item.timeliness, item.reliability, item.availability)
                for item in self.snapshots
            ]
        )
        delays = np.asarray(self.delays)
        return {
            "admitted_tasks": float(self.admitted),
            "completed_tasks": float(self.completed),
            "mean_odte": float(odte.mean()),
            "p05_odte": float(np.percentile(odte, 5)),
            "event_p05_odte": float(np.percentile(event, 5)),
            "p05_min_component": float(np.percentile(minimum, 5)),
            "timely_completion_rate": self.timely_completed / self.admitted,
            "total_sensor_energy": self.total_energy,
            # equation (63): effective budget is averaged over heterogeneous sensors.
            "energy_budget_ratio": self.total_energy
            / (self.sensors * self.slots)
            / self.energy_budget_per_slot,
            "final_energy_debt_per_sensor": energy_debt,
            "mean_delay_slots": float(delays.mean()),
            "p95_delay_slots": float(np.percentile(delays, 95)),
            "mean_total_queue": self.queue_sum / self.queue_samples,
            "migrations": float(self.migrations),
        }
