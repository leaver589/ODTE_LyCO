"""Deadline-matured ODTE cohorts and timeliness/reliability/availability virtual queues."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import exp, log
import numpy as np
from .config import ServiceClassConfig
from .models import Task
from .queues import VirtualQueue

COMPONENTS = ("timeliness", "reliability", "availability")


@dataclass(frozen=True)
class ODTESnapshot:
    service_class: str
    timeliness: float
    reliability: float
    availability: float
    odte: float
    trust_loss: float
    target: float
    violation: bool
    arrivals: int
    commits: int
    timeliness_target: float
    reliability_target: float
    availability_target: float
    timeliness_loss: float
    reliability_loss: float
    availability_loss: float


@dataclass
class _AvailabilityCounter:
    samples: int = 0
    available: int = 0


@dataclass
class _CohortRecord:
    task_id: int
    service_class: str
    arrival_slot: int
    deadline_slot: int
    deadline_slots: int
    commit_slot: int | None = None

    def freshness(self) -> float:
        if self.commit_slot is None:
            return 0.0
        delay = self.commit_slot - self.arrival_slot + 1
        if delay <= self.deadline_slots:
            return 1.0
        # equation (17): freshness is one through the deadline and decays for late submissions.
        return exp(-(delay - self.deadline_slots) / max(1, self.deadline_slots))

    @property
    def reliable_commit(self) -> bool:
        return self.commit_slot is not None and self.commit_slot <= self.deadline_slot


@dataclass
class ODTEMonitor:
    services: dict[str, ServiceClassConfig]
    epsilon: float = 1e-06
    availability: dict[str, _AvailabilityCounter] = field(init=False)
    cohorts: dict[int, _CohortRecord] = field(init=False, default_factory=dict)
    trust_queues: dict[str, dict[str, VirtualQueue]] = field(init=False)

    def __post_init__(self) -> None:
        self.availability = {name: _AvailabilityCounter() for name in self.services}
        self.trust_queues = {
            name: {component: VirtualQueue() for component in COMPONENTS}
            for name in self.services
        }

    def record_arrival(self, task: Task) -> None:
        cohort_id = task.cohort_id
        self.cohorts[cohort_id] = _CohortRecord(
            task_id=cohort_id,
            service_class=task.service_class,
            arrival_slot=task.arrival_slot,
            deadline_slot=task.deadline_slot,
            deadline_slots=task.deadline_slots,
        )

    def record_availability(self, service_class: str, available: bool) -> None:
        counter = self.availability[service_class]
        counter.samples += 1
        counter.available += int(available)

    def record_commit(self, task: Task, slot: int) -> None:
        record = self.cohorts.get(task.cohort_id)
        if record is not None and record.commit_slot is None:
            record.commit_slot = int(slot)

    def component_target(self, service_class: str, component: str) -> float:
        # equation (22): equal component targets give b_ck = -log(theta_c)/3.
        return self.services[service_class].odte_target ** (1.0 / 3.0)

    def component_debts(self, service_class: str) -> dict[str, float]:
        return {
            component: queue.value
            for component, queue in self.trust_queues[service_class].items()
        }

    def debt(self, service_class: str, component: str | None = None) -> float:
        queues = self.trust_queues[service_class]
        if component is not None:
            return queues[component].value
        return sum((queue.value for queue in queues.values()))


    def finalize_frame(self, current_slot: int) -> list[ODTESnapshot]:
        matured_by_class: dict[str, list[_CohortRecord]] = {
            name: [] for name in self.services
        }
        matured_ids: list[int] = []
        for task_id, record in self.cohorts.items():
            if record.deadline_slot <= current_slot:
                matured_by_class[record.service_class].append(record)
                matured_ids.append(task_id)
        snapshots: list[ODTESnapshot] = []
        for name, service in self.services.items():
            records = matured_by_class[name]
            arrivals = len(records)
            observed_commits = [
                item for item in records if item.commit_slot is not None
            ]
            reliable_commits = sum((item.reliable_commit for item in records))
            if arrivals == 0:
                timeliness = 1.0
                reliability = 1.0
            else:
                # equation (18): mean freshness among submitted requests in this matured cohort.
                timeliness = (
                    sum((item.freshness() for item in observed_commits))
                    / len(observed_commits)
                    if observed_commits
                    else 0.0
                )
                # equation (19): all matured requests remain in the reliability denominator.
                reliability = reliable_commits / arrivals
            availability_counter = self.availability[name]
            availability = (
                availability_counter.available / availability_counter.samples
                if availability_counter.samples
                else 1.0
            )
            values = {
                "timeliness": float(np.clip(timeliness, 0.0, 1.0)),
                "reliability": float(np.clip(reliability, 0.0, 1.0)),
                "availability": float(np.clip(availability, 0.0, 1.0)),
            }
            # equation (21): measured component log losses.
            losses = {
                component: -log(max(self.epsilon, value))
                for component, value in values.items()
            }
            targets = {
                component: self.component_target(name, component)
                for component in COMPONENTS
            }
            for component in COMPONENTS:
                # equation (24): component loss excess is settled once per six-slot window.
                self.trust_queues[name][component].update(
                    arrival=losses[component],
                    service=-log(max(self.epsilon, targets[component])),
                )
            # equation (20): ODTE = timeliness * reliability * availability.
            odte = float(np.clip(np.prod(tuple(values.values())), 0.0, 1.0))
            snapshot = ODTESnapshot(
                service_class=name,
                timeliness=values["timeliness"],
                reliability=values["reliability"],
                availability=values["availability"],
                odte=odte,
                trust_loss=sum(losses.values()),
                target=service.odte_target,
                violation=odte < service.odte_target,
                arrivals=arrivals,
                commits=reliable_commits,
                timeliness_target=targets["timeliness"],
                reliability_target=targets["reliability"],
                availability_target=targets["availability"],
                timeliness_loss=losses["timeliness"],
                reliability_loss=losses["reliability"],
                availability_loss=losses["availability"],
            )
            snapshots.append(snapshot)
            self.availability[name] = _AvailabilityCounter()
        for task_id in matured_ids:
            del self.cohorts[task_id]
        return snapshots
