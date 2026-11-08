"""Physical queues and the nonnegative virtual-queue update."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque
from .models import MigrationTransfer, Task


@dataclass
class VirtualQueue:
    value: float = 0.0

    def update(self, arrival: float, service: float) -> None:
        # equation (23): callers supply actual energy E_s and budget B_s.
        # equation (24): ODTE callers supply component log loss and its loss budget.
        # equation (25): telescoping yields mean energy <= B_s + Y_s(T)/T.
        # equation (26): telescoping yields mean loss <= b_ck + Z_ck(N)/N.
        # The long-term bounds follow when the corresponding queue is mean-rate stable.
        self.value = max(0.0, self.value + float(arrival) - float(service))


@dataclass
class TaskQueues:
    sensor_count: int
    source: list[Deque[Task]] = field(init=False)
    local_results: list[Deque[Task]] = field(init=False)
    edge_compute: Deque[Task] = field(default_factory=deque)
    edge_to_cloud: Deque[Task] = field(default_factory=deque)
    cloud_compute: Deque[Task] = field(default_factory=deque)
    cloud_to_edge: Deque[Task] = field(default_factory=deque)
    migration_edge_to_cloud: Deque[MigrationTransfer] = field(default_factory=deque)
    migration_cloud_to_edge: Deque[MigrationTransfer] = field(default_factory=deque)
    pending_at_edge: Deque[Task] = field(default_factory=deque)
    pending_at_cloud: Deque[Task] = field(default_factory=deque)
    dependency_wait: Deque[Task] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self.source = [deque() for _ in range(self.sensor_count)]
        self.local_results = [deque() for _ in range(self.sensor_count)]


