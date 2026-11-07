"""Physical queues and the nonnegative virtual-queue update."""

from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable
from .models import MigrationTransfer, Task


@dataclass
class VirtualQueue:
    value: float = 0.0
    maximum: float = 0.0

    def update(self, arrival: float, service: float) -> None:
        self.value = max(0.0, self.value + float(arrival) - float(service))
        self.maximum = max(self.maximum, self.value)


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

    def all_queues(self) -> Iterable[Deque[Task]]:
        yield from self.source
        yield from self.local_results
        yield self.edge_compute
        yield self.edge_to_cloud
        yield self.cloud_compute
        yield self.cloud_to_edge
        yield self.pending_at_edge
        yield self.pending_at_cloud
        yield self.dependency_wait

    def total_tasks(self) -> int:
        return sum((len(queue) for queue in self.all_queues()))

    def max_queue_length(self) -> int:
        return max((len(queue) for queue in self.all_queues()), default=0)
