"""Task DAG, physical state, actions and paper ablations."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Route(str, Enum):
    LOCAL = "local"
    EDGE = "edge"
    CLOUD = "cloud"


class Placement(str, Enum):
    EDGE = "edge"
    CLOUD = "cloud"


class PayloadKind(str, Enum):
    RAW_FOR_CLOUD = "raw_for_cloud"
    RESULT_FOR_TWIN = "result_for_twin"


@dataclass
class Task:
    task_id: int
    sensor_id: int
    service_class: str
    arrival_slot: int
    deadline_slots: int
    input_bits: float
    cpu_cycles: float
    output_bits: float
    remaining_bits: float
    remaining_cycles: float
    request_id: int | None = None
    dag_node_id: str = "task"
    predecessor_ids: tuple[str, ...] = ()
    successor_ids: tuple[str, ...] = ()
    dependency_input_bits: float = 0.0
    dependency_output_bits: float = 0.0
    is_dag_sink: bool = True
    ready_slot: int | None = None
    route: Route | None = None
    payload_kind: PayloadKind | None = None
    compute_location: Placement | None = None
    first_service_slot: int | None = None
    completed_slot: int | None = None
    failed_attempts: int = 0

    def age(self, slot: int) -> int:
        return max(0, slot - self.arrival_slot + 1)

    @property
    def cohort_id(self) -> int:
        return self.task_id if self.request_id is None else self.request_id

    @property
    def is_dag_root(self) -> bool:
        return not self.predecessor_ids

    @property
    def deadline_slot(self) -> int:
        return self.arrival_slot + self.deadline_slots - 1

    def mark_started(self, slot: int, route: Route) -> None:
        if self.first_service_slot is None:
            self.first_service_slot = slot
            self.route = route

    def delay(self) -> int:
        return self.completed_slot - self.arrival_slot + 1


@dataclass(frozen=True)
class NetworkState:
    slot: int
    se_success_probability: float
    ec_success_probability: float
    se_available: bool
    ec_available: bool
    edge_gateway_available: bool
    edge_compute_available: bool
    cloud_available: bool
    edge_capacity_multiplier: float
    cloud_capacity_multiplier: float
    ec_capacity_multiplier: float = 1.0
    sensor_se_success_probability: float | None = None
    sensor_se_available: bool | None = None

    @property
    def edge_available(self) -> bool:
        return self.edge_compute_available


@dataclass(frozen=True)
class SourceAction:
    task_id: int
    route: Route
    cpu_frequency_hz: float = 0.0
    transmit_power_w: float = 0.0
    predicted_delay_slots: float = 0.0
    predicted_timeliness: float = 1.0
    predicted_reliability: float = 1.0
    predicted_availability: float = 1.0
    predicted_trust_loss: float = 0.0
    criticality: float = 0.0
    odte_weight: float = 0.0
    energy_weight: float = 0.0
    objective: float = 0.0


@dataclass(frozen=True)
class MigrationEvent:
    frame_index: int
    slot: int
    service_class: str
    source: Placement
    target: Placement
    predicted_odte_gain: float
    predicted_availability_gain: float
    predicted_queue_reduction: float
    sensor_energy_debt_per_sensor: float
    joint_utility_gain: float
    required_utility_gain: float
    reverses_recent_migration: bool = False


@dataclass
class MigrationTransfer:
    event: MigrationEvent
    state_bits: float
    remaining_bits: float
    started_slot: int
    attempted_bits: float = 0.0
    delivered_bits: float = 0.0
    failed_attempts: int = 0
    completed_slot: int | None = None


@dataclass(frozen=True)
class StrategySpec:
    name: str
    use_lyapunov: bool
    use_odte: bool
    use_dynamic_weights: bool = False
    cost_aggregation: str = "additive"
    use_energy_feedback: bool = True


@dataclass(frozen=True)
class LayerControlSpec:
    name: str
    use_fast_feedback: bool
    use_slow_placement: bool
    share_fast_state: bool
    task_decision_cadence: str
    placement_policy: str


LAYER_CONTROL_MODES: dict[str, LayerControlSpec] = {
    "all_fast": LayerControlSpec(
        "all_fast",
        use_fast_feedback=True,
        use_slow_placement=False,
        share_fast_state=False,
        task_decision_cadence="slot",
        placement_policy="fast_adaptive",
    ),
    "all_slow": LayerControlSpec(
        "all_slow",
        use_fast_feedback=False,
        use_slow_placement=True,
        share_fast_state=False,
        task_decision_cadence="window",
        placement_policy="slow_adaptive",
    ),
    "fixed_edge": LayerControlSpec(
        "fixed_edge",
        use_fast_feedback=True,
        use_slow_placement=False,
        share_fast_state=False,
        task_decision_cadence="slot",
        placement_policy="fixed_edge",
    ),
    "coordinated": LayerControlSpec(
        "coordinated",
        use_fast_feedback=True,
        use_slow_placement=True,
        share_fast_state=True,
        task_decision_cadence="slot",
        placement_policy="slow_coordinated",
    ),
}
LAYER_ABLATION_MODE_NAMES = tuple(LAYER_CONTROL_MODES)
COORDINATED_LAYER_MODE_NAME = "coordinated"
STRATEGIES = {
    "base": StrategySpec("base", True, True),
    "dynamic": StrategySpec("dynamic", True, True, use_dynamic_weights=True),
    "odte_lyco": StrategySpec(
        "odte_lyco", True, True, use_dynamic_weights=True, cost_aggregation="bottleneck"
    ),
    "no_y": StrategySpec(
        "no_y", True, True, use_dynamic_weights=True,
        cost_aggregation="bottleneck", use_energy_feedback=False,
    ),
}
