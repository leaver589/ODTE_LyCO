"""Fast/slow DT placement with explicit migration costs and state transfer."""

from __future__ import annotations
from dataclasses import dataclass, field
from math import exp, log1p
from typing import Mapping
from .config import ExperimentConfig
from .models import MigrationEvent, NetworkState, Placement
from .odte import ODTEMonitor, ODTESnapshot


@dataclass(frozen=True)
class PlacementControlState:
    queue_workloads: Mapping[str, float]
    mean_sensor_energy_debt: float
    service_queue_workloads: Mapping[str, Mapping[str, float]] = field(
        default_factory=dict
    )


@dataclass
class _WindowObservation:
    samples: int = 0
    se_success_sum: float = 0.0
    ec_success_sum: float = 0.0
    se_available_sum: int = 0
    ec_available_sum: int = 0
    edge_gateway_sum: int = 0
    edge_compute_sum: int = 0
    cloud_sum: int = 0

    def observe(self, state: NetworkState) -> None:
        self.samples += 1
        self.se_success_sum += state.se_success_probability
        self.ec_success_sum += state.ec_success_probability
        self.se_available_sum += int(state.se_available)
        self.ec_available_sum += int(state.ec_available)
        self.edge_gateway_sum += int(state.edge_gateway_available)
        self.edge_compute_sum += int(state.edge_compute_available)
        self.cloud_sum += int(state.cloud_available)

    def mean(self, value: float | int) -> float:
        return float(value) / max(1, self.samples)


@dataclass(frozen=True)
class _PlacementEstimate:
    odte: float
    availability: float
    queue_workload: float


@dataclass
class PlacementManager:
    config: ExperimentConfig
    initial_location: Placement | None = None
    initial_locations: Mapping[str, Placement] | None = None
    use_energy_feedback: bool = True
    locations: dict[str, Placement] = field(init=False)
    last_migration_frame: dict[str, int] = field(init=False)
    migration_until_slot: dict[str, int] = field(init=False)
    pending_events: dict[str, MigrationEvent] = field(init=False)
    last_completed_events: dict[str, MigrationEvent] = field(init=False)
    score_ema: dict[str, dict[Placement, float]] = field(init=False)
    observation: _WindowObservation = field(default_factory=_WindowObservation)
    migrations: int = 0

    def __post_init__(self) -> None:
        initial = self.initial_location or Placement(
            str(self.config.placement["initial"])
        )
        overrides = self.initial_locations or {}
        self.locations = {
            name: overrides.get(name, initial) for name in self.config.service_classes
        }
        self.last_migration_frame = {
            name: -(10**9) for name in self.config.service_classes
        }
        self.migration_until_slot = {name: -1 for name in self.config.service_classes}
        self.pending_events = {}
        self.last_completed_events = {}
        self.score_ema = {
            name: {Placement.EDGE: 0.0, Placement.CLOUD: 0.0}
            for name in self.config.service_classes
        }

    def location(self, service_class: str) -> Placement:
        return self.locations[service_class]

    def is_migrating(self, service_class: str, slot: int) -> bool:
        return (
            service_class in self.pending_events
            or slot < self.migration_until_slot[service_class]
        )

    def complete_migration(self, service_class: str, slot: int) -> MigrationEvent:
        event = self.pending_events.pop(service_class)
        self.locations[service_class] = event.target
        self.migration_until_slot[service_class] = slot + int(
            self.config.placement["migration_slots"]
        )
        self.last_completed_events[service_class] = event
        return event

    def observe(self, state: NetworkState) -> None:
        self.observation.observe(state)

    def _counterfactual_queue_workload(
        self, service_class: str, location: Placement, control: PlacementControlState
    ) -> float:
        workloads = control.service_queue_workloads.get(
            service_class, control.queue_workloads
        )
        common = sum(
            (
                value
                for key, value in workloads.items()
                if key.startswith("source_") or key.startswith("local_results_")
            )
        )
        if location is Placement.EDGE:
            downstream = (
                workloads.get("edge_compute", 0.0)
                + workloads.get("cloud_to_edge", 0.0)
                + workloads.get("pending_at_edge", 0.0)
            )
        else:
            downstream = (
                workloads.get("edge_to_cloud", 0.0)
                + workloads.get("cloud_compute", 0.0)
                + workloads.get("pending_at_cloud", 0.0)
            )
        return max(0.0, float(common + downstream))

    def _placement_estimate(
        self,
        service_class: str,
        location: Placement,
        compute_load_slots: float,
        control: PlacementControlState,
    ) -> _PlacementEstimate:
        observation = self.observation
        if observation.samples <= 0:
            return _PlacementEstimate(0.0, 0.0, 0.0)
        deadline = max(1, self.config.service_classes[service_class].deadline_slots)
        base_delay = 1.0 if location is Placement.EDGE else 2.0
        delay = base_delay + max(0.0, compute_load_slots)
        timeliness = 1.0 if delay <= deadline else exp(-(delay - deadline) / deadline)
        se_reliability = observation.mean(observation.se_success_sum)
        ec_reliability = observation.mean(observation.ec_success_sum)
        se_available = observation.mean(observation.se_available_sum)
        ec_available = observation.mean(observation.ec_available_sum)
        gateway_available = observation.mean(observation.edge_gateway_sum)
        edge_compute_available = observation.mean(observation.edge_compute_sum)
        cloud_available = observation.mean(observation.cloud_sum)
        if location is Placement.EDGE:
            reliability = se_reliability
            availability = se_available * gateway_available * edge_compute_available
        else:
            reliability = se_reliability * ec_reliability
            availability = (
                se_available * gateway_available * ec_available * cloud_available
            )
        odte = max(0.0, min(1.0, timeliness * reliability * availability))
        return _PlacementEstimate(
            odte=odte,
            availability=max(0.0, min(1.0, availability)),
            queue_workload=self._counterfactual_queue_workload(
                service_class, location, control
            ),
        )

    def _placement_odte(
        self, service_class: str, location: Placement, load_slots: float
    ) -> float:
        return self._placement_estimate(
            service_class,
            location,
            load_slots,
            PlacementControlState(queue_workloads={}, mean_sensor_energy_debt=0.0),
        ).odte

    def _joint_utility(
        self,
        *,
        odte: float,
        queue_workload: float,
        trust_debt: float,
        mean_sensor_energy_debt: float,
        use_lyapunov: bool,
    ) -> float:
        placement = self.config.placement
        debt_reference = max(1e-12, float(placement["joint_trust_debt_reference"]))
        trust_pressure = min(1.0, max(0.0, trust_debt / debt_reference))
        trust_scale = 1.0 + float(placement["joint_trust_gain_boost"]) * trust_pressure
        utility = trust_scale * odte
        if not use_lyapunov:
            return utility
        energy_reference = max(1e-12, float(placement["joint_energy_debt_reference_j"]))
        energy_pressure = (
            min(1.0, max(0.0, mean_sensor_energy_debt / energy_reference))
            if self.use_energy_feedback else 0.0
        )
        queue_reference = max(1e-12, float(placement["joint_queue_reference_slots"]))
        queue_weight = float(placement["joint_queue_weight"]) * (
            1.0 + float(placement["joint_energy_queue_coupling"]) * energy_pressure
        )
        return utility - queue_weight * queue_workload / queue_reference

    def update(
        self,
        *,
        frame_index: int,
        slot: int,
        edge_load: float,
        cloud_load: float,
        control: PlacementControlState,
        snapshots: list[ODTESnapshot],
        monitor: ODTEMonitor,
        use_lyapunov: bool,
        force_evaluation: bool = False,
        min_residence_decisions: int | None = None,
        score_ema_alpha: float | None = None,
    ) -> list[MigrationEvent]:
        if self.observation.samples <= 0:
            return []
        placement = self.config.placement
        min_residence = int(
            placement["min_residence_frames"]
            if min_residence_decisions is None
            else min_residence_decisions
        )
        base_margin = float(placement["switch_margin"])
        migration_slots = int(placement["migration_slots"])
        migration_penalty = float(placement["migration_penalty"])
        ema_alpha = float(
            placement["score_ema_alpha"] if score_ema_alpha is None else score_ema_alpha
        )
        availability_debt_trigger = float(placement["availability_debt_trigger"])
        max_per_frame = int(placement["max_migrations_per_frame"])
        snapshot_by_class = {item.service_class: item for item in snapshots}
        estimates: dict[str, dict[Placement, _PlacementEstimate]] = {}
        for name in self.locations:
            class_workloads = control.service_queue_workloads.get(name, {})
            class_edge_load = class_workloads.get("edge_compute", edge_load)
            class_cloud_load = class_workloads.get("cloud_compute", cloud_load)
            estimates[name] = {
                Placement.EDGE: self._placement_estimate(
                    name, Placement.EDGE, class_edge_load, control
                ),
                Placement.CLOUD: self._placement_estimate(
                    name, Placement.CLOUD, class_cloud_load, control
                ),
            }
            for location, estimate in estimates[name].items():
                previous = self.score_ema[name][location]
                self.score_ema[name][location] = (
                    estimate.odte
                    if frame_index == 0
                    else ema_alpha * estimate.odte + (1.0 - ema_alpha) * previous
                )
        horizon = max(1, int(placement["migration_evaluation_horizon_slots"]))
        candidates: list[tuple[float, MigrationEvent]] = []
        for name, current in self.locations.items():
            if name in self.pending_events:
                continue
            snapshot = snapshot_by_class.get(name)
            availability_debt = monitor.debt(name, "availability")
            trust_debt = monitor.debt(name)
            class_queue_pressure = sum(
                control.service_queue_workloads.get(name, {}).values()
            )
            queue_trigger = float(placement["queue_migration_trigger_slots"])
            has_pressure = (
                force_evaluation
                or (snapshot is not None and snapshot.violation)
                or availability_debt >= availability_debt_trigger
                or (use_lyapunov and class_queue_pressure >= queue_trigger)
            )
            if not has_pressure:
                continue
            if frame_index - self.last_migration_frame[name] < min_residence:
                continue
            alternative = (
                Placement.CLOUD if current is Placement.EDGE else Placement.EDGE
            )
            state_bits = float(placement["dt_state_bits"][name])
            nominal_backhaul = max(
                1.0, float(self.config.resources["ec_rate_bits_per_slot"])
            )
            transfer_slots = state_bits / nominal_backhaul
            downtime_fraction = min(1.0, (migration_slots + transfer_slots) / horizon)
            current_estimate = estimates[name][current]
            alternative_estimate = estimates[name][alternative]
            current_odte = self.score_ema[name][current]
            alternative_odte = self.score_ema[name][alternative] * (
                1.0 - downtime_fraction
            )
            if use_lyapunov and alternative_odte < current_odte - float(
                placement["max_joint_odte_sacrifice"]
            ):
                continue
            current_utility = self._joint_utility(
                odte=current_odte,
                queue_workload=current_estimate.queue_workload,
                trust_debt=trust_debt,
                mean_sensor_energy_debt=control.mean_sensor_energy_debt,
                use_lyapunov=use_lyapunov,
            )
            alternative_utility = self._joint_utility(
                odte=alternative_odte,
                queue_workload=alternative_estimate.queue_workload,
                trust_debt=trust_debt,
                mean_sensor_energy_debt=control.mean_sensor_energy_debt,
                use_lyapunov=use_lyapunov,
            )
            utility_gain = alternative_utility - current_utility
            required_gain = base_margin + migration_penalty * downtime_fraction
            if utility_gain <= required_gain:
                continue
            event = MigrationEvent(
                frame_index=frame_index,
                slot=slot,
                service_class=name,
                source=current,
                target=alternative,
                predicted_odte_gain=alternative_odte - current_odte,
                predicted_availability_gain=alternative_estimate.availability
                * (1.0 - downtime_fraction)
                - current_estimate.availability,
                predicted_queue_reduction=current_estimate.queue_workload
                - alternative_estimate.queue_workload,
                sensor_energy_debt_per_sensor=control.mean_sensor_energy_debt,
                joint_utility_gain=utility_gain,
                required_utility_gain=required_gain,
                reverses_recent_migration=name in self.last_completed_events
                and self.last_completed_events[name].source is alternative
                and (
                    frame_index - self.last_completed_events[name].frame_index
                    <= int(placement["ineffective_migration_horizon_frames"])
                    * (self.config.monitor_window if force_evaluation else 1)
                ),
            )
            priority = (utility_gain - required_gain) * (1.0 + log1p(trust_debt))
            candidates.append((priority, event))
        events: list[MigrationEvent] = []
        for _, event in sorted(
            candidates, key=lambda item: (item[0], item[1].service_class), reverse=True
        ):
            if len(events) >= max_per_frame:
                break
            self.last_migration_frame[event.service_class] = frame_index
            self.pending_events[event.service_class] = event
            self.migrations += 1
            events.append(event)
        self.observation = _WindowObservation()
        return events
