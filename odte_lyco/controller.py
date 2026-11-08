"""Online route/resource search, dynamic weights and smooth bottleneck objective."""

from __future__ import annotations
from dataclasses import dataclass
from math import exp, log, log1p, log2, sqrt
from typing import Callable, Mapping, Sequence
import numpy as np
from heapq import heappop, heappush
from functools import cached_property
from .config import ExperimentConfig
from .models import NetworkState, Placement, Route, SourceAction, StrategySpec, Task
from .odte import COMPONENTS, ODTEMonitor
from .placement import PlacementManager
from .queues import TaskQueues, VirtualQueue


@dataclass(frozen=True)
class ControllerContext:
    state: NetworkState
    queues: TaskQueues
    energy_queues: Sequence[VirtualQueue]
    monitor: ODTEMonitor
    placement: PlacementManager
    energy_budget_per_slot: float | None = None
    sensor_energy_budgets: Sequence[float] | None = None
    arrival_pressure: float = 0.0
    queue_growth_pressure: float = 0.0
    queue_pressure: float = 0.0


@dataclass(frozen=True)
class _Prediction:
    delay: float
    timeliness: float
    reliability: float
    availability: float
    energy: float

    @cached_property
    def component_losses(self) -> dict[str, float]:
        # equation (21): predicted component log losses with epsilon = 1e-6.
        return {
            "timeliness": -log(max(1e-06, self.timeliness)),
            "reliability": -log(max(1e-06, self.reliability)),
            "availability": -log(max(1e-06, self.availability)),
        }


@dataclass(frozen=True)
class _ControlProfile:
    v_delay: float
    queue_weight: float
    energy_weight: float
    resource_cost_weight: float
    trust_weight: float


class LycoController:

    def __init__(self, config: ExperimentConfig, strategy: StrategySpec) -> None:
        self.config = config
        self.strategy = strategy
        self.slot_seconds = config.slot_seconds
        self.resources = config.resources
        self.energy = config.energy
        self.ctrl = config.controller
        self._frequencies = np.linspace(
            float(self.resources["sensor_cpu_min_hz"]),
            float(self.resources["sensor_cpu_max_hz"]),
            int(self.ctrl["cpu_search_points"]),
        )
        self._powers = np.linspace(
            float(self.resources["tx_power_min_w"]),
            float(self.resources["tx_power_max_w"]),
            int(self.ctrl["power_search_points"]),
        )

    def _base_weight(self, name: str) -> float:
        profile = self.ctrl["fixed_weight_profiles"].get(self.strategy.name, self.ctrl)
        return float(profile[name])

    def _physical_se_rate_bits(self, power_w: float) -> float:
        if power_w <= 0.0:
            return 0.0
        bandwidth = float(self.resources["se_bandwidth_hz"])
        pmax = float(self.resources["tx_power_max_w"])
        snr = float(self.resources["se_snr_at_max_power"])
        # equation (6): physical wireless transmission volume per slot.
        return bandwidth * self.slot_seconds * log2(1.0 + snr * power_w / pmax)

    def se_rate_bits(self, power_w: float, state: NetworkState) -> float:
        se_available = (
            state.sensor_se_available
            if state.sensor_se_available is not None
            else state.se_available
        )
        if not se_available:
            return 0.0
        return self._physical_se_rate_bits(power_w)

    def se_success_probability(self, power_w: float, state: NetworkState) -> float:
        se_available = (
            state.sensor_se_available
            if state.sensor_se_available is not None
            else state.se_available
        )
        if not se_available or power_w <= 0.0:
            return 0.0
        pmax = float(self.resources["tx_power_max_w"])
        power_ratio = min(1.0, max(0.0, power_w / pmax))
        # equation (7): power-dependent link success probability.
        quality = 0.55 + 0.45 * sqrt(power_ratio)
        base_probability = (
            state.sensor_se_success_probability
            if state.sensor_se_success_probability is not None
            else state.se_success_probability
        )
        return min(1.0, base_probability * quality)


    def _expected_se_capacity(self, state: NetworkState) -> float:
        # equation (11): positive conversion scale for global or sensor-specific q_SE.
        pmax = float(self.resources["tx_power_max_w"])
        probability = max(
            0.05,
            (
                state.sensor_se_success_probability
                if state.sensor_se_success_probability is not None
                else state.se_success_probability
            ),
        )
        return max(1.0, self._physical_se_rate_bits(pmax) * probability)

    def _expected_ec_capacity(self, state: NetworkState) -> float:
        # equation (12): positive backhaul conversion scale; actual failures remain explicit.
        probability = max(0.05, state.ec_success_probability)
        return max(
            1.0,
            float(self.resources["ec_rate_bits_per_slot"])
            * state.ec_capacity_multiplier
            * probability,
        )

    def _edge_capacity(self, state: NetworkState) -> float:
        # equation (13): positive edge CPU conversion scale.
        return max(
            1.0,
            float(self.resources["edge_cpu_hz"])
            * self.slot_seconds
            * state.edge_capacity_multiplier,
        )

    def _cloud_capacity(self, state: NetworkState) -> float:
        # equation (13): positive cloud CPU conversion scale.
        return max(
            1.0,
            float(self.resources["cloud_cpu_hz"])
            * self.slot_seconds
            * state.cloud_capacity_multiplier,
        )

    def _source_task_workload(self, task: Task, state: NetworkState) -> float:
        # equation (14): remaining CPU/upload service slots under the locked route.
        local = task.remaining_cycles / max(
            1.0, float(self.resources["sensor_cpu_max_hz"]) * self.slot_seconds
        )
        remote = task.remaining_bits / self._expected_se_capacity(state)
        if task.route is Route.LOCAL:
            return local
        if task.route in (Route.EDGE, Route.CLOUD):
            return remote
        return min(local, remote)

    def queue_workloads(
        self, context: ControllerContext, service_class: str | None = None
    ) -> dict[str, float]:
        state = context.state
        queues = context.queues
        se_capacity = self._expected_se_capacity(state)
        ec_capacity = self._expected_ec_capacity(state)
        edge_capacity = self._edge_capacity(state)
        cloud_capacity = self._cloud_capacity(state)
        selected = (
            lambda task: service_class is None or task.service_class == service_class
        )
        workloads: dict[str, float] = {}
        for sensor_id, queue in enumerate(queues.source):
            workloads[f"source_{sensor_id}"] = sum(
                (
                    self._source_task_workload(task, state)
                    for task in queue
                    if selected(task)
                )
            )
        for sensor_id, queue in enumerate(queues.local_results):
            workloads[f"local_results_{sensor_id}"] = sum(
                (task.remaining_bits / se_capacity for task in queue if selected(task))
            )
        workloads["edge_compute"] = sum(
            (
                task.remaining_cycles / edge_capacity
                for task in queues.edge_compute
                if selected(task)
            )
        )
        workloads["edge_to_cloud"] = sum(
            (
                task.remaining_bits / ec_capacity
                for task in queues.edge_to_cloud
                if selected(task)
            )
        )
        workloads["cloud_compute"] = sum(
            (
                task.remaining_cycles / cloud_capacity
                for task in queues.cloud_compute
                if selected(task)
            )
        )
        workloads["cloud_to_edge"] = sum(
            (
                task.remaining_bits / ec_capacity
                for task in queues.cloud_to_edge
                if selected(task)
            )
        )
        workloads["pending_at_edge"] = float(
            sum((1 for task in queues.pending_at_edge if selected(task)))
        )
        workloads["pending_at_cloud"] = float(
            sum((1 for task in queues.pending_at_cloud if selected(task)))
        )
        dependency_workload = 0.0
        sensor_capacity = max(
            1.0, float(self.resources["sensor_cpu_max_hz"]) * self.slot_seconds
        )
        for task in queues.dependency_wait:
            if not selected(task):
                continue
            if task.route is Route.LOCAL:
                capacity = sensor_capacity
            elif task.route is Route.EDGE:
                capacity = edge_capacity
            elif task.route is Route.CLOUD:
                capacity = cloud_capacity
            else:
                capacity = max(sensor_capacity, edge_capacity, cloud_capacity)
            dependency_workload += task.remaining_cycles / max(1.0, capacity)
        workloads["dependency_wait"] = dependency_workload
        return workloads

    def _route_destination_workload(
        self,
        task: Task,
        route: Route,
        context: ControllerContext,
        workloads: Mapping[str, float] | None = None,
    ) -> float:
        workloads = (
            workloads if workloads is not None else self.queue_workloads(context)
        )
        if route is Route.LOCAL:
            return workloads[f"local_results_{task.sensor_id}"]
        if route is Route.EDGE:
            return workloads["edge_compute"]
        return workloads["edge_to_cloud"] + workloads["cloud_compute"]

    def criticality(
        self,
        task: Task,
        context: ControllerContext,
        workloads: Mapping[str, float] | None = None,
        *,
        component_debts: Mapping[str, float] | None = None,
    ) -> float:
        if not bool(self.ctrl["adaptive_mode_enabled"]):
            return 0.0
        deadline_pressure = min(
            1.0, task.age(context.state.slot) / max(1, task.deadline_slots)
        )
        workloads = (
            workloads if workloads is not None else self.queue_workloads(context)
        )
        source_work = workloads[f"source_{task.sensor_id}"]
        queue_reference = max(
            1e-06, float(self.ctrl["criticality_queue_reference_slots"])
        )
        queue_pressure = min(1.0, source_work / queue_reference)
        route_availability = [
            self._path_factors(task, Route.LOCAL, context)[1],
            self._path_factors(task, Route.EDGE, context)[1],
            self._path_factors(task, Route.CLOUD, context)[1],
        ]
        network_pressure = 1.0 - min(1.0, max(route_availability))
        trust_pressure = 0.0
        if self.strategy.use_odte:
            trust_reference = max(
                1e-06, float(self.ctrl["criticality_trust_reference"])
            )
            total_trust_debt = (
                sum((max(0.0, float(value)) for value in component_debts.values()))
                if component_debts is not None
                else context.monitor.debt(task.service_class)
            )
            trust_pressure = min(1.0, total_trust_debt / trust_reference)
        weights = (
            float(self.ctrl["criticality_deadline_weight"]),
            float(self.ctrl["criticality_queue_weight"]),
            float(self.ctrl["criticality_network_weight"]),
            float(self.ctrl["criticality_trust_weight"]),
        )
        # equation (48): weighted deadline, source queue, path and ODTE pressures.
        weighted_pressure = (
            weights[0] * deadline_pressure
            + weights[1] * queue_pressure
            + weights[2] * network_pressure
            + weights[3] * trust_pressure
        )
        base_criticality = weighted_pressure / max(1e-12, sum(weights))
        arrival_pressure = min(1.0, max(0.0, context.arrival_pressure))
        queue_growth_pressure = min(1.0, max(0.0, context.queue_growth_pressure))
        event_pressure = max(arrival_pressure, queue_growth_pressure)
        # equation (49): immediate arrival pressure and prior-slot queue-slope pressure.
        event_guard = float(self.ctrl["criticality_event_floor"]) * event_pressure
        return float(max(0.0, min(1.0, max(base_criticality, event_guard))))

    def _control_profile(
        self,
        task: Task,
        context: ControllerContext,
        workloads: Mapping[str, float] | None = None,
        *,
        component_debts: Mapping[str, float] | None = None,
    ) -> _ControlProfile:
        dynamic_enabled = bool(
            self.strategy.use_lyapunov
            and self.strategy.use_dynamic_weights
            and self.ctrl["dynamic_weights_enabled"]
        )
        criticality = (
            self.criticality(task, context, workloads, component_debts=component_debts)
            if dynamic_enabled
            else 0.0
        )
        congestion = (
            min(1.0, max(0.0, float(context.queue_pressure)))
            if dynamic_enabled
            else 0.0
        )
        local_gating_enabled = bool(
            dynamic_enabled and self.ctrl["dynamic_local_gating_enabled"]
        )
        # equation (53): bounded dynamic delay multiplier.
        delay_multiplier = (
            1.0
            + float(self.ctrl["criticality_delay_boost"]) * criticality
            + float(self.ctrl["dynamic_delay_gain"]) * congestion
        )
        delay_multiplier = min(
            float(self.ctrl["dynamic_delay_max_multiplier"]), delay_multiplier
        )
        queue_multiplier = 1.0 + (
            0.0
            if local_gating_enabled
            else float(self.ctrl["dynamic_queue_gain"]) * congestion
        )
        queue_multiplier = min(
            float(self.ctrl["dynamic_queue_max_multiplier"]), queue_multiplier
        )
        # equation (56): criticality relief, deficit recovery and fusion multiplier.
        energy_multiplier = max(
            0.0, 1.0 - float(self.ctrl["criticality_energy_relief"]) * criticality
        )
        if dynamic_enabled:
            energy_multiplier /= (
                1.0 + float(self.ctrl["dynamic_energy_relief"]) * congestion
            )
            energy_multiplier = max(
                float(self.ctrl["dynamic_energy_min_multiplier"]), energy_multiplier
            )
        if local_gating_enabled:
            debt_pressure = self._energy_debt_pressure(task.sensor_id, context)
            recovery = debt_pressure * (1.0 - congestion)
            deadline_threshold = min(
                1.0,
                max(
                    0.0, float(self.ctrl["dynamic_energy_recovery_deadline_threshold"])
                ),
            )
            deadline_pressure = min(
                1.0,
                max(
                    0.0,
                    task.age(context.state.slot) / max(1.0, float(task.deadline_slots)),
                ),
            )
            if deadline_pressure > deadline_threshold:
                deadline_guard = (deadline_pressure - deadline_threshold) / max(
                    1e-12, 1.0 - deadline_threshold
                )
                recovery *= 1.0 - min(1.0, deadline_guard)
            recovery_multiplier = (
                1.0 + float(self.ctrl["dynamic_energy_recovery_gain"]) * recovery
            )
            recovery_multiplier = min(
                float(self.ctrl["dynamic_energy_recovery_max_multiplier"]),
                recovery_multiplier,
            )
            energy_multiplier *= recovery_multiplier
        fusion_energy_multiplier = (
            float(self.ctrl["fusion_energy_weight_multiplier"])
            if self.strategy.use_lyapunov and self.strategy.use_odte
            else 1.0
        )
        return _ControlProfile(
            v_delay=self._base_weight("v_delay") * delay_multiplier,
            queue_weight=self._base_weight("queue_weight") * queue_multiplier,
            energy_weight=self._base_weight("energy_weight")
            * energy_multiplier
            * fusion_energy_multiplier,
            # equation (54): instantaneous-resource weight.
            resource_cost_weight=self._base_weight("resource_cost_weight")
            * max(
                0.0, 1.0 - float(self.ctrl["criticality_resource_relief"]) * criticality
            ),
            # equation (55): ODTE deficit weight.
            trust_weight=self._base_weight("trust_weight")
            * (1.0 + float(self.ctrl["criticality_trust_boost"]) * criticality),
        )

    def _energy_debt_pressure(
        self, sensor_id: int, context: ControllerContext
    ) -> float:
        if not self.strategy.use_lyapunov or not self.strategy.use_energy_feedback:
            return 0.0
        budget = self._sensor_energy_budget(sensor_id, context)
        reference = max(
            1e-12,
            budget * float(self.ctrl["dynamic_energy_debt_reference_budget_slots"]),
        )
        return float(
            min(1.0, max(0.0, context.energy_queues[sensor_id].value / reference))
        )

    def _sensor_energy_budget(
        self, sensor_id: int, context: ControllerContext
    ) -> float:
        if context.sensor_energy_budgets is not None:
            return float(context.sensor_energy_budgets[sensor_id])
        return float(
            context.energy_budget_per_slot
            if context.energy_budget_per_slot is not None
            else self.energy["budget_j_per_slot"]
        )

    def _local_queue_pressure(
        self,
        task: Task,
        context: ControllerContext,
        workloads: dict[str, float] | None = None,
    ) -> float:
        workloads = (
            workloads if workloads is not None else self.queue_workloads(context)
        )
        local_work = (
            workloads[f"source_{task.sensor_id}"]
            + workloads[f"local_results_{task.sensor_id}"]
        )
        reference = max(1e-12, float(self.ctrl["dynamic_local_queue_reference_slots"]))
        return float(min(1.0, max(0.0, local_work / reference)))

    def _route_capacity_gate(
        self,
        task: Task,
        route: Route,
        context: ControllerContext,
        workloads: dict[str, float] | None = None,
    ) -> float:
        workloads = (
            workloads if workloads is not None else self.queue_workloads(context)
        )
        reliability, availability = self._path_factors(
            task, route, context, power_w=float(self.resources["tx_power_max_w"])
        )
        destination = self._route_destination_workload(task, route, context, workloads)
        queue_reference = max(
            1e-12, float(self.ctrl["dynamic_path_queue_reference_slots"])
        )
        backlog_gate = 1.0 / (1.0 + destination / queue_reference)
        capacity_floor = min(
            1.0, max(0.0, float(self.ctrl["dynamic_path_capacity_floor"]))
        )
        # equation (57): route gate evaluated at maximum power for every candidate on the route.
        available_capacity = (
            capacity_floor + (1.0 - capacity_floor) * reliability * backlog_gate
        )
        return float(min(1.0, max(0.0, availability * available_capacity)))

    def _effective_queue_weight(
        self,
        task: Task,
        route: Route,
        context: ControllerContext,
        profile: _ControlProfile,
        workloads: dict[str, float] | None = None,
    ) -> float:
        if not (
            self.strategy.use_lyapunov
            and self.strategy.use_dynamic_weights
            and self.ctrl["dynamic_weights_enabled"]
            and self.ctrl["dynamic_local_gating_enabled"]
        ):
            return profile.queue_weight
        congestion = min(1.0, max(0.0, float(context.queue_pressure)))
        local_pressure = self._local_queue_pressure(task, context, workloads)
        debt_gate_floor = min(
            1.0, max(0.0, float(self.ctrl["dynamic_energy_debt_gate_floor"]))
        )
        debt_gate = debt_gate_floor + (1.0 - debt_gate_floor) * (
            1.0 - self._energy_debt_pressure(task.sensor_id, context)
        )
        deadline_threshold = min(
            1.0, max(0.0, float(self.ctrl["dynamic_queue_deadline_threshold"]))
        )
        deadline_pressure = min(
            1.0,
            max(
                0.0, task.age(context.state.slot) / max(1.0, float(task.deadline_slots))
            ),
        )
        if deadline_pressure > deadline_threshold:
            deadline_gate = (deadline_pressure - deadline_threshold) / max(
                1e-12, 1.0 - deadline_threshold
            )
            debt_gate = max(debt_gate, min(1.0, deadline_gate))
        path_gate = self._route_capacity_gate(task, route, context, workloads)
        # equation (58): congestion, local backlog, energy and deadline gated queue weight.
        multiplier = (
            1.0
            + float(self.ctrl["dynamic_queue_gain"])
            * congestion
            * local_pressure
            * debt_gate
            * path_gate
        )
        multiplier = min(float(self.ctrl["dynamic_queue_max_multiplier"]), multiplier)
        return float(self.ctrl["queue_weight"]) * multiplier

    def _path_factors(
        self,
        task: Task,
        route: Route,
        context: ControllerContext,
        power_w: float | None = None,
    ) -> tuple[float, float]:
        # equation (9): multiply directed-hop success probabilities, counting backhaul return.
        # equation (10): multiply required availability factors, with migration attenuation.
        state = context.state
        twin = context.placement.location(task.service_class)
        power = (
            float(self.resources["tx_power_max_w"])
            if power_w is None
            else float(power_w)
        )
        se_r = max(1e-06, self.se_success_probability(power, state))
        ec_r = max(1e-06, state.ec_success_probability if state.ec_available else 0.0)
        if route is Route.LOCAL:
            reliability = se_r * (ec_r if twin is Placement.CLOUD else 1.0)
            availability = float(
                (
                    state.sensor_se_available
                    if state.sensor_se_available is not None
                    else state.se_available
                )
                and state.edge_gateway_available
            )
            if twin is Placement.EDGE:
                availability *= float(state.edge_compute_available)
            else:
                availability *= float(state.ec_available and state.cloud_available)
        elif route is Route.EDGE:
            reliability = se_r * (ec_r if twin is Placement.CLOUD else 1.0)
            availability = float(
                (
                    state.sensor_se_available
                    if state.sensor_se_available is not None
                    else state.se_available
                )
                and state.edge_gateway_available
                and state.edge_compute_available
            )
            if twin is Placement.CLOUD:
                availability *= float(state.ec_available and state.cloud_available)
        else:
            reliability = se_r * ec_r
            availability = float(
                (
                    state.sensor_se_available
                    if state.sensor_se_available is not None
                    else state.se_available
                )
                and state.edge_gateway_available
                and state.ec_available
                and state.cloud_available
            )
            if twin is Placement.EDGE:
                reliability *= ec_r
                availability *= float(state.edge_compute_available)
        if context.placement.is_migrating(task.service_class, state.slot):
            availability *= 0.001
        return (max(1e-06, reliability), max(1e-06, availability))

    def _predict(
        self,
        task: Task,
        route: Route,
        frequency_hz: float,
        power_w: float,
        context: ControllerContext,
        workloads: Mapping[str, float] | None = None,
    ) -> _Prediction:
        state = context.state
        age = task.age(state.slot)
        pmax = float(self.resources["tx_power_max_w"])
        ec_capacity = self._expected_ec_capacity(state)
        if route is Route.LOCAL:
            # equation (27): local node latency, result upload and authoritative-DT forwarding.
            service = max(1.0, frequency_hz * self.slot_seconds)
            compute_slots = task.remaining_cycles / service
            result_slots = task.output_bits / self._expected_se_capacity(state)
            sync_slots = (
                task.output_bits / ec_capacity
                if context.placement.location(task.service_class) is Placement.CLOUD
                else 0.0
            )
            communication_slots = result_slots + sync_slots
            downstream = self._route_destination_workload(
                task, route, context, workloads
            )
            active_fraction = min(1.0, task.remaining_cycles / service)
            # equation (33): candidate local CPU energy for at most one slot.
            action_energy = (
                float(self.energy["cpu_coefficient"])
                * frequency_hz**3
                * self.slot_seconds
                * active_fraction
            )
            path_power = pmax
        elif route is Route.EDGE:
            # equation (28): edge upload, computation, forwarding and destination backlog.
            rate = max(1.0, self.se_rate_bits(power_w, state))
            communication_slots = task.remaining_bits / max(1.0, rate)
            if context.placement.location(task.service_class) is Placement.CLOUD:
                communication_slots += task.output_bits / ec_capacity
            compute_slots = task.cpu_cycles / self._edge_capacity(state)
            downstream = self._route_destination_workload(
                task, route, context, workloads
            )
            active_fraction = min(1.0, task.remaining_bits / rate)
            # equation (34): candidate upload energy uses physical capacity.
            action_energy = power_w * self.slot_seconds * active_fraction
            path_power = power_w
        else:
            # equation (29): cloud latency includes the full node input on the backhaul.
            rate = max(1.0, self.se_rate_bits(power_w, state))
            communication_slots = task.remaining_bits / max(1.0, rate)
            communication_slots += task.input_bits / ec_capacity
            if context.placement.location(task.service_class) is Placement.EDGE:
                communication_slots += task.output_bits / ec_capacity
            compute_slots = task.cpu_cycles / self._cloud_capacity(state)
            downstream = self._route_destination_workload(
                task, route, context, workloads
            )
            active_fraction = min(1.0, task.remaining_bits / rate)
            action_energy = power_w * self.slot_seconds * active_fraction
            path_power = power_w
        predicted_delay = age + compute_slots + communication_slots + downstream
        deadline = max(1.0, float(task.deadline_slots))
        # equation (30): exponential timeliness proxy after the deadline.
        timeliness = (
            1.0
            if predicted_delay <= deadline
            else exp(-(predicted_delay - deadline) / deadline)
        )
        path_reliability, path_availability = self._path_factors(
            task, route, context, power_w=path_power
        )
        # equation (32): epsilon-clipped path availability.
        availability = path_availability
        deadline_factor = (
            1.0
            if predicted_delay <= deadline
            else exp(-4.0 * (predicted_delay - deadline) / deadline)
        )
        # equation (31): path reliability with a fourfold deadline-overrun discount.
        reliability = max(1e-06, path_reliability * deadline_factor)
        return _Prediction(
            delay=predicted_delay,
            timeliness=max(1e-06, timeliness),
            reliability=reliability,
            availability=availability,
            energy=action_energy,
        )

    def _service_rate_in_workload_units(
        self,
        task: Task,
        route: Route,
        frequency_hz: float,
        power_w: float,
        context: ControllerContext,
    ) -> float:
        if route is Route.LOCAL:
            return frequency_hz / max(1.0, float(self.resources["sensor_cpu_max_hz"]))
        expected_bits = self.se_rate_bits(
            power_w, context.state
        ) * self.se_success_probability(power_w, context.state)
        return expected_bits / self._expected_se_capacity(context.state)

    def _performance_components(
        self, task, prediction, context, profile, budget, *, delay_reference,
        component_debts=None,
    ) -> list[tuple[float, float]]:
        components = [
            # equation (59): delay weight and deadline-normalized node latency.
            (profile.v_delay, prediction.delay / max(1e-12, delay_reference)),
            # equation (60): budget-scaled resource weight and normalized candidate energy.
            (profile.resource_cost_weight * budget, prediction.energy / max(1e-12, budget)),
        ]
        if self.strategy.use_odte:
            debts = (component_debts if component_debts is not None
                     else context.monitor.component_debts(task.service_class))
            losses = prediction.component_losses
            # equation (61): each component uses its own deficit and predicted log loss.
            components.extend((profile.trust_weight * debts[k], losses[k]) for k in COMPONENTS)
        return [(max(0.0, w), max(0.0, g)) for w, g in components if w > 0.0]

    def _aggregate_performance_penalty(
        self,
        task: Task,
        prediction: _Prediction,
        context: ControllerContext,
        profile: _ControlProfile,
        budget: float,
        *,
        delay_reference: float,
        component_debts: Mapping[str, float] | None = None,
    ) -> float:
        components = self._performance_components(
            task,
            prediction,
            context,
            profile,
            budget,
            delay_reference=delay_reference,
            component_debts=component_debts,
        )
        if not components:
            return 0.0
        mode = self.strategy.cost_aggregation
        if mode == "additive":
            return sum((weight * loss for weight, loss in components))
        if mode == "bottleneck":
            # equation (62): bounded losses and numerically stabilized weighted LogSumExp.
            active = [
                (weight, 1.0 - exp(-min(700.0, max(0.0, loss))))
                for weight, loss in components
                if weight > 0.0
            ]
            total_weight = sum((weight for weight, _ in active))
            if total_weight <= 0.0:
                return 0.0
            temperature = float(self.ctrl["performance_bottleneck_temperature"])
            maximum = max((value for _, value in active))
            weighted_exp = sum(
                (
                    weight / total_weight * exp((value - maximum) / temperature)
                    for weight, value in active
                )
            )
            smooth_max = maximum + temperature * log(max(1e-300, weighted_exp))
            return total_weight * max(0.0, smooth_max)

    def _trust_barrier(
        self, task: Task, prediction: _Prediction, context: ControllerContext
    ) -> float:
        if not self.strategy.use_odte:
            return 0.0
        barrier = 0.0
        for component, loss in prediction.component_losses.items():
            target = context.monitor.component_target(task.service_class, component)
            component_budget = -log(max(float(self.ctrl["odte_epsilon"]), target))
            barrier += max(0.0, loss - component_budget)
        return self._base_weight("trust_barrier_weight") * barrier

    def _source_action_evaluations(
        self,
        task: Task,
        context: ControllerContext,
        *,
        allow_local: bool = True,
        allow_remote: bool = True,
        workloads: Mapping[str, float] | None = None,
        component_debts: Mapping[str, float] | None = None,
    ) -> list[tuple[SourceAction, _Prediction]]:
        sensor_se_available = (
            context.state.sensor_se_available
            if context.state.sensor_se_available is not None
            else context.state.se_available
        )
        routes = (
            [task.route]
            if task.route is not None
            else [Route.LOCAL, Route.EDGE, Route.CLOUD]
        )
        workloads = (
            workloads if workloads is not None else self.queue_workloads(context)
        )
        if component_debts is None and self.strategy.use_odte:
            component_debts = context.monitor.component_debts(task.service_class)
        profile = self._control_profile(
            task, context, workloads, component_debts=component_debts
        )
        budget = self._sensor_energy_budget(task.sensor_id, context)
        pressure = (context.energy_queues[task.sensor_id].value / max(1e-12, budget)
                    if self.strategy.use_lyapunov and self.strategy.use_energy_feedback else 0.0)
        source_q = workloads[f"source_{task.sensor_id}"]
        evaluations: list[tuple[SourceAction, _Prediction]] = []
        for route in routes:
            if route is None:
                continue
            if route is Route.LOCAL and (not allow_local):
                continue
            if route is not Route.LOCAL and (not allow_remote):
                continue
            if route is Route.EDGE and (
                not (
                    sensor_se_available
                    and context.state.edge_gateway_available
                    and context.state.edge_compute_available
                )
            ):
                continue
            if route is Route.CLOUD and (
                not (
                    sensor_se_available
                    and context.state.edge_gateway_available
                    and context.state.ec_available
                    and context.state.cloud_available
                )
            ):
                continue
            # Route-level terms are shared by all nine resource candidates.
            queue_term = 0.0
            if self.strategy.use_lyapunov:
                destination_q = self._route_destination_workload(task, route, context, workloads)
                queue_term = (self._effective_queue_weight(task, route, context, profile, workloads)
                              * (destination_q - source_q))
            resource_pairs = (
                ((float(frequency), 0.0) for frequency in self._frequencies)
                if route is Route.LOCAL
                else ((0.0, float(power)) for power in self._powers)
            )
            for frequency, power in resource_pairs:
                prediction = self._predict(task, route, frequency, power, context, workloads)
                # equation (35): performance + queue drift + energy deficit + excess-loss penalty.
                objective = self._aggregate_performance_penalty(
                    task, prediction, context, profile, budget,
                    delay_reference=float(max(1, task.deadline_slots)),
                    component_debts=component_debts,
                )
                if self.strategy.use_lyapunov:
                    service = self._service_rate_in_workload_units(task, route, frequency, power, context)
                    objective += queue_term * service
                    objective += profile.energy_weight * pressure * prediction.energy / max(1e-12, budget)
                objective += self._trust_barrier(task, prediction, context)
                evaluations.append(
                    (
                        SourceAction(
                            task_id=task.task_id,
                            route=route,
                            cpu_frequency_hz=frequency,
                            transmit_power_w=power,
                            odte_weight=profile.trust_weight,
                            energy_weight=profile.energy_weight,
                            objective=objective,
                        ),
                        prediction,
                    )
                )
        return evaluations


    def choose_source_action(
        self,
        task: Task,
        context: ControllerContext,
        *,
        allow_local: bool = True,
        allow_remote: bool = True,
        workloads: Mapping[str, float] | None = None,
    ) -> SourceAction | None:
        evaluations = self._source_action_evaluations(
            task,
            context,
            allow_local=allow_local,
            allow_remote=allow_remote,
            workloads=workloads,
        )
        if not evaluations:
            return None
        action, _prediction = min(evaluations, key=lambda item: item[0].objective)
        return action

    def select_source_task(
        self,
        sensor_id: int,
        context: ControllerContext,
        *,
        allow_local: bool = True,
        allow_remote: bool = True,
        priority_heap: list[tuple[float, int, int, Task]] | None = None,
    ) -> Task | None:
        if priority_heap is not None:
            deferred: list[tuple[float, int, int, Task]] = []
            while priority_heap:
                entry = heappop(priority_heap)
                task = entry[3]
                if self._source_task_is_eligible(
                    task, context, allow_local=allow_local, allow_remote=allow_remote
                ):
                    for skipped in deferred:
                        heappush(priority_heap, skipped)
                    return task
                deferred.append(entry)
            for skipped in deferred:
                heappush(priority_heap, skipped)
            return None
        eligible = [
            task
            for task in context.queues.source[sensor_id]
            if self._source_task_is_eligible(
                task, context, allow_local=allow_local, allow_remote=allow_remote
            )
        ]
        if not eligible:
            return None
        return max(eligible, key=self._source_priority_key(context.state.slot, context))

    def build_source_priority_heap(
        self, sensor_id: int, context: ControllerContext
    ) -> list[tuple[float, int, int, Task]]:
        slot = context.state.slot
        priority_key = self._source_priority_key(slot, context)
        heap: list[tuple[float, int, int, Task]] = []
        for task in context.queues.source[sensor_id]:
            priority, deadline, task_id = priority_key(task)
            heappush(heap, (-priority, -deadline, task_id, task))
        return heap

    def restore_source_priority(
        self,
        task: Task,
        context: ControllerContext,
        priority_heap: list[tuple[float, int, int, Task]],
    ) -> None:
        priority_key = self._source_priority_key(context.state.slot, context)
        priority, deadline, task_id = priority_key(task)
        heappush(priority_heap, (-priority, -deadline, task_id, task))

    def _source_priority_key(
        self, slot: int, context: ControllerContext
    ) -> Callable[[Task], tuple[float, int, int]]:
        return lambda task: (
            self.task_priority(task, slot, context.monitor),
            -task.deadline_slot,
            -task.task_id,
        )

    def _source_task_is_eligible(
        self,
        task: Task,
        context: ControllerContext,
        *,
        allow_local: bool,
        allow_remote: bool,
    ) -> bool:
        sensor_se_available = (
            context.state.sensor_se_available
            if context.state.sensor_se_available is not None
            else context.state.se_available
        )
        if task.route is Route.LOCAL and (not allow_local):
            return False
        if task.route in (Route.EDGE, Route.CLOUD) and (not allow_remote):
            return False
        if task.route is Route.EDGE and (
            not (
                sensor_se_available
                and context.state.edge_gateway_available
                and context.state.edge_compute_available
            )
        ):
            return False
        if task.route is Route.CLOUD and (
            not (
                sensor_se_available
                and context.state.edge_gateway_available
                and context.state.ec_available
                and context.state.cloud_available
            )
        ):
            return False
        return task.route is not None or allow_local or allow_remote


    def choose_result_power(self, task: Task, context: ControllerContext) -> float:
        best_power = float(self._powers[-1])
        profile = self._control_profile(task, context)
        budget = self._sensor_energy_budget(task.sensor_id, context)
        pressure = (
            context.energy_queues[task.sensor_id].value / max(1e-12, budget)
            if self.strategy.use_lyapunov and self.strategy.use_energy_feedback
            else 0.0
        )
        candidates: list[tuple[float, _Prediction, float]] = []
        for power_value in self._powers:
            power = float(power_value)
            rate = self.se_rate_bits(power, context.state)
            if rate <= 0.0:
                continue
            active_fraction = min(1.0, task.remaining_bits / rate)
            delay = task.remaining_bits / max(
                1.0, rate * self.se_success_probability(power, context.state)
            )
            reliability, path_availability = self._path_factors(
                task, Route.LOCAL, context, power_w=power
            )
            availability = path_availability
            predicted_delay = task.age(context.state.slot) + delay
            deadline = max(1, task.deadline_slots)
            timeliness = (
                1.0
                if predicted_delay <= deadline
                else exp(-(predicted_delay - deadline) / deadline)
            )
            if predicted_delay > deadline:
                reliability *= exp(-4.0 * (predicted_delay - deadline) / deadline)
            prediction = _Prediction(
                delay=delay,
                timeliness=max(1e-06, timeliness),
                reliability=max(1e-06, reliability),
                availability=availability,
                energy=power * self.slot_seconds * active_fraction,
            )
            objective = self._aggregate_performance_penalty(
                task, prediction, context, profile, budget, delay_reference=1.0
            )
            objective += (
                profile.energy_weight
                * pressure
                * prediction.energy
                / max(1e-12, budget)
            )
            objective += self._trust_barrier(task, prediction, context)
            candidates.append((power, prediction, objective))
        if candidates:
            best_power = min(candidates, key=lambda item: item[2])[0]
        return best_power

    def task_priority(self, task: Task, slot: int, monitor: ODTEMonitor) -> float:
        # equation (36): request urgency before and after its absolute deadline.
        remaining = task.deadline_slot - slot
        urgency = 1.0 / (max(0, remaining) + 1.0)
        if remaining < 0:
            urgency += 1.0 + min(4.0, -remaining / max(1, task.deadline_slots))
        age_ratio = task.age(slot) / max(1, task.deadline_slots)
        debt = 0.0
        if self.strategy.use_odte:
            component_debts = monitor.component_debts(task.service_class)
            debt = sum(component_debts.values())
        # equation (37): source priority; heap ties use the earliest deadline.
        return (
            4.0 * urgency + age_ratio + float(self.ctrl["trust_weight"]) * log1p(debt)
        )
