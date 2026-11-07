"""Shared stochastic sensor-edge-cloud simulator; no diagnostic file output."""

from __future__ import annotations
from dataclasses import dataclass, replace
from heapq import heappush
from typing import Deque
import numpy as np
from .config import ExperimentConfig, ScenarioConfig
from .controller import ControllerContext, LycoController
from .metrics import RunMetrics
from .models import (
    COORDINATED_LAYER_MODE_NAME,
    LAYER_CONTROL_MODES,
    MigrationEvent,
    MigrationTransfer,
    NetworkState,
    PayloadKind,
    Placement,
    Route,
    SourceAction,
    StrategySpec,
    Task,
)
from .odte import ODTEMonitor
from .placement import PlacementControlState, PlacementManager
from .queues import TaskQueues, VirtualQueue


def _inside_window(slot: int, total_slots: int, window: dict[str, float]) -> bool:
    start = int(float(window.get("start", 0.0)) * total_slots)
    end = int(float(window.get("end", 0.0)) * total_slots)
    return start <= slot < max(start, end)


@dataclass
class EnvironmentOverrides:
    slots: int | None = None
    sensors: int | None = None


@dataclass
class _DAGRequestRuntime:
    request_id: int
    tasks: dict[str, Task]
    completed_nodes: set[str]
    route: Route | None = None


class LycoEnvironment:

    def __init__(
        self,
        config: ExperimentConfig,
        scenario: ScenarioConfig,
        strategy: StrategySpec,
        seed: int,
        overrides: EnvironmentOverrides | None = None,
        layer_mode: str = COORDINATED_LAYER_MODE_NAME,
        weight_trace_window_slots: int | None = None,
        collect_timeline: bool = False,
    ) -> None:
        self.config = config
        self.scenario = scenario
        self.strategy = strategy
        self.seed = int(seed)
        self.learning_feedback = None
        self.layer_control = LAYER_CONTROL_MODES[layer_mode]
        seed_sequence = np.random.SeedSequence(self.seed)
        arrival_seed, size_seed, network_seed = seed_sequence.spawn(3)
        self.arrival_rng = np.random.default_rng(arrival_seed)
        self.size_rng = np.random.default_rng(size_seed)
        self.network_rng = np.random.default_rng(network_seed)
        overrides = overrides or EnvironmentOverrides()
        self.slots = int(
            config.simulation["slots"] if overrides.slots is None else overrides.slots
        )
        self.sensors = int(
            config.simulation["sensors"]
            if overrides.sensors is None
            else overrides.sensors
        )
        self.timeline_sample_stride = max(1, (self.slots + 2399) // 2400)
        self.queues = TaskQueues(self.sensors)
        self.monitor = ODTEMonitor(
            config.service_classes, epsilon=float(config.controller["odte_epsilon"])
        )
        self.energy_queues = [VirtualQueue() for _ in range(self.sensors)]
        self.sensor_energy_budget = float(config.energy["budget_j_per_slot"]) * float(
            scenario.sensor_energy_budget_multiplier
        )
        self.sensor_energy_budgets = [
            self.sensor_energy_budget for _ in range(self.sensors)
        ]
        self.sensor_se_losses = [scenario.se_loss for _ in range(self.sensors)]
        self.sensor_se_outage_windows: list[tuple[dict[str, float], ...]] = [
            scenario.se_outage_windows for _ in range(self.sensors)
        ]
        group_start = 0
        for group in scenario.sensor_groups:
            group_size = max(
                1, int(round(float(group["sensor_fraction"]) * self.sensors))
            )
            group_end = min(self.sensors, group_start + group_size)
            for sensor_id in range(group_start, group_end):
                self.sensor_se_losses[sensor_id] = float(
                    group.get("se_loss", scenario.se_loss)
                )
                self.sensor_energy_budgets[sensor_id] = (
                    self.sensor_energy_budget
                    * float(group.get("sensor_energy_budget_multiplier", 1.0))
                )
                extra_windows = tuple(
                    (
                        {str(key): value for key, value in item.items()}
                        for item in group.get("se_outage_windows", ())
                    )
                )
                self.sensor_se_outage_windows[sensor_id] = (
                    scenario.se_outage_windows + extra_windows
                )
            group_start = group_end
        self.mean_sensor_energy_budget = sum(self.sensor_energy_budgets) / max(
            1, self.sensors
        )
        self.placement = PlacementManager(
            config,
            use_energy_feedback=strategy.use_energy_feedback,
            initial_location=(
                Placement.EDGE
                if self.layer_control.placement_policy == "fixed_edge"
                else Placement(scenario.initial_twin_location)
            ),
            initial_locations=(
                {}
                if self.layer_control.placement_policy == "fixed_edge"
                else {
                    name: Placement(location)
                    for name, location in scenario.initial_twin_locations.items()
                }
            ),
        )
        self.fast_strategy = strategy
        self.controller = LycoController(config, strategy)
        self.metrics = RunMetrics(
            sensors=self.sensors,
            slots=self.slots,
            energy_budget_per_slot=self.mean_sensor_energy_budget,
            evaluation_event_start_slot=(
                None
                if scenario.evaluation_event_start is None
                else int(scenario.evaluation_event_start * self.slots)
            ),
            evaluation_event_end_slot=(
                None
                if scenario.evaluation_event_end is None
                else int(scenario.evaluation_event_end * self.slots)
            ),
        )
        self.next_task_id = 0
        self.next_request_id = 0
        self.dag_requests: dict[int, _DAGRequestRuntime] = {}
        self.frame_index = 0
        self.last_state: NetworkState | None = None
        self.nominal_arrival_rate = sum(
            (
                service.arrival_rate
                * (self.sensors if service.arrival_scope == "per_sensor" else 1)
                for service in config.service_classes.values()
            )
        )
        self.arrival_ewma = self.nominal_arrival_rate
        self.arrival_baseline_ewma = self.nominal_arrival_rate
        self.arrival_pressure = 0.0
        self.queue_slope_ewma = 0.0
        self.queue_growth_pressure = 0.0
        self.queue_pressure_ewma = 0.0
        self.queue_pressure = 0.0
        self.congestion_active = False
        self.congestion_hold_slots = 0
        self.previous_total_queue_workload: float | None = None
        self.last_workloads: dict[str, float] = {}
        self.last_service_workloads: dict[str, dict[str, float]] = {}
        self.weight_trace_window_slots = weight_trace_window_slots
        self._weight_trace: dict[int, dict[str, float]] = {}
        self._weight_trace_odte_base = self.controller._base_weight("trust_weight")
        self._weight_trace_energy_base = self.controller._base_weight(
            "energy_weight"
        ) * (
            float(config.controller["fusion_energy_weight_multiplier"])
            if strategy.use_lyapunov and strategy.use_odte
            else 1.0
        )
        self._window_action_templates: dict[
            tuple[int, str, Route | None], SourceAction
        ] = {}
        self._window_task_priorities: dict[int, float] = {}
        self.timeline_enabled = bool(collect_timeline)
        self._timeline_points = {name: [] for name in self.config.service_classes}
        self._timeline_events = {name: [] for name in self.config.service_classes}

    @staticmethod
    def _clip_pressure(value: float) -> float:
        return min(1.0, max(0.0, float(value)))

    def _timeline_edge_pressure(self, state: NetworkState) -> float:
        reference = max(
            1.0,
            float(self.config.controller["dynamic_queue_reference_slots_per_sensor"])
            * self.sensors,
        )
        edge_workload = self.last_workloads.get(
            "edge_compute", 0.0
        ) + self.last_workloads.get("pending_at_edge", 0.0)
        queue_pressure = self._clip_pressure(edge_workload / reference)
        capacity_pressure = self._clip_pressure(
            1.0 - float(state.edge_capacity_multiplier)
        )
        availability_pressure = (
            0.0
            if state.edge_compute_available and state.edge_gateway_available
            else 1.0
        )
        return max(queue_pressure, capacity_pressure, availability_pressure)

    def _timeline_communication_state(self, state: NetworkState) -> str:
        if not (
            state.ec_available
            and state.edge_gateway_available
            and state.cloud_available
        ):
            return "interrupted"
        baseline = max(0.0, 1.0 - float(self.scenario.ec_loss))
        degraded = (
            state.ec_success_probability + 1e-12 < baseline
            or float(state.ec_capacity_multiplier) < 1.0 - 1e-12
        )
        return "degraded" if degraded else "normal"

    def _record_timeline_point(self, state):
        if state.slot % self.timeline_sample_stride and state.slot != self.slots - 1:
            return
        condition = {
            "time_seconds": state.slot * self.config.slot_seconds,
            "task_pressure": max(
                self.arrival_pressure, self.queue_growth_pressure, self.queue_pressure
            ),
            "edge_pressure": self._timeline_edge_pressure(state),
            "communication_state": self._timeline_communication_state(state),
        }
        for name in self.config.service_classes:
            self._timeline_points[name].append(
                {
                    **condition,
                    "location": self.placement.location(name).value,
                    "migrating": self.placement.is_migrating(name, state.slot),
                }
            )

    def _record_timeline_event(self, event, *, phase, slot):
        if self.timeline_enabled:
            self._timeline_events[event.service_class].append(
                {
                    "phase": phase,
                    "time_seconds": slot * self.config.slot_seconds,
                    "source": event.source.value,
                    "target": event.target.value,
                }
            )

    def timeline_trace(self):
        return {
            name: {
                "points": self._timeline_points[name],
                "events": self._timeline_events[name],
            }
            for name in self.config.service_classes
        }

    def _slow_layer_due(self, slot: int) -> bool:
        return (slot + 1) % self.config.slow_layer_period_slots == 0

    def _record_weight_action(self, slot: int, action: SourceAction) -> None:
        if self.weight_trace_window_slots is None:
            return
        window = slot // self.weight_trace_window_slots
        bucket = self._weight_trace.setdefault(
            window, {"odte_weight_sum": 0.0, "energy_weight_sum": 0.0, "decisions": 0.0}
        )
        bucket["odte_weight_sum"] += float(action.odte_weight)
        bucket["energy_weight_sum"] += float(action.energy_weight)
        bucket["decisions"] += 1.0

    def weight_trace_rows(self, horizon_slots=None):
        horizon_slots = self.slots if horizon_slots is None else int(horizon_slots)
        return [
            (
                0.5
                * (
                    window * self.weight_trace_window_slots
                    + min(horizon_slots, (window + 1) * self.weight_trace_window_slots)
                )
                * self.config.slot_seconds
                / 60,
                value["odte_weight_sum"]
                / value["decisions"]
                / self._weight_trace_odte_base,
                value["energy_weight_sum"]
                / value["decisions"]
                / self._weight_trace_energy_base,
            )
            for window, value in sorted(self._weight_trace.items())
        ]

    def _network_state(self, slot: int) -> NetworkState:
        dynamic = {
            "se_loss": self.scenario.se_loss,
            "ec_loss": self.scenario.ec_loss,
            "edge_gateway_availability": self.scenario.edge_gateway_availability,
            "edge_compute_availability": self.scenario.edge_compute_availability,
            "cloud_availability": self.scenario.cloud_availability,
            "ec_capacity_multiplier": self.scenario.ec_capacity_multiplier,
            "edge_capacity_multiplier": self.scenario.edge_capacity_multiplier,
            "cloud_capacity_multiplier": self.scenario.cloud_capacity_multiplier,
        }
        for window in self.scenario.resource_windows:
            if not _inside_window(slot, self.slots, window):
                continue
            for key in dynamic:
                if key in window:
                    dynamic[key] = float(window[key])
        se_outage = any(
            (
                _inside_window(slot, self.slots, item)
                for item in self.scenario.se_outage_windows
            )
        )
        ec_outage = any(
            (
                _inside_window(slot, self.slots, item)
                for item in self.scenario.ec_outage_windows
            )
        )
        edge_gateway_forced_outage = any(
            (
                _inside_window(slot, self.slots, item)
                for item in self.scenario.edge_gateway_outage_windows
            )
        )
        edge_compute_forced_outage = any(
            (
                _inside_window(slot, self.slots, item)
                for item in self.scenario.edge_compute_outage_windows
            )
        )
        cloud_forced_outage = any(
            (
                _inside_window(slot, self.slots, item)
                for item in self.scenario.cloud_outage_windows
            )
        )
        edge_gateway_available = (
            not edge_gateway_forced_outage
            and self.network_rng.random() < dynamic["edge_gateway_availability"]
        )
        edge_compute_available = (
            not edge_compute_forced_outage
            and self.network_rng.random() < dynamic["edge_compute_availability"]
        )
        cloud_available = (
            not cloud_forced_outage
            and self.network_rng.random() < dynamic["cloud_availability"]
        )
        return NetworkState(
            slot=slot,
            se_success_probability=0.0 if se_outage else 1.0 - dynamic["se_loss"],
            ec_success_probability=0.0 if ec_outage else 1.0 - dynamic["ec_loss"],
            se_available=not se_outage,
            ec_available=not ec_outage,
            edge_gateway_available=edge_gateway_available,
            edge_compute_available=edge_compute_available,
            cloud_available=cloud_available,
            edge_capacity_multiplier=dynamic["edge_capacity_multiplier"],
            cloud_capacity_multiplier=dynamic["cloud_capacity_multiplier"],
            ec_capacity_multiplier=dynamic["ec_capacity_multiplier"],
        )

    def _sensor_network_state(
        self, state: NetworkState, sensor_id: int
    ) -> NetworkState:
        sensor_outage = any(
            (
                _inside_window(state.slot, self.slots, item)
                for item in self.sensor_se_outage_windows[sensor_id]
            )
        )
        return replace(
            state,
            sensor_se_success_probability=(
                0.0 if sensor_outage else 1.0 - self.sensor_se_losses[sensor_id]
            ),
            sensor_se_available=state.se_available and (not sensor_outage),
        )

    def _sensor_context(
        self, context: ControllerContext, sensor_id: int
    ) -> ControllerContext:
        return replace(
            context, state=self._sensor_network_state(context.state, sensor_id)
        )

    def _arrival_multiplier(self, service_class: str, slot: int) -> float:
        multiplier = self.scenario.arrival_multiplier
        for window in self.scenario.burst_windows:
            target_class = window.get("service_class")
            if target_class is not None and str(target_class) != service_class:
                continue
            if _inside_window(slot, self.slots, window):
                multiplier *= float(window.get("multiplier", 1.0))
        return multiplier

    def _workload_multiplier(self, service_class: str, slot: int) -> float:
        multiplier = 1.0
        for window in self.scenario.burst_windows:
            target_class = window.get("service_class")
            if target_class is not None and str(target_class) != service_class:
                continue
            if _inside_window(slot, self.slots, window):
                multiplier *= float(window.get("workload_multiplier", 1.0))
        return multiplier

    def _generate_arrivals(self, slot: int) -> int:
        size_sigma = float(self.config.simulation["task_size_sigma"])
        arrivals = 0
        for name, service in self.config.service_classes.items():
            rate = service.arrival_rate * self._arrival_multiplier(name, slot)
            workload_multiplier = self._workload_multiplier(name, slot)
            if service.arrival_scope == "per_sensor":
                sources = (
                    (sensor_id, int(self.arrival_rng.poisson(rate)))
                    for sensor_id in range(self.sensors)
                )
            else:
                count = int(self.arrival_rng.poisson(rate))
                sources = (
                    (int(self.arrival_rng.integers(0, self.sensors)), 1)
                    for _ in range(count)
                )
            for sensor_id, count in sources:
                for _ in range(count):
                    factor = float(
                        self.size_rng.lognormal(-0.5 * size_sigma**2, size_sigma)
                    )
                    input_bits = max(
                        1.0, service.input_bits * workload_multiplier * factor
                    )
                    cycles = input_bits * service.cycles_per_bit
                    output_bits = max(256.0, input_bits * service.output_ratio)
                    self._create_dag_request(
                        sensor_id=sensor_id,
                        service_class=name,
                        slot=slot,
                        input_bits=input_bits,
                        cpu_cycles=cycles,
                        output_bits=output_bits,
                    )
                    arrivals += 1
        return arrivals

    def _create_dag_request(
        self,
        *,
        sensor_id: int,
        service_class: str,
        slot: int,
        input_bits: float,
        cpu_cycles: float,
        output_bits: float,
    ) -> Task:
        service = self.config.service_classes[service_class]
        template = service.dag
        request_id = self.next_request_id
        self.next_request_id += 1
        tasks: dict[str, Task] = {}
        for node in template.nodes:
            is_source = node.node_id == template.source_node
            is_sink = node.node_id == template.sink_node
            dependency_input_bits = input_bits * template.incoming_data_ratio(
                node.node_id
            )
            dependency_output_bits = input_bits * template.outgoing_data_ratio(
                node.node_id
            )
            node_cycles = cpu_cycles * node.compute_ratio
            task = Task(
                task_id=self.next_task_id,
                request_id=request_id,
                dag_node_id=node.node_id,
                predecessor_ids=template.predecessors(node.node_id),
                successor_ids=template.successors(node.node_id),
                dependency_input_bits=dependency_input_bits,
                dependency_output_bits=dependency_output_bits,
                is_dag_sink=is_sink,
                ready_slot=slot if is_source else None,
                sensor_id=sensor_id,
                service_class=service_class,
                arrival_slot=slot,
                deadline_slots=service.deadline_slots,
                input_bits=input_bits if is_source else dependency_input_bits,
                cpu_cycles=node_cycles,
                output_bits=output_bits if is_sink else 0.0,
                remaining_bits=input_bits if is_source else 0.0,
                remaining_cycles=node_cycles,
            )
            self.next_task_id += 1
            tasks[node.node_id] = task
        runtime = _DAGRequestRuntime(
            request_id=request_id, tasks=tasks, completed_nodes=set()
        )
        self.dag_requests[request_id] = runtime
        root = tasks[template.source_node]
        self.queues.source[sensor_id].append(root)
        for task in tasks.values():
            if task is not root:
                self.queues.dependency_wait.append(task)
        self.monitor.record_arrival(root)
        self.metrics.record_admission()
        return root

    def _update_arrival_signal(self, arrivals: int) -> None:
        alpha = float(self.config.controller["arrival_ewma_alpha"])
        self.arrival_ewma = alpha * float(arrivals) + (1.0 - alpha) * self.arrival_ewma
        baseline_alpha = float(self.config.controller["arrival_baseline_ewma_alpha"])
        self.arrival_baseline_ewma = (
            baseline_alpha * float(arrivals)
            + (1.0 - baseline_alpha) * self.arrival_baseline_ewma
        )
        relative_excess = max(
            0.0,
            (self.arrival_ewma - self.arrival_baseline_ewma)
            / max(1e-12, self.arrival_baseline_ewma),
        )
        activation_excess = float(
            self.config.controller["criticality_arrival_activation_excess"]
        )
        excess_reference = max(
            1e-12, float(self.config.controller["criticality_arrival_excess_reference"])
        )
        self.arrival_pressure = min(
            1.0, max(0.0, (relative_excess - activation_excess) / excess_reference)
        )

    def _update_queue_growth_signal(self, workloads: dict[str, float]) -> None:
        total = sum(workloads.values())
        slope = (
            0.0
            if self.previous_total_queue_workload is None
            else total - self.previous_total_queue_workload
        )
        alpha = float(self.config.controller["queue_slope_ewma_alpha"])
        self.queue_slope_ewma = alpha * slope + (1.0 - alpha) * self.queue_slope_ewma
        reference = max(
            1e-12,
            float(self.config.controller["criticality_queue_slope_reference_slots"]),
        )
        self.queue_growth_pressure = min(
            1.0, max(0.0, self.queue_slope_ewma / reference)
        )
        if bool(self.config.controller["dynamic_weights_enabled"]):
            queue_reference = max(
                1e-12,
                float(
                    self.config.controller["dynamic_queue_reference_slots_per_sensor"]
                )
                * self.sensors,
            )
            raw_pressure = min(
                1.0,
                max(
                    total / queue_reference,
                    self.queue_growth_pressure,
                    self.arrival_pressure,
                ),
            )
            pressure_alpha = float(
                self.config.controller["dynamic_pressure_ewma_alpha"]
            )
            smoothed = (
                pressure_alpha * raw_pressure
                + (1.0 - pressure_alpha) * self.queue_pressure_ewma
            )
            max_step = float(self.config.controller["dynamic_pressure_max_step"])
            delta = min(max_step, max(-max_step, smoothed - self.queue_pressure_ewma))
            self.queue_pressure_ewma = min(
                1.0, max(0.0, self.queue_pressure_ewma + delta)
            )
            enter = float(self.config.controller["dynamic_congestion_enter"])
            exit_threshold = float(self.config.controller["dynamic_congestion_exit"])
            if not self.congestion_active and self.queue_pressure_ewma >= enter:
                self.congestion_active = True
                self.congestion_hold_slots = int(
                    self.config.controller["dynamic_congestion_min_hold_slots"]
                )
            elif self.congestion_active:
                if self.congestion_hold_slots > 0:
                    self.congestion_hold_slots -= 1
                elif self.queue_pressure_ewma <= exit_threshold:
                    self.congestion_active = False
            target_pressure = (
                self.queue_pressure_ewma if self.congestion_active else 0.0
            )
            pressure_delta = min(
                max_step, max(-max_step, target_pressure - self.queue_pressure)
            )
            self.queue_pressure = min(
                1.0, max(0.0, self.queue_pressure + pressure_delta)
            )
        self.previous_total_queue_workload = total
        self.last_workloads = dict(workloads)

    def _record_availability(self, state: NetworkState) -> None:
        for name in self.config.service_classes:
            location = self.placement.location(name)
            migrating = self.placement.is_migrating(name, state.slot)
            if location is Placement.EDGE:
                available = (
                    state.se_available
                    and state.edge_gateway_available
                    and state.edge_compute_available
                    and (not migrating)
                )
            else:
                available = (
                    state.se_available
                    and state.ec_available
                    and state.edge_gateway_available
                    and state.cloud_available
                    and (not migrating)
                )
            self.monitor.record_availability(name, available)

    def _pop_best(self, queue: Deque[Task], slot: int) -> Task:
        index = max(
            range(len(queue)),
            key=lambda idx: self.controller.task_priority(
                queue[idx], slot, self.monitor
            ),
        )
        queue.rotate(-index)
        task = queue.popleft()
        queue.rotate(index)
        return task

    def _link_success(
        self, probability: float, *, slot: int, channel_code: int, entity_code: int
    ) -> bool:
        keyed_seed = np.random.SeedSequence(
            [self.seed, int(slot), int(channel_code), int(entity_code)]
        )
        draw = float(np.random.default_rng(keyed_seed).random())
        return draw < probability

    def _lock_request_route(self, task: Task, route: Route) -> None:
        runtime = self.dag_requests[task.cohort_id]
        runtime.route = route
        for node_task in runtime.tasks.values():
            node_task.route = route

    def _release_dependency_ready_nodes(self, slot: int) -> None:
        for runtime in tuple(self.dag_requests.values()):
            if runtime.route is None:
                continue
            for task in runtime.tasks.values():
                if (
                    task.ready_slot is not None
                    or task.dag_node_id in runtime.completed_nodes
                ):
                    continue
                if not set(task.predecessor_ids).issubset(runtime.completed_nodes):
                    continue
                task.ready_slot = slot
                self.queues.dependency_wait.remove(task)
                if runtime.route is Route.LOCAL:
                    self.queues.source[task.sensor_id].append(task)
                elif runtime.route is Route.EDGE:
                    task.compute_location = Placement.EDGE
                    self.queues.edge_compute.append(task)
                else:
                    task.compute_location = Placement.CLOUD
                    self.queues.cloud_compute.append(task)

    def _complete_dag_node(self, task: Task, slot: int) -> None:
        runtime = self.dag_requests[task.cohort_id]
        task.completed_slot = slot
        runtime.completed_nodes.add(task.dag_node_id)

    def _handle_compute_completed(
        self, task: Task, location: Placement | None, state: NetworkState
    ) -> None:
        if not task.is_dag_sink:
            self._complete_dag_node(task, state.slot)
            return
        if location is None:
            task.remaining_bits = task.output_bits
            task.compute_location = None
            self.queues.local_results[task.sensor_id].append(task)
        else:
            self._handle_result_at(task, location, state)

    def _commit(self, task: Task, slot: int) -> None:
        task.completed_slot = slot
        self.monitor.record_commit(task, slot)
        self.metrics.record_commit(task)
        if self.learning_feedback is not None:
            self.learning_feedback.observe_task_completion(task, slot)
        self.dag_requests.pop(task.cohort_id, None)

    @staticmethod
    def _compute_service_available(location: Placement, state: NetworkState) -> bool:
        return (
            state.edge_compute_available
            if location is Placement.EDGE
            else state.cloud_available
        )

    def _handle_result_at(
        self, task: Task, location: Placement, state: NetworkState
    ) -> None:
        slot = state.slot
        if self.placement.is_migrating(task.service_class, slot):
            pending = (
                self.queues.pending_at_edge
                if location is Placement.EDGE
                else self.queues.pending_at_cloud
            )
            pending.append(task)
            return
        target = self.placement.location(task.service_class)
        if target is location:
            if self._compute_service_available(location, state):
                self._commit(task, slot)
            else:
                pending = (
                    self.queues.pending_at_edge
                    if location is Placement.EDGE
                    else self.queues.pending_at_cloud
                )
                pending.append(task)
            return
        task.payload_kind = PayloadKind.RESULT_FOR_TWIN
        task.remaining_bits = task.output_bits
        if location is Placement.EDGE:
            self.queues.edge_to_cloud.append(task)
        else:
            self.queues.cloud_to_edge.append(task)

    def _release_pending_results(self, state: NetworkState) -> None:
        for location, queue in (
            (Placement.EDGE, self.queues.pending_at_edge),
            (Placement.CLOUD, self.queues.pending_at_cloud),
        ):
            count = len(queue)
            for _ in range(count):
                task = queue.popleft()
                if self.placement.is_migrating(task.service_class, state.slot):
                    queue.append(task)
                else:
                    self._handle_result_at(task, location, state)

    def _process_compute_queue(
        self,
        queue: Deque[Task],
        location: Placement,
        capacity_cycles: float,
        state: NetworkState,
    ) -> None:
        remaining_capacity = max(0.0, capacity_cycles)
        while queue and remaining_capacity > 1.0:
            task = self._pop_best(queue, state.slot)
            used = min(remaining_capacity, task.remaining_cycles)
            task.remaining_cycles -= used
            remaining_capacity -= used
            if task.remaining_cycles <= 1.0:
                self._handle_compute_completed(task, location, state)
            else:
                queue.append(task)

    def _process_backhaul_queue(
        self,
        queue: Deque[Task],
        direction: str,
        state: NetworkState,
        slot: int,
        capacity_bits: float,
    ) -> float:
        if not state.ec_available or not queue:
            return max(0.0, capacity_bits)
        if direction == "edge_to_cloud" and (
            not (state.edge_gateway_available and state.cloud_available)
        ):
            return max(0.0, capacity_bits)
        if direction == "cloud_to_edge" and (
            not (state.cloud_available and state.edge_gateway_available)
        ):
            return max(0.0, capacity_bits)
        remaining_capacity = max(0.0, float(capacity_bits))
        direction_code = 21 if direction == "edge_to_cloud" else 22
        while queue and remaining_capacity > 1.0:
            task = self._pop_best(queue, slot)
            attempted = min(remaining_capacity, task.remaining_bits)
            success = self._link_success(
                state.ec_success_probability,
                slot=slot,
                channel_code=direction_code,
                entity_code=task.task_id,
            )
            sent = attempted if success else 0.0
            remaining_capacity -= attempted
            if not success:
                task.failed_attempts += 1
                queue.appendleft(task)
                break
            task.remaining_bits -= sent
            if task.remaining_bits > 1.0:
                queue.appendleft(task)
                break
            if task.payload_kind is PayloadKind.RAW_FOR_CLOUD:
                task.remaining_cycles = task.cpu_cycles
                task.compute_location = Placement.CLOUD
                self.queues.cloud_compute.append(task)
            elif direction == "edge_to_cloud":
                self._handle_result_at(task, Placement.CLOUD, state)
            else:
                self._handle_result_at(task, Placement.EDGE, state)
        return remaining_capacity

    def _process_migration_queue(
        self,
        queue: Deque[MigrationTransfer],
        direction: str,
        state: NetworkState,
        slot: int,
        capacity_bits: float,
    ) -> float:
        remaining_capacity = max(0.0, float(capacity_bits))
        if not state.ec_available or not queue:
            return remaining_capacity
        if direction == "edge_to_cloud" and (
            not (state.edge_gateway_available and state.cloud_available)
        ):
            return remaining_capacity
        if direction == "cloud_to_edge" and (
            not (state.cloud_available and state.edge_gateway_available)
        ):
            return remaining_capacity
        direction_code = 31 if direction == "edge_to_cloud" else 32
        service_codes = {
            name: index + 1
            for index, name in enumerate(sorted(self.config.service_classes))
        }
        while queue and remaining_capacity > 1.0:
            transfer = queue[0]
            attempted = min(remaining_capacity, transfer.remaining_bits)
            success = self._link_success(
                state.ec_success_probability,
                slot=slot,
                channel_code=direction_code,
                entity_code=service_codes[transfer.event.service_class],
            )
            delivered = attempted if success else 0.0
            transfer.attempted_bits += attempted
            transfer.delivered_bits += delivered
            transfer.failed_attempts += int(not success)
            remaining_capacity -= attempted
            if not success:
                break
            transfer.remaining_bits -= delivered
            if transfer.remaining_bits > 1.0:
                break
            queue.popleft()
            transfer.completed_slot = slot
            self.placement.complete_migration(transfer.event.service_class, slot)
            self._record_timeline_event(transfer.event, phase="completed", slot=slot)
        return remaining_capacity

    def _start_migration_transfer(self, event: MigrationEvent, *, slot: int) -> None:
        state_bits = float(self.config.placement["dt_state_bits"][event.service_class])
        transfer = MigrationTransfer(
            event=event,
            state_bits=state_bits,
            remaining_bits=state_bits,
            started_slot=slot,
        )
        queue = (
            self.queues.migration_edge_to_cloud
            if event.source is Placement.EDGE
            else self.queues.migration_cloud_to_edge
        )
        queue.append(transfer)
        self.metrics.record_migration(event)
        self._record_timeline_event(event, phase="started", slot=slot)

    def _process_local_results(
        self,
        state: NetworkState,
        context: ControllerContext,
        energy_this_slot: list[float],
    ) -> list[float]:
        radio_remaining = [1.0 for _ in range(self.sensors)]
        for sensor_id, queue in enumerate(self.queues.local_results):
            sensor_context = self._sensor_context(context, sensor_id)
            sensor_state = sensor_context.state
            if not sensor_state.sensor_se_available or not state.edge_gateway_available:
                continue
            while queue and radio_remaining[sensor_id] > 1e-09:
                task = self._pop_best(queue, state.slot)
                power = self.controller.choose_result_power(task, sensor_context)
                rate = self.controller.se_rate_bits(power, sensor_state)
                if rate <= 0.0:
                    queue.appendleft(task)
                    break
                active_fraction = min(
                    radio_remaining[sensor_id], task.remaining_bits / rate
                )
                attempted_bits = min(task.remaining_bits, rate * active_fraction)
                energy_this_slot[sensor_id] += (
                    power * self.config.slot_seconds * active_fraction
                )
                if self.learning_feedback is not None:
                    self.learning_feedback.observe_task_energy(
                        task,
                        power * self.config.slot_seconds * active_fraction,
                        state.slot,
                    )
                success = self._link_success(
                    self.controller.se_success_probability(power, sensor_state),
                    slot=state.slot,
                    channel_code=11,
                    entity_code=task.task_id,
                )
                sent = attempted_bits if success else 0.0
                radio_remaining[sensor_id] -= active_fraction
                if success:
                    task.remaining_bits -= sent
                else:
                    task.failed_attempts += 1
                    queue.appendleft(task)
                    radio_remaining[sensor_id] = 0.0
                    break
                if task.remaining_bits <= 1.0:
                    self._handle_result_at(task, Placement.EDGE, state)
                else:
                    queue.appendleft(task)
                    break
        return radio_remaining

    def _build_source_priority_heap(
        self, sensor_id: int, context: ControllerContext
    ) -> list[tuple[float, int, int, Task]]:
        if self.layer_control.task_decision_cadence == "slot":
            return self.controller.build_source_priority_heap(sensor_id, context)
        heap: list[tuple[float, int, int, Task]] = []
        for task in context.queues.source[sensor_id]:
            priority = self._window_task_priorities.setdefault(
                task.task_id,
                self.controller.task_priority(
                    task, context.state.slot, context.monitor
                ),
            )
            heappush(heap, (-priority, task.deadline_slot, task.task_id, task))
        return heap

    def _restore_source_priority(
        self,
        task: Task,
        context: ControllerContext,
        priority_heap: list[tuple[float, int, int, Task]],
    ) -> None:
        if self.layer_control.task_decision_cadence == "slot":
            self.controller.restore_source_priority(task, context, priority_heap)
            return
        priority = self._window_task_priorities.setdefault(
            task.task_id,
            self.controller.task_priority(task, context.state.slot, context.monitor),
        )
        heappush(priority_heap, (-priority, task.deadline_slot, task.task_id, task))

    @staticmethod
    def _held_action_is_feasible(
        action: SourceAction,
        task: Task,
        context: ControllerContext,
        *,
        allow_local: bool,
        allow_remote: bool,
    ) -> bool:
        if task.route is not None and action.route is not task.route:
            return False
        state = context.state
        sensor_available = (
            state.sensor_se_available
            if state.sensor_se_available is not None
            else state.se_available
        )
        if action.route is Route.LOCAL:
            return allow_local
        if (
            not allow_remote
            or not sensor_available
            or (not state.edge_gateway_available)
        ):
            return False
        if action.route is Route.EDGE:
            return state.edge_compute_available
        return state.ec_available and state.cloud_available

    def _choose_source_action(
        self, task, context, *, allow_local, allow_remote, workloads
    ):
        cache_key = (task.sensor_id, task.service_class, task.route)
        cached = self._window_action_templates.get(cache_key)
        if self.layer_control.task_decision_cadence == "window" and cached is not None:
            action = replace(cached, task_id=task.task_id)
            if self._held_action_is_feasible(
                action,
                task,
                context,
                allow_local=allow_local,
                allow_remote=allow_remote,
            ):
                return action
            return None
        action = self.controller.choose_source_action(
            task,
            context,
            allow_local=allow_local,
            allow_remote=allow_remote,
            workloads=workloads,
        )
        if action is not None and self.layer_control.task_decision_cadence == "window":
            self._window_action_templates[cache_key] = action
        return action

    def _process_source_actions(
        self,
        context: ControllerContext,
        radio_remaining: list[float],
        energy_this_slot: list[float],
    ) -> None:
        if self.strategy.name in ("lydroo_3tier", "external_baseline"):
            sensor_contexts = tuple(
                (
                    self._sensor_context(context, sensor_id)
                    for sensor_id in range(self.sensors)
                )
            )
            self.controller.prepare_source_slot(
                sensor_contexts,
                tuple((remaining > 1e-09 for remaining in radio_remaining)),
            )
        for sensor_id, queue in enumerate(self.queues.source):
            sensor_context = self._sensor_context(context, sensor_id)
            priority_heap = self._build_source_priority_heap(sensor_id, sensor_context)
            state = sensor_context.state
            cpu_remaining = 1.0
            while queue and (
                cpu_remaining > 1e-09 or radio_remaining[sensor_id] > 1e-09
            ):
                allow_local = cpu_remaining > 1e-09
                allow_remote = radio_remaining[sensor_id] > 1e-09
                task = self.controller.select_source_task(
                    sensor_id,
                    sensor_context,
                    allow_local=allow_local,
                    allow_remote=allow_remote,
                    priority_heap=priority_heap,
                )
                if task is None:
                    break
                workloads = self.controller.queue_workloads(sensor_context)
                action = self._choose_source_action(
                    task,
                    sensor_context,
                    allow_local=allow_local,
                    allow_remote=allow_remote,
                    workloads=workloads,
                )
                if action is None:
                    break
                self._record_weight_action(state.slot, action)
                if action.route is Route.LOCAL:
                    if task.first_service_slot is None:
                        task.mark_started(state.slot, Route.LOCAL)
                        self._lock_request_route(task, Route.LOCAL)
                    frequency = action.cpu_frequency_hz
                    service_cycles = frequency * self.config.slot_seconds
                    active_fraction = min(
                        cpu_remaining, task.remaining_cycles / max(1.0, service_cycles)
                    )
                    cycles = min(
                        task.remaining_cycles, service_cycles * active_fraction
                    )
                    energy_this_slot[sensor_id] += (
                        float(self.config.energy["cpu_coefficient"])
                        * frequency**3
                        * self.config.slot_seconds
                        * active_fraction
                    )
                    if self.learning_feedback is not None:
                        self.learning_feedback.observe_task_energy(
                            task,
                            float(self.config.energy["cpu_coefficient"])
                            * frequency**3
                            * self.config.slot_seconds
                            * active_fraction,
                            state.slot,
                        )
                    task.remaining_cycles -= cycles
                    cpu_remaining -= active_fraction
                    if task.remaining_cycles <= 1.0:
                        queue.remove(task)
                        self._handle_compute_completed(task, None, state)
                    else:
                        cpu_remaining = 0.0
                        self._restore_source_priority(
                            task, sensor_context, priority_heap
                        )
                    continue
                power = action.transmit_power_w
                rate = self.controller.se_rate_bits(power, state)
                if rate <= 0.0:
                    radio_remaining[sensor_id] = 0.0
                    self._restore_source_priority(task, sensor_context, priority_heap)
                    continue
                if task.first_service_slot is None:
                    task.mark_started(state.slot, action.route)
                    self._lock_request_route(task, action.route)
                active_fraction = min(
                    radio_remaining[sensor_id], task.remaining_bits / rate
                )
                attempted_bits = min(task.remaining_bits, rate * active_fraction)
                energy_this_slot[sensor_id] += (
                    power * self.config.slot_seconds * active_fraction
                )
                if self.learning_feedback is not None:
                    self.learning_feedback.observe_task_energy(
                        task,
                        power * self.config.slot_seconds * active_fraction,
                        state.slot,
                    )
                success = self._link_success(
                    self.controller.se_success_probability(power, state),
                    slot=state.slot,
                    channel_code=12,
                    entity_code=task.task_id,
                )
                sent = attempted_bits if success else 0.0
                radio_remaining[sensor_id] -= active_fraction
                if success:
                    task.remaining_bits -= sent
                else:
                    task.failed_attempts += 1
                    radio_remaining[sensor_id] = 0.0
                    self._restore_source_priority(task, sensor_context, priority_heap)
                    continue
                if task.remaining_bits > 1.0:
                    radio_remaining[sensor_id] = 0.0
                    self._restore_source_priority(task, sensor_context, priority_heap)
                    continue
                queue.remove(task)
                task.remaining_cycles = task.cpu_cycles
                if action.route is Route.EDGE:
                    task.compute_location = Placement.EDGE
                    self.queues.edge_compute.append(task)
                else:
                    task.payload_kind = PayloadKind.RAW_FOR_CLOUD
                    task.remaining_bits = task.input_bits
                    self.queues.edge_to_cloud.append(task)

    def _placement_control_state(self) -> PlacementControlState:
        has_state_observation = (
            self.layer_control.share_fast_state
            or self.layer_control.placement_policy == "slow_adaptive"
        )
        if has_state_observation:
            return PlacementControlState(
                queue_workloads=self.last_workloads,
                mean_sensor_energy_debt=sum(
                    (queue.value for queue in self.energy_queues)
                )
                / max(1, self.sensors),
                service_queue_workloads=self.last_service_workloads,
            )
        return PlacementControlState(queue_workloads={}, mean_sensor_energy_debt=0.0)

    def _update_fast_placement(self, state: NetworkState) -> None:
        if self.layer_control.placement_policy != "fast_adaptive":
            return
        if any(
            (
                self.placement.is_migrating(name, state.slot)
                for name in self.config.service_classes
            )
        ):
            return
        control = PlacementControlState(
            queue_workloads=self.last_workloads,
            mean_sensor_energy_debt=sum((queue.value for queue in self.energy_queues))
            / max(1, self.sensors),
            service_queue_workloads=self.last_service_workloads,
        )
        window_alpha = float(self.config.placement["score_ema_alpha"])
        slot_alpha = 1.0 - (1.0 - window_alpha) ** (
            1.0 / max(1, self.config.monitor_window)
        )
        events = self.placement.update(
            frame_index=state.slot,
            slot=state.slot,
            edge_load=self.last_workloads.get("edge_compute", 0.0),
            cloud_load=self.last_workloads.get("cloud_compute", 0.0),
            control=control,
            snapshots=[],
            monitor=self.monitor,
            use_lyapunov=self.strategy.use_lyapunov,
            force_evaluation=True,
            min_residence_decisions=int(self.config.placement["min_residence_frames"])
            * self.config.monitor_window,
            score_ema_alpha=slot_alpha,
        )
        for event in events:
            self._start_migration_transfer(event, slot=state.slot)

    def _finish_frame(self, state: NetworkState) -> None:
        snapshots = self.monitor.finalize_frame(state.slot)
        self.metrics.add_snapshots(snapshots, slot=state.slot)
        if not self._slow_layer_due(state.slot):
            return
        control = self._placement_control_state()
        has_state_observation = (
            self.layer_control.share_fast_state
            or self.layer_control.placement_policy == "slow_adaptive"
        )
        edge_load = (
            self.last_workloads.get("edge_compute", 0.0)
            if has_state_observation
            else 0.0
        )
        cloud_load = (
            self.last_workloads.get("cloud_compute", 0.0)
            if has_state_observation
            else 0.0
        )
        events = []
        if (
            self.layer_control.placement_policy in {"slow_adaptive", "slow_coordinated"}
            and self.strategy.use_odte
        ):
            events = self.placement.update(
                frame_index=self.frame_index,
                slot=state.slot + 1,
                edge_load=edge_load,
                cloud_load=cloud_load,
                control=control,
                snapshots=snapshots,
                monitor=self.monitor,
                use_lyapunov=self.strategy.use_lyapunov and has_state_observation,
            )
        for event in events:
            self._start_migration_transfer(event, slot=state.slot + 1)
        self.frame_index += 1

    def step(self, slot: int) -> None:
        if slot % self.config.slow_layer_period_slots == 0:
            self._window_action_templates.clear()
            self._window_task_priorities.clear()
        state = self._network_state(slot)
        self.last_state = state
        self.placement.observe(state)
        self._update_fast_placement(state)
        arrivals = self._generate_arrivals(slot)
        self._release_dependency_ready_nodes(slot)
        self._update_arrival_signal(arrivals)
        self._record_availability(state)
        self._release_pending_results(state)
        edge_capacity = (
            float(self.config.resources["edge_cpu_hz"])
            * self.config.slot_seconds
            * state.edge_capacity_multiplier
            if state.edge_compute_available
            else 0.0
        )
        cloud_capacity = (
            float(self.config.resources["cloud_cpu_hz"])
            * self.config.slot_seconds
            * state.cloud_capacity_multiplier
            if state.cloud_available
            else 0.0
        )
        self._process_compute_queue(
            self.queues.edge_compute, Placement.EDGE, edge_capacity, state
        )
        self._process_compute_queue(
            self.queues.cloud_compute, Placement.CLOUD, cloud_capacity, state
        )
        direction_capacity = (
            float(self.config.resources["ec_rate_bits_per_slot"])
            * state.ec_capacity_multiplier
        )
        edge_to_cloud_remaining = self._process_migration_queue(
            self.queues.migration_edge_to_cloud,
            "edge_to_cloud",
            state,
            slot,
            direction_capacity,
        )
        self._process_backhaul_queue(
            self.queues.edge_to_cloud,
            "edge_to_cloud",
            state,
            slot,
            edge_to_cloud_remaining,
        )
        cloud_to_edge_remaining = self._process_migration_queue(
            self.queues.migration_cloud_to_edge,
            "cloud_to_edge",
            state,
            slot,
            direction_capacity,
        )
        self._process_backhaul_queue(
            self.queues.cloud_to_edge,
            "cloud_to_edge",
            state,
            slot,
            cloud_to_edge_remaining,
        )
        energy_this_slot = [
            float(self.config.energy["sampling_j_per_slot"])
            for _ in range(self.sensors)
        ]
        context = ControllerContext(
            state=state,
            queues=self.queues,
            energy_queues=self.energy_queues,
            monitor=self.monitor,
            placement=self.placement,
            energy_budget_per_slot=self.sensor_energy_budget,
            sensor_energy_budgets=self.sensor_energy_budgets,
            arrival_pressure=self.arrival_pressure,
            queue_growth_pressure=self.queue_growth_pressure,
            queue_pressure=self.queue_pressure,
        )
        radio_remaining = self._process_local_results(state, context, energy_this_slot)
        self._process_source_actions(context, radio_remaining, energy_this_slot)
        for sensor_id, consumed in enumerate(energy_this_slot):
            self.metrics.record_energy(consumed, sensor_id=sensor_id, slot=slot)
            budget = self.sensor_energy_budgets[sensor_id]
            self.energy_queues[sensor_id].update(arrival=consumed, service=budget)
        workloads = self.controller.queue_workloads(context)
        self.last_service_workloads = {
            name: self.controller.queue_workloads(context, service_class=name)
            for name in self.config.service_classes
        }
        self.metrics.record_queue(
            sum(workloads.values()), max(workloads.values(), default=0.0)
        )
        self._update_queue_growth_signal(workloads)
        if self.learning_feedback is not None:
            self.learning_feedback.finish_source_slot(context)
        if (slot + 1) % self.config.monitor_window == 0:
            self._finish_frame(state)
        if self.timeline_enabled:
            self._record_timeline_point(state)

    def run(self):
        for slot in range(self.slots):
            self.step(slot)
        return self.summarize(self.slots)

    def summarize(self, completed_slots):
        self.metrics.slots = completed_slots
        debt = sum((queue.value for queue in self.energy_queues)) / self.sensors
        return self.metrics.finalize(debt)
