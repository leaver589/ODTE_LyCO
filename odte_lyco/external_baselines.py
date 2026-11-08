"""Three-tier adapters; the copied upstream networks learn online from seeded initialization."""

from __future__ import annotations
from dataclasses import dataclass
import logging
import numpy as np
from .config import ExperimentConfig
from .controller import ControllerContext, LycoController
from .models import Route, SourceAction, StrategySpec, Task

BASELINE_STRATEGY = StrategySpec("external_baseline", False, False)
LYDROO_STRATEGY = StrategySpec("lydroo_3tier", True, False)


def _network_lydroo():
    from baselines import lydroo

    return lydroo


def _network_qeco():
    from baselines import qeco

    return qeco


class _AdaptedController(LycoController):

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__(config, BASELINE_STRATEGY)

    def _adapt_route(
        self,
        task: Task,
        desired_route: Route,
        context: ControllerContext,
        *,
        allow_local: bool,
        allow_remote: bool,
        workloads: dict[str, float] | None,
    ) -> SourceAction | None:
        if task.route is not None:
            desired_route = task.route
        evaluations = self._source_action_evaluations(
            task,
            context,
            allow_local=allow_local,
            allow_remote=allow_remote,
            workloads=workloads,
        )
        if not evaluations:
            return None
        matching = [
            action for action, _ in evaluations if action.route is desired_route
        ]
        candidates = matching or [action for action, _ in evaluations]
        return min(candidates, key=lambda action: action.objective)


class LyDROO3TierController(_AdaptedController):

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        sensors: int,
        seed: int,
        memory_size: int = 1024,
        delta: int = 32,
    ) -> None:
        LycoController.__init__(self, config, LYDROO_STRATEGY)
        self.sensors = int(sensors)
        self.seed = int(seed)
        self._module = _network_lydroo()
        np.random.seed(self.seed)
        self._module.torch.manual_seed(self.seed)
        self.memory = self._module.MemoryDNN(
            net=[self.sensors * 3, 256, 128, self.sensors],
            learning_rate=0.01,
            training_interval=20,
            batch_size=128,
            memory_size=int(memory_size),
        )
        self.delta = max(1, int(delta))
        self.k = self.sensors
        self._winner_indices: list[int] = []
        self._prepared_slot = -1
        self._slot_modes = np.zeros(self.sensors, dtype=int)
        self._planned_actions: dict[int, SourceAction] = {}

    @staticmethod
    def _sensor_link_feature(context: ControllerContext) -> float:
        state = context.state
        available = (
            state.sensor_se_available
            if state.sensor_se_available is not None
            else state.se_available
        )
        probability = (
            state.sensor_se_success_probability
            if state.sensor_se_success_probability is not None
            else state.se_success_probability
        )
        return float(np.clip(float(available) * probability, 0.0, 1.0))

    def _normalized_state(
        self,
        contexts: tuple[ControllerContext, ...],
        workloads_by_sensor: tuple[dict[str, float], ...],
    ) -> np.ndarray:
        queue_reference = max(
            1e-12, float(self.ctrl["dynamic_queue_reference_slots_per_sensor"])
        )
        debt_slots = max(
            1.0, float(self.ctrl["dynamic_energy_debt_reference_budget_slots"])
        )
        channels = np.asarray(
            [self._sensor_link_feature(context) for context in contexts], dtype=float
        )
        queues = np.asarray(
            [
                np.clip(
                    workloads_by_sensor[sensor_id].get(f"source_{sensor_id}", 0.0)
                    / queue_reference,
                    0.0,
                    10.0,
                )
                for sensor_id in range(self.sensors)
            ],
            dtype=float,
        )
        debts = np.asarray(
            [
                np.clip(
                    contexts[sensor_id].energy_queues[sensor_id].value
                    / max(
                        1e-12,
                        self._sensor_energy_budget(sensor_id, contexts[sensor_id])
                        * debt_slots,
                    ),
                    0.0,
                    10.0,
                )
                for sensor_id in range(self.sensors)
            ],
            dtype=float,
        )
        return np.concatenate((channels, queues, debts))

    def _domain_action(
        self,
        task: Task,
        context: ControllerContext,
        *,
        offload: bool,
        allow_local: bool,
        allow_remote: bool,
        workloads: dict[str, float],
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
        if task.route is not None:
            matching = [
                action for action, _ in evaluations if action.route is task.route
            ]
        elif offload:
            matching = [
                action for action, _ in evaluations if action.route is not Route.LOCAL
            ]
        else:
            matching = [
                action for action, _ in evaluations if action.route is Route.LOCAL
            ]
        candidates = matching or [action for action, _ in evaluations]
        return min(candidates, key=lambda action: action.objective)

    def _update_candidate_count(self, slot: int) -> None:
        if slot <= 0 or slot % self.delta != 0 or (not self._winner_indices):
            return
        recent = self._winner_indices[-self.delta :]
        max_k = max((index % max(1, self.k) for index in recent)) + 1
        self.k = min(max_k + 1, self.sensors)

    def prepare_source_slot(
        self,
        contexts: tuple[ControllerContext, ...],
        remote_available: tuple[bool, ...],
    ) -> None:
        slot = int(contexts[0].state.slot)
        if slot == self._prepared_slot:
            return
        self._prepared_slot = slot
        self._planned_actions = {}
        workloads_by_sensor = tuple(
            (self.queue_workloads(context) for context in contexts)
        )
        selected_tasks: list[Task | None] = []
        for sensor_id, context in enumerate(contexts):
            selected_tasks.append(
                self.select_source_task(
                    sensor_id,
                    context,
                    allow_local=True,
                    allow_remote=bool(remote_available[sensor_id]),
                )
            )
        active = [task is not None for task in selected_tasks]
        if not any(active):
            self._slot_modes = np.zeros(self.sensors, dtype=int)
            return
        nn_input = self._normalized_state(contexts, workloads_by_sensor)
        self._update_candidate_count(slot)
        raw_candidates = self.memory.decode(nn_input, max(1, self.k))
        candidates = [
            np.asarray(candidate, dtype=int).reshape(self.sensors)
            for candidate in raw_candidates
        ]
        if not candidates:
            candidates = [np.zeros(self.sensors, dtype=int)]
        # Each sensor has two domains; candidates reuse their identical resource evaluations.
        domain_actions: dict[tuple[int, bool], SourceAction | None] = {}
        scored: list[tuple[float, np.ndarray, dict[int, SourceAction]]] = []
        for candidate in candidates:
            repaired = candidate.copy()
            actions: dict[int, SourceAction] = {}
            objective = 0.0
            for sensor_id, task in enumerate(selected_tasks):
                if task is None:
                    repaired[sensor_id] = 0
                    continue
                if task.route is not None:
                    requested_offload = task.route is not Route.LOCAL
                else:
                    requested_offload = bool(candidate[sensor_id])
                key = (sensor_id, requested_offload)
                if key not in domain_actions:
                    domain_actions[key] = self._domain_action(
                        task, contexts[sensor_id], offload=requested_offload,
                        allow_local=True, allow_remote=bool(remote_available[sensor_id]),
                        workloads=workloads_by_sensor[sensor_id],
                    )
                action = domain_actions[key]
                if action is None:
                    objective += 1000000000000.0
                    repaired[sensor_id] = int(requested_offload)
                    continue
                repaired[sensor_id] = int(action.route is not Route.LOCAL)
                actions[sensor_id] = action
                objective += float(action.objective)
            scored.append((objective, repaired, actions))
        winner_index = min(range(len(scored)), key=lambda index: scored[index][0])
        _, winning_modes, winning_actions = scored[winner_index]
        self._winner_indices.append(winner_index)
        self._slot_modes = winning_modes
        self._planned_actions = winning_actions
        self.memory.encode(nn_input, winning_modes)

    def choose_source_action(
        self,
        task: Task,
        context: ControllerContext,
        *,
        allow_local: bool = True,
        allow_remote: bool = True,
        workloads: dict[str, float] | None = None,
    ) -> SourceAction | None:
        planned = self._planned_actions.get(task.sensor_id)
        if planned is not None and planned.task_id == task.task_id:
            if task.route is None or planned.route is task.route:
                if planned.route is Route.LOCAL and allow_local:
                    return planned
                if planned.route is not Route.LOCAL and allow_remote:
                    return planned
        workloads = (
            workloads if workloads is not None else self.queue_workloads(context)
        )
        offload = (
            task.route is not Route.LOCAL
            if task.route is not None
            else bool(self._slot_modes[task.sensor_id])
        )
        return self._domain_action(
            task,
            context,
            offload=offload,
            allow_local=allow_local,
            allow_remote=allow_remote,
            workloads=workloads,
        )


@dataclass
class _QECOFeedback:
    task: Task
    observation: np.ndarray
    history: np.ndarray
    action: int
    decision_slot: int
    energy_j: float = 0.0
    next_observation: np.ndarray | None = None
    next_history: np.ndarray | None = None
    completed_slot: int | None = None
    reward: float | None = None


class QECOController(_AdaptedController):
    preference_value = 0.75
    training_start_slots = 200
    training_interval = 10
    replay_memory_size = 500
    training_batch_size = 32
    target_update_interval = 200

    def __init__(
        self, config: ExperimentConfig, *, sensors: int, seed: int = 0
    ) -> None:
        super().__init__(config)
        self.sensors = int(sensors)
        self.seed = int(seed)
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        self._module = _network_qeco()
        self._module.tf.get_logger().setLevel("ERROR")
        self.edge_targets = 2
        self.n_actions = 1 + self.edge_targets
        self.n_features = 4 + self.edge_targets
        self.n_lstm_state = self.edge_targets
        self.n_lstm_step = 10
        self.n_lstm_hidden = 20
        self.hidden_width = 20
        self.agents = []
        agent_class = self._module.DuelingDoubleDeepQNetwork
        for sensor_id in range(self.sensors):
            agent = agent_class(
                self.n_actions,
                self.n_features,
                self.n_lstm_state,
                n_lstm_step=self.n_lstm_step,
                N_L1=self.hidden_width,
                N_lstm=self.n_lstm_hidden,
                seed=(self.seed * 1009 + sensor_id) % (2**31 - 1),
                memory_size=self.replay_memory_size,
                batch_size=self.training_batch_size,
                replace_target_iter=self.target_update_interval,
            )
            self.agents.append(agent)
        self._history_slot = -1
        self._training_slot = -1
        self._feedback: dict[int, _QECOFeedback] = {}
        self._awaiting_next_state: list[_QECOFeedback] = []

    def prepare_source_slot(
        self,
        contexts: tuple[ControllerContext, ...],
        remote_available: tuple[bool, ...],
    ) -> None:
        context = contexts[0]
        workloads = self.queue_workloads(context)
        self._update_history(context, workloads)
        remaining = []
        for feedback in self._awaiting_next_state:
            if feedback.decision_slot >= context.state.slot:
                remaining.append(feedback)
                continue
            feedback.next_observation = self._observation(feedback.task, workloads)
            feedback.next_history = np.asarray(
                self.agents[feedback.task.sensor_id].lstm_history, dtype=np.float32
            ).copy()
        self._awaiting_next_state = remaining
        self._store_ready_feedback()

    def observe_task_energy(self, task: Task, energy_j: float, slot: int) -> None:
        feedback = self._feedback.get(task.cohort_id)
        if feedback is not None and feedback.reward is None:
            feedback.energy_j += float(energy_j)

    def observe_task_completion(self, task: Task, slot: int) -> None:
        feedback = self._feedback.get(task.cohort_id)
        if feedback is not None and feedback.reward is None:
            feedback.completed_slot = int(slot)

    @classmethod
    def qos_reward(
        cls, *, delay_slots: int, deadline_slots: int, energy_j: float, unfinished: bool
    ) -> float:
        scaled_energy = float(energy_j) / 20.0 * 10.0
        cost = 2.0 * (
            cls.preference_value * min(delay_slots, deadline_slots)
            + (1.0 - cls.preference_value) * scaled_energy
        )
        return (0.0 if unfinished else deadline_slots * 4.0) - cost

    def _store_ready_feedback(self) -> None:
        ready = [
            key
            for key, feedback in self._feedback.items()
            if feedback.reward is not None and feedback.next_observation is not None
        ]
        for key in ready:
            feedback = self._feedback.pop(key)
            self.agents[feedback.task.sensor_id].store_transition(
                feedback.observation,
                feedback.history[-1],
                feedback.action,
                feedback.reward,
                feedback.next_observation,
                feedback.next_history[-1],
                history=feedback.history,
                next_history=feedback.next_history,
            )

    def finish_source_slot(self, context: ControllerContext) -> None:
        slot = int(context.state.slot)
        if slot == self._training_slot:
            return
        self._training_slot = slot
        for feedback in self._feedback.values():
            if feedback.reward is not None:
                continue
            completed = feedback.completed_slot
            if completed is None and slot < feedback.task.deadline_slot:
                continue
            timely = completed is not None and completed <= feedback.task.deadline_slot
            delay = feedback.task.age(completed if completed is not None else slot)
            feedback.reward = self.qos_reward(
                delay_slots=delay,
                deadline_slots=feedback.task.deadline_slots,
                energy_j=feedback.energy_j,
                unfinished=not timely,
            )
        self._store_ready_feedback()
        step = slot + 1
        if step <= self.training_start_slots or step % self.training_interval:
            return
        for agent in self.agents:
            if agent.memory_counter < self.training_batch_size:
                continue
            agent.learn()

    def _update_history(
        self, context: ControllerContext, workloads: dict[str, float]
    ) -> None:
        slot = int(context.state.slot)
        if slot == self._history_slot:
            return
        edge_state = np.asarray(
            [
                min(20.0, workloads.get("edge_compute", 0.0)),
                min(20.0, workloads.get("cloud_compute", 0.0)),
            ],
            dtype=float,
        )
        for agent in self.agents:
            agent.update_lstm(edge_state)
        self._history_slot = slot

    def _observation(self, task: Task, workloads: dict[str, float]) -> np.ndarray:
        task_size = float(np.clip(task.input_bits / 60000.0, 1.0, 7.0))
        local_wait = min(10.0, workloads.get(f"source_{task.sensor_id}", 0.0))
        transmit_wait = min(10.0, workloads.get(f"local_results_{task.sensor_id}", 0.0))
        edge_wait = min(10.0, workloads.get("edge_compute", 0.0))
        cloud_wait = min(10.0, workloads.get("cloud_compute", 0.0))
        return np.asarray(
            [
                task_size,
                local_wait,
                transmit_wait,
                edge_wait,
                cloud_wait,
                self.preference_value,
            ],
            dtype=float,
        )

    def choose_source_action(
        self,
        task: Task,
        context: ControllerContext,
        *,
        allow_local: bool = True,
        allow_remote: bool = True,
        workloads: dict[str, float] | None = None,
    ) -> SourceAction | None:
        workloads = (
            workloads if workloads is not None else self.queue_workloads(context)
        )
        if task.route is None:
            observation = self._observation(task, workloads)
            action = int(self.agents[task.sensor_id].choose_action(observation))
            desired_route = (Route.LOCAL, Route.EDGE, Route.CLOUD)[action]
        else:
            desired_route = task.route
        adapted = self._adapt_route(
            task,
            desired_route,
            context,
            allow_local=allow_local,
            allow_remote=allow_remote,
            workloads=workloads,
        )
        if (
            task.route is None
            and (adapted is not None)
            and (task.cohort_id not in self._feedback)
        ):
            feedback = _QECOFeedback(
                task=task,
                observation=observation.copy(),
                history=np.asarray(
                    self.agents[task.sensor_id].lstm_history, dtype=np.float32
                ).copy(),
                action=(Route.LOCAL, Route.EDGE, Route.CLOUD).index(adapted.route),
                decision_slot=int(context.state.slot),
            )
            self._feedback[task.cohort_id] = feedback
            self._awaiting_next_state.append(feedback)
        return adapted

    def close(self) -> None:
        for agent in self.agents:
            agent.sess.close()
