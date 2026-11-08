"""Paper equations, queue semantics, migration timing and statistical units."""

import csv
from dataclasses import replace
from math import exp, log, log2, sqrt
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments import ROOT, SCENARIOS
from odte_lyco.config import load_config
from odte_lyco.controller import ControllerContext
from odte_lyco.environment import EnvironmentOverrides, LycoEnvironment
from odte_lyco.models import Placement, Route, STRATEGIES, Task
from odte_lyco.odte import ODTEMonitor
from odte_lyco.placement import PlacementControlState
from odte_lyco.queues import VirtualQueue
from tables import FIELDS, main_cells, mean_ci, seed_values, table_main


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "config.yaml")

    def environment(self, strategy="base", mode="coordinated"):
        return LycoEnvironment(self.config, self.config.scenarios[SCENARIOS["S1"]],
                               STRATEGIES[strategy], 0, EnvironmentOverrides(slots=540, sensors=8),
                               layer_mode=mode)

    def context(self, env):
        state = replace(env._network_state(0), se_success_probability=.9,
                        ec_success_probability=.8, se_available=True, ec_available=True,
                        edge_gateway_available=True, edge_compute_available=True, cloud_available=True)
        return ControllerContext(state, env.queues, env.energy_queues, env.monitor, env.placement,
                                 sensor_energy_budgets=env.sensor_energy_budgets)

    @staticmethod
    def task(task_id=0, arrival=0, deadline=12):
        return Task(task_id, 0, "status", arrival, deadline,
                    60000., 30000000., 6000., 60000., 30000000.)

    def test_student_t_interval(self):
        mean, half = mean_ci([1., 2., 3.])
        self.assertAlmostEqual(mean, 2.)
        self.assertAlmostEqual(half, 2.484137711750329, places=9)
        self.assertEqual(mean_ci([2.]), (2., None))

    def test_macro_interval_pairs_scenarios_within_each_seed(self):
        # Each seed's scenario-average is 1 although scenario means vary.
        results = [
            {"scene": scene, "method": "base", "seed": seed,
             "metrics": {"mean_odte": value}}
            for scene, values in zip(SCENARIOS, ([0., 1., 2.], [2., 1., 0.], [1., 1., 1.]))
            for seed, value in enumerate(values)
        ]
        self.assertEqual(mean_ci(seed_values(results, "Macro", "base", "mean_odte")), (1., 0.))

    def test_no_y_removes_feedback_but_keeps_energy_accounting(self):
        config = load_config(ROOT / "config.yaml")
        env = LycoEnvironment(config, config.scenarios[SCENARIOS["S1"]],
                              STRATEGIES["no_y"], 0, EnvironmentOverrides(slots=540, sensors=8))
        task = Task(0, 0, "status", 0, 12, 60000., 30000000., 6000., 60000., 30000000.)
        context = ControllerContext(
            state=env._network_state(0), queues=env.queues, energy_queues=env.energy_queues,
            monitor=env.monitor, placement=env.placement,
            sensor_energy_budgets=env.sensor_energy_budgets,
            energy_budget_per_slot=env.sensor_energy_budget,
        )
        def scores():
            return [a.objective for a, _ in env.controller._source_action_evaluations(task, context)]
        zero_debt_scores = scores()
        self.assertTrue(zero_debt_scores)
        zero_debt_power = env.controller.choose_result_power(task, context)
        for queue in env.energy_queues:
            queue.value = 5.
        self.assertEqual(scores(), zero_debt_scores)
        self.assertEqual(env.controller.choose_result_power(task, context), zero_debt_power)
        def utility(debt):
            return env.placement._joint_utility(odte=.8, queue_workload=2.,
                trust_debt=.1, mean_sensor_energy_debt=debt, use_lyapunov=True)
        self.assertEqual(utility(0.), utility(50.))
        env.step(0)
        self.assertGreater(env.metrics.total_energy, 0.)
        self.assertTrue(all(q.value > 0. for q in env.energy_queues))

    def test_wireless_capacity_and_power_success(self):
        env = self.environment()
        context = self.context(env)
        p = .1
        expected = 400000 * .1 * log2(1 + 15 * p / .25)
        self.assertAlmostEqual(env.controller.se_rate_bits(p, context.state), expected)
        self.assertAlmostEqual(env.controller.se_success_probability(p, context.state),
                               .9 * (.55 + .45 * sqrt(p / .25)))
        failed = replace(context.state, se_available=False)
        self.assertEqual(env.controller.se_rate_bits(p, failed), 0)
        self.assertGreater(env.controller._expected_se_capacity(failed), 0)

    def test_cloud_path_counts_return_hop_and_current_authority(self):
        env = self.environment()
        context = self.context(env)
        task = self.task()
        self.assertAlmostEqual(env.controller._path_factors(task, Route.CLOUD, context)[0], .9 * .8**2)
        env.placement.locations["status"] = Placement.CLOUD
        self.assertAlmostEqual(env.controller._path_factors(task, Route.CLOUD, context)[0], .9 * .8)
        failed_edge = replace(context, state=replace(context.state, edge_compute_available=False))
        self.assertEqual(env.controller._path_factors(task, Route.CLOUD, failed_edge)[1], 1.)

    def test_cloud_latency_forwards_full_input_after_partial_sensor_upload(self):
        env = self.environment()
        context = self.context(env)
        task = self.task()
        task.remaining_bits /= 2
        predicted = env.controller._predict(task, Route.CLOUD, 0, .25, context)
        expected = (1 + task.remaining_bits / env.controller.se_rate_bits(.25, context.state)
                    + task.input_bits / env.controller._expected_ec_capacity(context.state)
                    + task.cpu_cycles / env.controller._cloud_capacity(context.state)
                    + task.output_bits / env.controller._expected_ec_capacity(context.state))
        self.assertAlmostEqual(predicted.delay, expected)

    def test_candidates_and_route_lock(self):
        env = self.environment()
        context = self.context(env)
        task = self.task()
        self.assertEqual(len(env.controller._source_action_evaluations(task, context)), 27)
        task.route = Route.LOCAL
        actions = env.controller._source_action_evaluations(task, context)
        self.assertEqual(len(actions), 9)
        self.assertTrue(all(a.route is Route.LOCAL for a, _ in actions))

    def test_priority_heap_resolves_equal_scores_by_earlier_deadline(self):
        env = self.environment()
        context = self.context(env)
        early, late = self.task(0, deadline=12), self.task(1, deadline=24)
        env.queues.source[0].extend([late, early])
        with patch.object(env.controller, "task_priority", return_value=1.):
            heap = env.controller.build_source_priority_heap(0, context)
            chosen = env.controller.select_source_task(0, context, priority_heap=heap)
        self.assertIs(chosen, early)

    def test_matured_odte_cohort_settles_once_with_unsubmitted_denominator(self):
        monitor = ODTEMonitor(self.config.service_classes)
        tasks = [self.task(i) for i in range(3)]
        for task in tasks:
            monitor.record_arrival(task)
        monitor.record_commit(tasks[0], 11)
        monitor.record_commit(tasks[1], 13)
        snapshot = monitor.finalize_frame(17)[0]
        self.assertAlmostEqual(snapshot.timeliness, (1 + exp(-2 / 12)) / 2)
        self.assertAlmostEqual(snapshot.reliability, 1 / 3)
        self.assertAlmostEqual(snapshot.odte, snapshot.timeliness / 3)
        monitor.record_commit(tasks[2], 18)
        next_snapshot = monitor.finalize_frame(23)[0]
        self.assertEqual(next_snapshot.timeliness, 1.)
        self.assertEqual(next_snapshot.reliability, 1.)
        self.assertEqual(len(monitor.cohorts), 0)

    def test_log_loss_queues_and_component_budget(self):
        monitor = ODTEMonitor(self.config.service_classes)
        monitor.record_arrival(self.task())
        snapshot = monitor.finalize_frame(11)[0]
        self.assertEqual(snapshot.odte, 0.)
        self.assertAlmostEqual(snapshot.timeliness_loss, -log(1e-6))
        budget = -log(.85) / 3
        self.assertAlmostEqual(monitor.debt("status", "timeliness"), -log(1e-6) - budget)

    def test_virtual_queue_telescoping_energy_bound(self):
        queue = VirtualQueue()
        energies, budget = [.1, .0, .03, .02], .025
        for energy in energies:
            queue.update(energy, budget)
        self.assertLessEqual(sum(energies) / len(energies), budget + queue.value / len(energies) + 1e-12)

    def test_dag_route_lock_next_slot_release_and_final_submission(self):
        env = self.environment()
        root = env._create_dag_request(sensor_id=0, service_class="diagnostic", slot=0,
                                       input_bits=180000., cpu_cycles=270000000., output_bits=14400.)
        env._lock_request_route(root, Route.LOCAL)
        env.queues.source[0].remove(root)
        runtime = env.dag_requests[root.cohort_id]
        env._handle_compute_completed(root, None, self.context(env).state)
        self.assertEqual(env.metrics.completed, 0)
        env._release_dependency_ready_nodes(1)
        successor = env.queues.source[0][0]
        self.assertEqual(successor.ready_slot, 1)
        self.assertIs(successor.route, Route.LOCAL)
        self.assertAlmostEqual(sum(t.cpu_cycles for t in runtime.tasks.values()), 270000000.)
        sink = next(t for t in runtime.tasks.values() if t.is_dag_sink)
        env.placement.migration_until_slot["diagnostic"] = 10
        env._handle_result_at(sink, Placement.EDGE, self.context(env).state)
        self.assertEqual(env.metrics.completed, 0)
        context = replace(self.context(env), state=replace(self.context(env).state, slot=10))
        env._release_pending_results(context.state)
        self.assertEqual(env.metrics.completed, 1)
        self.assertEqual(sink.delay(), 11)

    def test_failed_source_transmission_consumes_time_energy_and_retains_data(self):
        env = self.environment()
        root = env._create_dag_request(sensor_id=0, service_class="status", slot=0,
                                       input_bits=60000., cpu_cycles=30000000., output_bits=6000.)
        root.route = Route.EDGE
        context = self.context(env)
        radio, energy = [1.] * 8, [0.] * 8
        with patch.object(env, "_link_success", return_value=False):
            env._process_source_actions(context, radio, energy)
        self.assertGreater(energy[0], 0.)
        self.assertEqual(radio[0], 0.)
        self.assertEqual(root.remaining_bits, 60000.)
        self.assertEqual(root.failed_attempts, 1)

    def test_all_fast_observation_accumulates_through_migration_pause(self):
        env = self.environment(mode="all_fast")
        env.placement.migration_until_slot["status"] = 3
        observed = []
        def capture(**kwargs):
            observed.append(env.placement.observation.samples)
            return []
        with patch.object(env.placement, "update", side_effect=capture):
            env.step(0)
            env.step(1)
            env.step(2)
            self.assertEqual(env.placement.observation.samples, 3)
            env.step(3)
        self.assertEqual(observed, [4])

    def test_placement_skips_classes_during_state_transfer(self):
        env = self.environment()
        manager = env.placement
        manager.pending_events.update({name: None for name in self.config.service_classes})
        manager.observe(self.context(env).state)
        estimate = SimpleNamespace(odte=.5, availability=.5, queue_workload=0.)
        with patch.object(manager, "_placement_estimate", return_value=estimate), \
             patch.object(manager, "_joint_utility", side_effect=AssertionError("Migrating class evaluated")):
            events = manager.update(frame_index=20, slot=60, edge_load=0, cloud_load=0,
                                    control=PlacementControlState({}, 0.), snapshots=[],
                                    monitor=env.monitor, use_lyapunov=True, force_evaluation=True)
        self.assertEqual(events, [])
        self.assertEqual(manager.observation.samples, 0)

    def test_classes_in_post_transfer_pause_enter_candidate_evaluation(self):
        env = self.environment()
        manager = env.placement
        manager.migration_until_slot.update({name: 100 for name in self.config.service_classes})
        manager.observe(self.context(env).state)
        estimate = SimpleNamespace(odte=.5, availability=.5, queue_workload=0.)
        with patch.object(manager, "_placement_estimate", return_value=estimate), \
             patch.object(manager, "_joint_utility", wraps=manager._joint_utility) as utility:
            manager.update(frame_index=20, slot=60, edge_load=0, cloud_load=0,
                           control=PlacementControlState({}, 0.), snapshots=[],
                           monitor=env.monitor, use_lyapunov=True, force_evaluation=True)
        self.assertEqual(utility.call_count, 6)

    def test_migration_trace_preserves_single_slot_peak_and_outage(self):
        env = self.environment()
        state = self.context(env).state
        env._record_timeline_point(state)
        env.arrival_pressure = 1.
        env._record_timeline_point(replace(state, slot=1, ec_available=False))
        env.arrival_pressure = 0.
        env._record_timeline_point(replace(state, slot=2))
        points = env.timeline_trace()["status"]["points"]
        self.assertEqual([p["time_seconds"] for p in points], [0., .1, .2])
        self.assertEqual(points[1]["task_pressure"], 1.)
        self.assertEqual(points[1]["communication_state"], "interrupted")
        self.assertEqual(points[2]["communication_state"], "degraded")

    def test_smooth_aggregate_matches_weighted_logsumexp(self):
        env = self.environment("odte_lyco")
        context = self.context(env)
        task = self.task()
        prediction = env.controller._predict(task, Route.LOCAL, 2e8, 0., context)
        profile = env.controller._control_profile(task, context)
        budget = env.sensor_energy_budgets[0]
        components = env.controller._performance_components(task, prediction, context, profile,
                                                            budget, delay_reference=12)
        weights, losses = np.asarray(components).T
        values = 1 - np.exp(-losses)
        expected = weights.sum() * .03 * log(np.sum(weights / weights.sum() * np.exp(values / .03)))
        actual = env.controller._aggregate_performance_penalty(task, prediction, context, profile,
                                                               budget, delay_reference=12)
        self.assertAlmostEqual(actual, expected)

    def test_dynamic_weights_and_capacity_gate_follow_paper_constants(self):
        env = self.environment("dynamic")
        task = self.task()
        env.queues.source[0].append(task)
        for name, debt in zip(("timeliness", "reliability", "availability"), (.6, .3, .1)):
            env.monitor.trust_queues["status"][name].value = debt
        env.energy_queues[0].value = 150 * env.sensor_energy_budgets[0]
        context = self.context(env)
        context = replace(context, state=replace(context.state, slot=5),
                          arrival_pressure=.7, queue_growth_pressure=.4, queue_pressure=.2)
        workloads = env.controller.queue_workloads(context)
        criticality = max(.4 * .5 + .2 * (.3 / 1.5) + .25 * .5, .75 * .7)
        self.assertAlmostEqual(env.controller.criticality(task, context, workloads), criticality)
        profile = env.controller._control_profile(task, context, workloads)
        self.assertAlmostEqual(profile.v_delay, .4 * min(2, 1 + 1.15 * criticality + .5 * .2))
        self.assertAlmostEqual(profile.resource_cost_weight, .02 * (1 - .6 * criticality))
        self.assertAlmostEqual(profile.trust_weight, .7 * (1 + .95 * criticality))
        self.assertAlmostEqual(profile.energy_weight,
                               .055 * max(.25, 1 - .95 * criticality) * (1 + .3 * .5 * .8) * .2)
        gate = .55 + .45 * .9
        self.assertAlmostEqual(env.controller._route_capacity_gate(task, Route.LOCAL, context, workloads), gate)
        self.assertAlmostEqual(env.controller._effective_queue_weight(task, Route.LOCAL, context, profile, workloads),
                               1 + 3 * .2 * .2 * .775 * gate)

    def test_heterogeneous_sensor_budget_energy_ratio(self):
        env = LycoEnvironment(self.config, self.config.scenarios[SCENARIOS["S3"]],
                               STRATEGIES["base"], 0, EnvironmentOverrides(slots=540, sensors=8))
        for slot in range(120):
            env.step(slot)
        summary = env.summarize(120)
        self.assertAlmostEqual(summary["energy_budget_ratio"],
                               env.metrics.total_energy / (120 * sum(env.sensor_energy_budgets)))

    def test_table_three_exports_every_metric_for_every_method(self):
        from experiments import METHODS
        results = [{"scene": scene, "method": method, "seed": seed,
                    "metrics": {field: .5 for field in FIELDS}}
                   for scene in SCENARIOS for method in METHODS for seed in range(2)]
        for method in METHODS:
            self.assertEqual(len(main_cells(results, "Macro", method)), 12)
        with TemporaryDirectory(prefix=".test_", dir=ROOT) as directory:
            from pathlib import Path
            folder = Path(directory)
            table_main(results, folder)
            with (folder / "table_3.csv").open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows), 33)
            self.assertTrue(all(len(row) == 14 for row in rows))
            self.assertTrue(all(cell and cell != "--" for row in rows for cell in row))
            self.assertNotIn(r"\textemdash", (folder / "table_3.tex").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
