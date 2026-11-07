"""Statistical-unit and isolated energy-ablation checks (no learned models)."""

import unittest

from experiments import ROOT, SCENARIOS
from odte_lyco.config import load_config
from odte_lyco.controller import ControllerContext
from odte_lyco.environment import EnvironmentOverrides, LycoEnvironment
from odte_lyco.models import STRATEGIES, Task
from tables import mean_ci, seed_values


class ReproductionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
