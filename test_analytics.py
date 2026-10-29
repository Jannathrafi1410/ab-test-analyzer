"""
test_analytics.py - Automated Unit & Statistical Tests for ABTestAnalyzer
Verifies:
1. Experiment Quality & Telemetry Audit (SRM, user collision, sequence monotonicity, event bounds, missing outcomes, maturity)
2. Hypothesis testing suite & Non-Inferiority calculation for AOV
3. Prospective power & sample size targets
4. Treatment-by-segment interaction test
5. Decision engine quality gating
"""

import unittest
import pandas as pd
from analysis import ABTestAnalyzer


class TestABTestAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv('ab_test_data.csv')
        cls.analyzer = ABTestAnalyzer(cls.df)

    def test_data_integrity_audit(self):
        audit = self.analyzer.experiment_quality_audit(planned_sample_size=40000, planned_min_duration_days=14)
        self.assertEqual(audit['overall_status'], 'PASSED')
        self.assertTrue(audit['checks']['srm']['passed'])
        self.assertTrue(audit['checks']['user_integrity']['passed'])
        self.assertTrue(audit['checks']['funnel_monotonicity']['passed'])
        self.assertTrue(audit['checks']['event_bounds']['passed'])
        self.assertTrue(audit['checks']['outcome_completeness']['passed'])
        self.assertTrue(audit['checks']['maturity_and_sample']['passed'])

    def test_overall_metrics(self):
        metrics = self.analyzer.overall_metrics()
        self.assertIn('control', metrics)
        self.assertIn('treatment', metrics)
        self.assertGreater(metrics['control']['n_users'], 20000)
        self.assertGreater(metrics['treatment']['n_users'], 20000)

    def test_statistical_tests(self):
        tests = self.analyzer.statistical_tests(confidence_level=0.95, aov_margin_pct=0.03)
        # Primary conversion test
        self.assertTrue(tests['cvr']['significant'])
        self.assertGreater(tests['cvr']['lift'], 0)
        # Power framework
        self.assertIn('post_hoc_power_caveat', tests['power'])
        self.assertTrue(tests['power']['sample_target_reached'])

    def test_aov_non_inferiority(self):
        # At 3% margin, non-inferiority should be unconfirmed (lower CI < -$1.97)
        ni_3pct = self.analyzer.aov_non_inferiority_test(margin_pct=0.03)
        self.assertFalse(ni_3pct['non_inferior'])
        
        # At 6% margin, non-inferiority should be confirmed
        ni_6pct = self.analyzer.aov_non_inferiority_test(margin_pct=0.06)
        self.assertTrue(ni_6pct['non_inferior'])

    def test_segment_interaction(self):
        res = self.analyzer.segment_interaction_test()
        self.assertTrue(res['omnibus_significant'])
        self.assertLess(res['omnibus_p_value'], 0.001)
        self.assertTrue(res['power_user_interaction_sig'])
        self.assertLess(res['power_user_interaction_coef'], 0)

    def test_decision_engine(self):
        decision = self.analyzer.decision_engine(confidence_level=0.95, aov_margin_pct=0.05)
        # With 5% margin and power user drop confirmed, conditional rollout is recommended
        self.assertEqual(decision['verdict'], 'CONDITIONAL_ROLLOUT')


if __name__ == '__main__':
    unittest.main()
