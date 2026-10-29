"""
analysis.py - Rigorous Statistical & Funnel Analytics Engine
Simulated Case Study & Portfolio Project

Provides:
1. Automated Experiment Quality & Telemetry Audit (SRM, duplicates, monotonicity, event bounds, missing data, maturity)
2. Hypothesis testing suite with Two-Proportion Z-Test, Welch's T-Test, and Mann-Whitney U
3. Non-Inferiority Testing for Guardrail Metrics (AOV) against a business-approved decline limit
4. Prospective Power & Sample Size Target estimation (distinguished from observed post-test power)
5. Heterogeneous Treatment Effects with formal Treatment-by-Segment Logistic Interaction Tests
6. Multiple comparisons corrections (Bonferroni FWER & Benjamini-Hochberg FDR)
7. Funnel leak progression and time-series novelty burn-in diagnostics
8. Quality-gated Product Decision Engine
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


class ABTestAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self._precompute_user_level()

    def _precompute_user_level(self):
        """
        Aggregates event-level data to user-level records (1 row per unique user).
        """
        # User metadata (variant, segment, country, timestamp, session_duration)
        user_meta = self.df.drop_duplicates(subset=['user_id'])[
            ['user_id', 'session_id', 'variant', 'user_segment', 'country', 'timestamp', 'session_duration_sec']
        ].copy()
        
        # Max funnel step completed
        completed_steps = self.df[self.df['step_completed'] == True]
        max_step = completed_steps.groupby('user_id')['funnel_step'].max().rename('max_step')
        
        # Purchase indicator: completed step 5
        purchased = (max_step == 5).rename('purchased')
        
        # Order values for purchasers
        order_vals = self.df[self.df['order_value'].notna()].groupby('user_id')['order_value'].first().rename('order_value')
        
        # Merge into single user dataframe
        user_df = user_meta.merge(max_step, on='user_id', how='left').fillna({'max_step': 0})
        user_df = user_df.merge(purchased, on='user_id', how='left').fillna({'purchased': False})
        user_df = user_df.merge(order_vals, on='user_id', how='left').fillna({'order_value': 0.0})
        
        user_df['purchased'] = user_df['purchased'].astype(bool)
        user_df['is_purchased'] = user_df['purchased'].astype(int)
        user_df['max_step'] = user_df['max_step'].astype(int)
        user_df['date'] = user_df['timestamp'].dt.date
        
        self.user_df = user_df

    # =========================================================================
    # 1. EXPERIMENT QUALITY & TELEMETRY AUDIT
    # =========================================================================
    def experiment_quality_audit(self, planned_sample_size=40000, planned_min_duration_days=14):
        """
        Comprehensive Pre-Analysis Quality & Telemetry Audit Suite.
        Checks for:
        1. Sample Ratio Mismatch (SRM) via Chi-Square Goodness-of-Fit
        2. Missing or Duplicate Users (null IDs, cross-variant contamination, duplicate events)
        3. Funnel Sequence Integrity (monotonicity check: no user completes step k without step k-1)
        4. Unexpected Event Counts & Anomalies (exact 5 events/user, non-negative durations)
        5. Missing or Contaminated Outcome Data (purchases missing revenue, non-purchases with revenue)
        6. Experiment Maturity (duration in days >= planned minimum, sample size >= planned target)
        """
        # 1. SRM Check
        srm = self.sample_ratio_mismatch()
        
        # 2. Missing or Duplicate Users
        null_users = int(self.df['user_id'].isna().sum())
        variant_counts_per_user = self.df.groupby('user_id')['variant'].nunique()
        cross_variant_collisions = int((variant_counts_per_user > 1).sum())
        duplicate_event_steps = int(self.df.duplicated(subset=['user_id', 'funnel_step']).sum())
        user_integrity_pass = (null_users == 0) and (cross_variant_collisions == 0) and (duplicate_event_steps == 0)
        
        # 3. Funnel Sequence Monotonicity
        user_step_matrix = self.df.pivot(index='user_id', columns='funnel_step', values='step_completed')
        sequence_violations = 0
        for step in range(2, 6):
            if step in user_step_matrix.columns and (step - 1) in user_step_matrix.columns:
                violations = ((user_step_matrix[step] == True) & (user_step_matrix[step - 1] == False)).sum()
                sequence_violations += int(violations)
        sequence_integrity_pass = (sequence_violations == 0)
        
        # 4. Event Bounds & Duration Anomalies
        steps_per_user = self.df.groupby('user_id')['funnel_step'].count()
        users_with_unexpected_step_count = int((steps_per_user != 5).sum())
        negative_durations = int((self.df['session_duration_sec'] < 0).sum())
        event_counts_pass = (users_with_unexpected_step_count == 0) and (negative_durations == 0)
        
        # 5. Missing Outcome Data
        purchasers_no_revenue = int(
            ((self.df['funnel_step'] == 5) & (self.df['step_completed'] == True) & 
             (self.df['order_value'].isna() | (self.df['order_value'] <= 0))).sum()
        )
        non_purchasers_with_revenue = int(
            (((self.df['funnel_step'] != 5) | (self.df['step_completed'] == False)) & 
             self.df['order_value'].notna()).sum()
        )
        missing_dimensions = {
            col: int(self.df[col].isna().sum()) 
            for col in ['variant', 'user_segment', 'country', 'timestamp']
        }
        outcome_data_pass = (purchasers_no_revenue == 0) and (non_purchasers_with_revenue == 0) and (sum(missing_dimensions.values()) == 0)
        
        # 6. Maturity & Sample Adequacy
        duration_seconds = (self.df['timestamp'].max() - self.df['timestamp'].min()).total_seconds()
        duration_days = float(duration_seconds / 86400.0)
        actual_sample_size = int(self.df['user_id'].nunique())
        
        duration_met = duration_days >= planned_min_duration_days
        sample_size_met = actual_sample_size >= planned_sample_size
        maturity_pass = duration_met and sample_size_met
        
        # Overall Status
        has_critical_failure = (
            srm['srm_detected'] or 
            not user_integrity_pass or 
            not sequence_integrity_pass or 
            not outcome_data_pass or 
            (negative_durations > 0)
        )
        
        if has_critical_failure:
            overall_status = "FAILED"
            summary_message = "Critical telemetry or data integrity violation detected. Experiment results cannot be evaluated for rollout."
        elif not maturity_pass:
            overall_status = "WARNING"
            summary_message = f"Data integrity verified, but experiment has not met maturity targets ({duration_days:.1f}/{planned_min_duration_days} days, {actual_sample_size:,}/{planned_sample_size:,} users)."
        else:
            overall_status = "PASSED"
            summary_message = f"All 6 telemetry integrity and maturity checks passed ({actual_sample_size:,} users over {duration_days:.1f} days)."

        return {
            'overall_status': overall_status,
            'summary_message': summary_message,
            'checks': {
                'srm': {
                    'name': 'Sample Ratio Mismatch (SRM)',
                    'passed': not srm['srm_detected'],
                    'status': 'PASS' if not srm['srm_detected'] else 'FAIL',
                    'details': f"p = {srm['p_value']:.4f} (Chi2 = {srm['chi2_stat']:.4f})",
                    'data': srm
                },
                'user_integrity': {
                    'name': 'User Identity & Assignment Integrity',
                    'passed': user_integrity_pass,
                    'status': 'PASS' if user_integrity_pass else 'FAIL',
                    'details': f"Null Users: {null_users}, Variant Collisions: {cross_variant_collisions}, Duplicate Step Events: {duplicate_event_steps}"
                },
                'funnel_monotonicity': {
                    'name': 'Funnel Sequence Monotonicity',
                    'passed': sequence_integrity_pass,
                    'status': 'PASS' if sequence_integrity_pass else 'FAIL',
                    'details': f"Sequence violations (skipping prior steps): {sequence_violations}"
                },
                'event_bounds': {
                    'name': 'Event Count & Duration Bounds',
                    'passed': event_counts_pass,
                    'status': 'PASS' if event_counts_pass else 'FAIL',
                    'details': f"Invalid Step Counts (!=5): {users_with_unexpected_step_count}, Negative Durations: {negative_durations}"
                },
                'outcome_completeness': {
                    'name': 'Outcome Data Completeness',
                    'passed': outcome_data_pass,
                    'status': 'PASS' if outcome_data_pass else 'FAIL',
                    'details': f"Purchases missing revenue: {purchasers_no_revenue}, Non-purchases with revenue: {non_purchasers_with_revenue}"
                },
                'maturity_and_sample': {
                    'name': 'Experiment Maturity & Sample Adequacy',
                    'passed': maturity_pass,
                    'status': 'PASS' if maturity_pass else 'WARN',
                    'details': f"Sample Size: {actual_sample_size:,} / {planned_sample_size:,} planned ({'Met' if sample_size_met else 'Unmet'}) | Duration: {duration_days:.1f} / {planned_min_duration_days} days planned ({'Met' if duration_met else 'Unmet'})",
                    'duration_days': duration_days,
                    'planned_min_duration_days': planned_min_duration_days,
                    'actual_sample_size': actual_sample_size,
                    'planned_sample_size': planned_sample_size
                }
            }
        }

    def sample_ratio_mismatch(self, expected_ratio=(0.50, 0.50)):
        """
        Tests for Sample Ratio Mismatch (SRM) using Pearson's Chi-Square Goodness-of-Fit Test.
        A statistically significant deviation (p < 0.01) indicates assignment or logging bias.
        """
        variant_counts = self.user_df['variant'].value_counts()
        c_count = int(variant_counts.get('control', 0))
        t_count = int(variant_counts.get('treatment', 0))
        total = c_count + t_count
        
        if total == 0:
            return {'control_count': 0, 'treatment_count': 0, 'chi2_stat': 0.0, 'p_value': 1.0, 'srm_detected': False}
            
        expected = np.array([total * expected_ratio[0], total * expected_ratio[1]])
        observed = np.array([c_count, t_count])
        
        chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
        
        return {
            'control_count': c_count,
            'treatment_count': t_count,
            'total_users': total,
            'observed_split': (c_count / total, t_count / total) if total > 0 else (0, 0),
            'expected_split': expected_ratio,
            'chi2_stat': float(chi2_stat),
            'p_value': float(p_value),
            'srm_detected': bool(p_value < 0.01)
        }

    def overall_metrics(self):
        """
        Calculates key high-level product and business metrics per variant.
        """
        results = {}
        for variant in ['control', 'treatment']:
            subset = self.user_df[self.user_df['variant'] == variant]
            n_users = len(subset)
            purchasers = subset[subset['purchased']]
            n_purchases = len(purchasers)
            cvr = n_purchases / n_users if n_users > 0 else 0.0
            
            order_values = purchasers['order_value']
            aov = float(order_values.mean()) if n_purchases > 0 else 0.0
            aov_median = float(order_values.median()) if n_purchases > 0 else 0.0
            aov_std = float(order_values.std()) if n_purchases > 1 else 0.0
            
            total_rev = float(subset['order_value'].sum())
            rpu = total_rev / n_users if n_users > 0 else 0.0
            avg_duration = float(subset['session_duration_sec'].mean()) if n_users > 0 else 0.0
            
            results[variant] = {
                'n_users': n_users,
                'n_purchases': n_purchases,
                'cvr': cvr,
                'aov': aov,
                'aov_median': aov_median,
                'aov_std': aov_std,
                'total_revenue': total_rev,
                'rpu': rpu,
                'avg_duration_sec': avg_duration
            }
        return results

    # =========================================================================
    # 2. NON-INFERIORITY TEST FOR GUARDRAIL METRIC (AOV)
    # =========================================================================
    def aov_non_inferiority_test(self, margin_pct=0.03, margin_dollars=None, confidence_level=0.95):
        """
        Formal Non-Inferiority Test for Average Order Value (AOV).
        
        Statistical Rationale:
        A standard two-sided Welch's t-test failing to reach statistical significance (p > 0.05)
        merely demonstrates absence of evidence of a difference—it does NOT prove that any decline
        is small enough for the business to safely absorb (absence of evidence != evidence of absence).
        
        Hypotheses:
          H0: mu_treatment - mu_control <= -delta  (Treatment AOV is inferior by more than margin delta)
          H1: mu_treatment - mu_control > -delta   (Treatment AOV is non-inferior; decline is bounded within margin)
        
        Margin delta:
          Can be specified as a relative percentage of control AOV (default 3.0%) or an absolute dollar amount.
        """
        c_purchasers = self.user_df[(self.user_df['variant'] == 'control') & (self.user_df['purchased'])]['order_value']
        t_purchasers = self.user_df[(self.user_df['variant'] == 'treatment') & (self.user_df['purchased'])]['order_value']
        
        n_c = len(c_purchasers)
        n_t = len(t_purchasers)
        
        if n_c <= 1 or n_t <= 1:
            return {
                'delta': 0.0,
                'margin_dollars': 0.0,
                'margin_pct': margin_pct,
                'se': 0.0,
                't_stat': 0.0,
                'df': 1.0,
                'p_value_ni': 1.0,
                'lower_bound_ci': 0.0,
                'non_inferior': False,
                'confidence_level': confidence_level,
                'interpretation': "Insufficient purchaser sample size for non-inferiority testing."
            }
            
        mean_c = float(c_purchasers.mean())
        mean_t = float(t_purchasers.mean())
        var_c = float(c_purchasers.var())
        var_t = float(t_purchasers.var())
        
        delta_aov = mean_t - mean_c
        
        # Establish non-inferiority margin delta
        if margin_dollars is not None:
            delta_margin = float(abs(margin_dollars))
            effective_margin_pct = (delta_margin / mean_c) if mean_c > 0 else 0.0
        else:
            delta_margin = float(margin_pct * mean_c)
            effective_margin_pct = margin_pct
            
        # Standard error under Welch's unequal variance formulation
        se_aov = float(np.sqrt((var_t / n_t) + (var_c / n_c)))
        
        # Welch-Satterthwaite degrees of freedom
        numerator = ((var_t / n_t) + (var_c / n_c)) ** 2
        denominator = (((var_t / n_t) ** 2) / (n_t - 1)) + (((var_c / n_c) ** 2) / (n_c - 1))
        df_welch = float(numerator / denominator) if denominator > 0 else float(n_t + n_c - 2)
        
        # One-sided t-statistic against lower margin: (delta - (-delta_margin)) / se
        t_ni = float((delta_aov + delta_margin) / se_aov) if se_aov > 0 else 0.0
        
        # One-sided p-value for non-inferiority
        p_value_ni = float(1.0 - stats.t.cdf(t_ni, df=df_welch))
        
        # One-sided lower confidence bound at (1 - alpha): delta - t_crit * se
        alpha = 1.0 - confidence_level
        t_crit_onesided = float(stats.t.ppf(1.0 - alpha, df=df_welch))
        lower_bound_ci = float(delta_aov - t_crit_onesided * se_aov)
        
        # Non-inferiority is demonstrated if lower confidence bound > -delta_margin (or p_ni < alpha)
        non_inferior = bool(lower_bound_ci > -delta_margin)
        
        if non_inferior:
            interpretation = (
                f"GUARDRAIL CONFIRMED: Non-inferiority established at {confidence_level*100:.0f}% confidence "
                f"(p_NI = {p_value_ni:.4f} < {alpha:.2f}). The lower bound of AOV change (${lower_bound_ci:+.2f}) "
                f"is strictly above the business-approved decline limit of -${delta_margin:.2f} (-{effective_margin_pct*100:.1f}%)."
            )
        else:
            interpretation = (
                f"GUARDRAIL UNCONFIRMED: Test fails to establish non-inferiority at {confidence_level*100:.0f}% confidence "
                f"(p_NI = {p_value_ni:.4f} >= {alpha:.2f}). Although the observed drop is only ${delta_aov:+.2f}, "
                f"sampling uncertainty allows a potential loss down to ${lower_bound_ci:+.2f}, which exceeds the "
                f"business-approved limit of -${delta_margin:.2f} (-{effective_margin_pct*100:.1f}%)."
            )
            
        return {
            'delta': delta_aov,
            'margin_dollars': delta_margin,
            'margin_pct': effective_margin_pct,
            'se': se_aov,
            't_stat': t_ni,
            'df': df_welch,
            'p_value_ni': p_value_ni,
            'lower_bound_ci': lower_bound_ci,
            'non_inferior': non_inferior,
            'confidence_level': confidence_level,
            'interpretation': interpretation
        }

    # =========================================================================
    # 3. STATISTICAL TESTS & METHODOLOGY SEPARATION
    # =========================================================================
    def statistical_tests(self, confidence_level=0.95, aov_margin_pct=0.03, aov_margin_dollars=None):
        """
        Executes complete statistical testing suite:
        - Two-proportion Z-test for CVR (Primary Metric) + Risk Difference & Lift Confidence Intervals
        - Welch's T-test, Mann-Whitney U test, and NON-INFERIORITY test for AOV (Guardrail Metric)
        - Welch's T-test for Revenue Per User (RPU)
        - Welch's T-test for Session Duration
        - Methodologically rigorous Power Analysis:
          * Separates observed effect and uncertainty (CI)
          * Pre-planned Minimum Detectable Effect (MDE) and planned sample size target
          * Explicit caveat on the post-hoc power fallacy
        """
        alpha = 1.0 - confidence_level
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        
        metrics = self.overall_metrics()
        c = metrics['control']
        t = metrics['treatment']
        
        # 1. CVR Two-Proportion Z-Test
        if c['n_users'] > 0 and t['n_users'] > 0:
            z_cvr, p_cvr = proportions_ztest(
                count=[t['n_purchases'], c['n_purchases']],
                nobs=[t['n_users'], c['n_users']],
                alternative='two-sided'
            )
            # Absolute Difference
            abs_diff = t['cvr'] - c['cvr']
            se_diff = np.sqrt(
                (c['cvr'] * (1.0 - c['cvr']) / c['n_users']) + 
                (t['cvr'] * (1.0 - t['cvr']) / t['n_users'])
            )
            abs_ci_low = abs_diff - z_crit * se_diff
            abs_ci_high = abs_diff + z_crit * se_diff
            
            # Relative Lift
            lift = (t['cvr'] - c['cvr']) / c['cvr'] if c['cvr'] > 0 else 0.0
            # Standard error of relative lift via Delta Method
            se_lift = np.sqrt(
                (t['cvr'] * (1.0 - t['cvr']) / t['n_users']) + 
                (c['cvr'] * (1.0 - c['cvr']) / c['n_users'])
            ) / c['cvr'] if c['cvr'] > 0 else 0.0
            lift_ci_low = lift - z_crit * se_lift
            lift_ci_high = lift + z_crit * se_lift
        else:
            z_cvr, p_cvr, abs_diff, abs_ci_low, abs_ci_high = 0.0, 1.0, 0.0, 0.0, 0.0
            lift, lift_ci_low, lift_ci_high, se_lift = 0.0, 0.0, 0.0, 0.0

        # 2. AOV Welch's T-Test + Mann-Whitney U test + Non-Inferiority Test (Purchasers only)
        c_purchasers = self.user_df[(self.user_df['variant'] == 'control') & (self.user_df['purchased'])]['order_value']
        t_purchasers = self.user_df[(self.user_df['variant'] == 'treatment') & (self.user_df['purchased'])]['order_value']
        
        if len(c_purchasers) > 1 and len(t_purchasers) > 1:
            t_aov, p_aov = stats.ttest_ind(t_purchasers, c_purchasers, equal_var=False)
            u_stat, p_aov_mann = stats.mannwhitneyu(t_purchasers, c_purchasers, alternative='two-sided')
            aov_diff = float(t_purchasers.mean() - c_purchasers.mean())
            aov_lift = aov_diff / float(c_purchasers.mean()) if c_purchasers.mean() > 0 else 0.0
            
            se_aov = np.sqrt((t_purchasers.var() / len(t_purchasers)) + (c_purchasers.var() / len(c_purchasers)))
            aov_ci_low = aov_diff - z_crit * se_aov
            aov_ci_high = aov_diff + z_crit * se_aov
        else:
            t_aov, p_aov, u_stat, p_aov_mann = 0.0, 1.0, 0.0, 1.0
            aov_diff, aov_lift, aov_ci_low, aov_ci_high = 0.0, 0.0, 0.0, 0.0

        # Non-inferiority test for AOV guardrail
        aov_ni = self.aov_non_inferiority_test(
            margin_pct=aov_margin_pct,
            margin_dollars=aov_margin_dollars,
            confidence_level=confidence_level
        )

        # 3. Revenue Per User (RPU) Welch's T-Test (All users)
        c_rpu = self.user_df[self.user_df['variant'] == 'control']['order_value']
        t_rpu = self.user_df[self.user_df['variant'] == 'treatment']['order_value']
        
        if len(c_rpu) > 1 and len(t_rpu) > 1:
            t_rpu_stat, p_rpu = stats.ttest_ind(t_rpu, c_rpu, equal_var=False)
            rpu_diff = float(t_rpu.mean() - c_rpu.mean())
            rpu_lift = rpu_diff / float(c_rpu.mean()) if c_rpu.mean() > 0 else 0.0
            se_rpu = np.sqrt((t_rpu.var() / len(t_rpu)) + (c_rpu.var() / len(c_rpu)))
            rpu_ci_low = rpu_diff - z_crit * se_rpu
            rpu_ci_high = rpu_diff + z_crit * se_rpu
        else:
            t_rpu_stat, p_rpu, rpu_diff, rpu_lift, rpu_ci_low, rpu_ci_high = 0.0, 1.0, 0.0, 0.0, 0.0, 0.0

        # 4. Session Duration T-Test
        c_dur = self.user_df[self.user_df['variant'] == 'control']['session_duration_sec']
        t_dur = self.user_df[self.user_df['variant'] == 'treatment']['session_duration_sec']
        t_dur_stat, p_dur = stats.ttest_ind(t_dur, c_dur, equal_var=False) if len(c_dur) > 1 and len(t_dur) > 1 else (0.0, 1.0)

        # 5. Methodological Power & Minimum Detectable Effect (MDE) Framework
        power_analyzer = NormalIndPower()
        
        # A. Pre-planned prospective sample size target for MDE = 1.0 percentage point at 80% power
        planned_mde_pp = 0.01
        target_power = 0.80
        if c['cvr'] > 0:
            h_mde_1pp = 2 * (np.arcsin(np.sqrt(c['cvr'] + planned_mde_pp)) - np.arcsin(np.sqrt(c['cvr'])))
            required_sample_1pp = power_analyzer.solve_power(
                effect_size=h_mde_1pp,
                power=target_power,
                alpha=alpha,
                ratio=1.0,
                alternative='two-sided'
            )
            # Sample adequacy: Actual N vs Planned Target N
            actual_n = min(c['n_users'], t['n_users'])
            sample_adequacy_ratio = (actual_n / required_sample_1pp) if required_sample_1pp > 0 else 0.0
            
            # B. Observed post-test power calculation (provided with explicit analytical caution)
            if t['cvr'] > 0:
                h_observed = 2 * (np.arcsin(np.sqrt(t['cvr'])) - np.arcsin(np.sqrt(c['cvr'])))
                observed_post_hoc_power = power_analyzer.solve_power(
                    effect_size=abs(h_observed),
                    nobs1=actual_n,
                    alpha=alpha,
                    ratio=1.0,
                    alternative='two-sided'
                )
            else:
                observed_post_hoc_power = 0.0
        else:
            required_sample_1pp = 0
            sample_adequacy_ratio = 0.0
            observed_post_hoc_power = 0.0

        return {
            'cvr': {
                'z_stat': float(z_cvr),
                'p_value': float(p_cvr),
                'significant': bool(p_cvr < alpha),
                'abs_diff': float(abs_diff),
                'abs_ci_low': float(abs_ci_low),
                'abs_ci_high': float(abs_ci_high),
                'lift': float(lift),
                'lift_ci_low': float(lift_ci_low),
                'lift_ci_high': float(lift_ci_high),
                'se_lift': float(se_lift)
            },
            'aov': {
                't_stat': float(t_aov),
                'p_value': float(p_aov),
                'p_value_mannwhitney': float(p_aov_mann),
                'significant': bool(p_aov < alpha),
                'diff': float(aov_diff),
                'lift': float(aov_lift),
                'ci_low': float(aov_ci_low),
                'ci_high': float(aov_ci_high),
                'non_inferiority': aov_ni
            },
            'rpu': {
                't_stat': float(t_rpu_stat),
                'p_value': float(p_rpu),
                'significant': bool(p_rpu < alpha),
                'diff': float(rpu_diff),
                'lift': float(rpu_lift),
                'ci_low': float(rpu_ci_low),
                'ci_high': float(rpu_ci_high),
                'se_rpu': float(se_rpu)
            },
            'session_duration': {
                't_stat': float(t_dur_stat),
                'p_value': float(p_dur),
                'significant': bool(p_dur < alpha),
                'control_mean': float(c['avg_duration_sec']),
                'treatment_mean': float(t['avg_duration_sec'])
            },
            'power': {
                # 1. Observed effect and uncertainty
                'observed_abs_diff': float(abs_diff),
                'observed_abs_ci': (float(abs_ci_low), float(abs_ci_high)),
                'observed_lift': float(lift),
                'observed_lift_ci': (float(lift_ci_low), float(lift_ci_high)),
                
                # 2. Planned MDE & Sample Size target
                'planned_mde_pp': float(planned_mde_pp),
                'target_power': float(target_power),
                'required_sample_per_variant': int(required_sample_1pp) if not np.isnan(required_sample_1pp) else 0,
                'actual_sample_per_variant': min(c['n_users'], t['n_users']),
                'sample_adequacy_ratio': float(sample_adequacy_ratio),
                'sample_target_reached': bool(sample_adequacy_ratio >= 1.0),
                
                # 3. Post-test observed power (with analytical caveat)
                'observed_post_hoc_power': float(observed_post_hoc_power),
                'post_hoc_power_caveat': (
                    "METHODOLOGICAL NOTE ON POST-TEST POWER: "
                    "Calculating power using the observed effect size is known as the 'post-hoc power fallacy' "
                    "(Hoenig & Heisey, 2001). Observed post-test power is a redundant monotonic transformation of the p-value "
                    "and provides no additional evidence. Experiment decisions should be guided by (1) the observed effect and "
                    "its confidence interval, and (2) whether the test achieved its prospective pre-planned sample size target."
                )
            }
        }

    def funnel_by_variant(self):
        """
        Computes stage-by-stage funnel progression and drop-off metrics for both variants.
        Funnel Steps: 1: Landing -> 2: Cart -> 3: Checkout -> 4: Payment -> 5: Purchase
        """
        step_names = {1: 'Landing', 2: 'Cart', 3: 'Checkout', 4: 'Payment', 5: 'Purchase'}
        
        funnel_records = []
        for step in range(1, 6):
            step_name = step_names[step]
            c_users = int((self.user_df['variant'] == 'control').sum())
            t_users = int((self.user_df['variant'] == 'treatment').sum())
            
            c_reached = int(((self.user_df['variant'] == 'control') & (self.user_df['max_step'] >= step)).sum())
            t_reached = int(((self.user_df['variant'] == 'treatment') & (self.user_df['max_step'] >= step)).sum())
            
            c_cum_pct = (c_reached / c_users * 100.0) if c_users > 0 else 0.0
            t_cum_pct = (t_reached / t_users * 100.0) if t_users > 0 else 0.0
            
            funnel_records.append({
                'funnel_step': step,
                'step_name': step_name,
                'control_users': c_reached,
                'treatment_users': t_reached,
                'control_cum_pct': c_cum_pct,
                'treatment_cum_pct': t_cum_pct,
                'pct_delta': t_cum_pct - c_cum_pct
            })
            
        funnel_df = pd.DataFrame(funnel_records)
        
        # Step-to-step conditional conversion rate (Step N / Step N-1)
        funnel_df['control_step_cvr'] = 100.0
        funnel_df['treatment_step_cvr'] = 100.0
        funnel_df['control_dropoff_pct'] = 0.0
        funnel_df['treatment_dropoff_pct'] = 0.0
        
        for i in range(1, len(funnel_df)):
            c_prev = funnel_df.loc[i - 1, 'control_users']
            t_prev = funnel_df.loc[i - 1, 'treatment_users']
            c_curr = funnel_df.loc[i, 'control_users']
            t_curr = funnel_df.loc[i, 'treatment_users']
            
            c_conv = (c_curr / c_prev * 100.0) if c_prev > 0 else 0.0
            t_conv = (t_curr / t_prev * 100.0) if t_prev > 0 else 0.0
            
            funnel_df.loc[i, 'control_step_cvr'] = c_conv
            funnel_df.loc[i, 'treatment_step_cvr'] = t_conv
            funnel_df.loc[i, 'control_dropoff_pct'] = 100.0 - c_conv
            funnel_df.loc[i, 'treatment_dropoff_pct'] = 100.0 - t_conv
            
        return funnel_df

    # =========================================================================
    # 4. SEGMENT ANALYSIS & FORMAL INTERACTION TESTING
    # =========================================================================
    def segment_analysis(self, dimension='user_segment', confidence_level=0.95):
        """
        Analyzes Heterogeneous Treatment Effects (HTE) across cohorts.
        Applies both Bonferroni and Benjamini-Hochberg (FDR) corrections for multiple comparisons.
        
        Methodological Note:
        Testing within each cohort tells us whether an effect is statistically significant within that cohort.
        However, demonstrating that Segment A's lift is significant and Segment B's is not DOES NOT prove
        that the treatment effects differ between Segment A and Segment B (Gelman & Stern, 2006).
        To establish heterogeneous treatment effects, see segment_interaction_test().
        """
        alpha = 1.0 - confidence_level
        z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
        
        categories = sorted(self.user_df[dimension].dropna().unique())
        results = []
        raw_pvalues = []
        
        # Categorize whether segment is preplanned or exploratory
        is_preplanned = (dimension == 'user_segment')
        
        for cat in categories:
            subset = self.user_df[self.user_df[dimension] == cat]
            c = subset[subset['variant'] == 'control']
            t = subset[subset['variant'] == 'treatment']
            
            n_c = len(c)
            n_t = len(t)
            pur_c = int(c['purchased'].sum())
            pur_t = int(t['purchased'].sum())
            
            cvr_c = pur_c / n_c if n_c > 0 else 0.0
            cvr_t = pur_t / n_t if n_t > 0 else 0.0
            lift = (cvr_t - cvr_c) / cvr_c if cvr_c > 0 else 0.0
            
            if n_c > 0 and n_t > 0 and (pur_c + pur_t) > 0:
                z_stat, p_val = proportions_ztest([pur_t, pur_c], [n_t, n_c], alternative='two-sided')
                se_lift = np.sqrt(
                    (cvr_t * (1.0 - cvr_t) / n_t) + 
                    (cvr_c * (1.0 - cvr_c) / n_c)
                ) / cvr_c if cvr_c > 0 else 0.0
                ci_low = lift - z_crit * se_lift
                ci_high = lift + z_crit * se_lift
            else:
                z_stat, p_val, se_lift, ci_low, ci_high = 0.0, 1.0, 0.0, 0.0, 0.0
                
            raw_pvalues.append(p_val)
            results.append({
                dimension: cat,
                'cohort_type': 'Pre-planned Primary' if is_preplanned else 'Exploratory Post-Hoc',
                'n_control': n_c,
                'n_treatment': n_t,
                'purchases_control': pur_c,
                'purchases_treatment': pur_t,
                'cvr_control': cvr_c,
                'cvr_treatment': cvr_t,
                'lift': lift,
                'lift_ci_low': ci_low,
                'lift_ci_high': ci_high,
                'z_stat': z_stat,
                'raw_p_value': p_val,
                'significant_raw': p_val < alpha
            })
            
        seg_df = pd.DataFrame(results)
        
        if len(raw_pvalues) > 0:
            m = len(raw_pvalues)
            seg_df['bonferroni_p_value'] = [min(1.0, p * m) for p in raw_pvalues]
            seg_df['significant_bonferroni'] = seg_df['bonferroni_p_value'] < alpha
            
            reject, fdr_pvalues, _, _ = multipletests(raw_pvalues, alpha=alpha, method='fdr_bh')
            seg_df['fdr_p_value'] = fdr_pvalues
            seg_df['significant_fdr'] = reject
        else:
            seg_df['bonferroni_p_value'] = 1.0
            seg_df['significant_bonferroni'] = False
            seg_df['fdr_p_value'] = 1.0
            seg_df['significant_fdr'] = False
            
        return seg_df

    def segment_interaction_test(self, reference_segment='desktop'):
        """
        Direct Treatment-by-Segment Logistic Regression Interaction Test.
        
        Statistical Rationale:
        A difference in significance between subgroups (e.g. new users p < 0.001 vs power users p = 0.002)
        does not establish that the treatment effect varies across subgroups.
        To statistically confirm Heterogeneous Treatment Effects (HTE), we estimate a logistic model:
          logit(P(purchase=1)) = beta_0 + beta_treat*Treat + sum(beta_seg*Seg) + sum(gamma_seg*(Treat x Seg))
        
        We evaluate:
        1. Omnibus Likelihood-Ratio Test: Comparing the full interaction model against the additive main-effects model.
        2. Segment-specific interaction terms (gamma): Testing whether Treatment effect on each segment
           is significantly different from the reference segment (desktop).
        """
        model_df = self.user_df.copy()
        
        try:
            # Full interaction model
            formula_full = (
                f"is_purchased ~ C(variant, Treatment('control')) * "
                f"C(user_segment, Treatment('{reference_segment}'))"
            )
            model_full = smf.logit(formula_full, data=model_df).fit(disp=False)
            
            # Additive main effects model (no interaction terms)
            formula_base = (
                f"is_purchased ~ C(variant, Treatment('control')) + "
                f"C(user_segment, Treatment('{reference_segment}'))"
            )
            model_base = smf.logit(formula_base, data=model_df).fit(disp=False)
            
            # Omnibus Likelihood-Ratio Test
            lr_stat = float(2 * (model_full.llf - model_base.llf))
            df_diff = int(model_full.df_model - model_base.df_model)
            omnibus_p_value = float(stats.chi2.sf(lr_stat, df_diff))
            omnibus_significant = (omnibus_p_value < 0.05)
            
            # Extract interaction coefficients
            interaction_terms = []
            for param in model_full.params.index:
                if ":" in param:
                    coef = float(model_full.params[param])
                    se = float(model_full.bse[param])
                    z_val = float(model_full.tvalues[param])
                    p_val = float(model_full.pvalues[param])
                    ci = model_full.conf_int().loc[param]
                    ci_low, ci_high = float(ci[0]), float(ci[1])
                    odds_ratio = float(np.exp(coef))
                    
                    # Clean label
                    if "[T." in param:
                        seg_name = param.split("[T.")[-1].split("]")[0]
                    else:
                        seg_name = param
                    
                    interaction_terms.append({
                        'interaction': f"Treatment x {seg_name}",
                        'segment': seg_name,
                        'reference_segment': reference_segment,
                        'coef': coef,
                        'odds_ratio': odds_ratio,
                        'std_err': se,
                        'z_stat': z_val,
                        'p_value': p_val,
                        'ci_low': ci_low,
                        'ci_high': ci_high,
                        'significant': p_val < 0.05
                    })
                    
            interaction_df = pd.DataFrame(interaction_terms)
            
            # Check specifically for power_user interaction
            pu_row = interaction_df[interaction_df['segment'] == 'power_user']
            if len(pu_row) > 0:
                pu_interaction_sig = bool(pu_row['significant'].iloc[0])
                pu_interaction_p = float(pu_row['p_value'].iloc[0])
                pu_interaction_coef = float(pu_row['coef'].iloc[0])
                pu_interaction_z = float(pu_row['z_stat'].iloc[0])
            else:
                pu_interaction_sig, pu_interaction_p, pu_interaction_coef, pu_interaction_z = False, 1.0, 0.0, 0.0
                
            interpretation = (
                f"Omnibus LR Test: Chi2({df_diff}) = {lr_stat:.2f}, p = {omnibus_p_value:.4e}. "
                + (
                    "Heterogeneous treatment effects are statistically confirmed. Specifically, "
                    f"the 'Treatment x power_user' interaction term is significantly negative (beta = {pu_interaction_coef:.4f}, z = {pu_interaction_z:.2f}, p = {pu_interaction_p:.4e}), "
                    f"confirming that power users respond differently to treatment than {reference_segment} users."
                    if pu_interaction_sig else
                    "No significant segment interaction detected."
                )
            )
            
            return {
                'omnibus_lr_stat': lr_stat,
                'omnibus_df': df_diff,
                'omnibus_p_value': omnibus_p_value,
                'omnibus_significant': omnibus_significant,
                'power_user_interaction_sig': pu_interaction_sig,
                'power_user_interaction_p': pu_interaction_p,
                'power_user_interaction_coef': pu_interaction_coef,
                'interaction_table': interaction_df,
                'interpretation': interpretation
            }
        except Exception as e:
            return {
                'omnibus_lr_stat': 0.0,
                'omnibus_df': 0,
                'omnibus_p_value': 1.0,
                'omnibus_significant': False,
                'power_user_interaction_sig': False,
                'power_user_interaction_p': 1.0,
                'power_user_interaction_coef': 0.0,
                'interaction_table': pd.DataFrame(),
                'interpretation': f"Interaction test could not converge: {str(e)}"
            }

    def time_series_analysis(self):
        """
        Calculates daily and running cumulative metrics over the 28-day window
        to detect novelty effects, burn-in periods, or conversion rate stabilization.
        """
        daily = self.user_df.groupby(['date', 'variant']).agg(
            users=('user_id', 'count'),
            purchases=('purchased', 'sum'),
            revenue=('order_value', 'sum')
        ).reset_index()
        
        daily['cvr'] = daily['purchases'] / daily['users']
        daily['rpu'] = daily['revenue'] / daily['users']
        
        cum_df = self.user_df.sort_values('timestamp').copy()
        cum_records = []
        
        dates = sorted(self.user_df['date'].unique())
        c_users_cum = 0
        c_purch_cum = 0
        t_users_cum = 0
        t_purch_cum = 0
        
        for dt in dates:
            day_data = cum_df[cum_df['date'] == dt]
            
            c_day = day_data[day_data['variant'] == 'control']
            t_day = day_data[day_data['variant'] == 'treatment']
            
            c_users_cum += len(c_day)
            c_purch_cum += int(c_day['purchased'].sum())
            t_users_cum += len(t_day)
            t_purch_cum += int(t_day['purchased'].sum())
            
            c_cvr = c_purch_cum / c_users_cum if c_users_cum > 0 else 0.0
            t_cvr = t_purch_cum / t_users_cum if t_users_cum > 0 else 0.0
            lift = (t_cvr - c_cvr) / c_cvr if c_cvr > 0 else 0.0
            
            cum_records.append({
                'date': dt,
                'control_users_cum': c_users_cum,
                'treatment_users_cum': t_users_cum,
                'control_cvr_cum': c_cvr,
                'treatment_cvr_cum': t_cvr,
                'cum_lift': lift
            })
            
        return daily, pd.DataFrame(cum_records)

    # =========================================================================
    # 5. AUTOMATED PRODUCT DECISION ENGINE
    # =========================================================================
    def decision_engine(self, confidence_level=0.95, aov_margin_pct=0.03, planned_sample_size=40000, planned_min_duration_days=14):
        """
        Automated Product Decision Engine with Quality Gating & Guardrail Non-Inferiority.
        
        Evaluation Flow:
        1. Experiment Quality Audit:
           - Halts recommendation if SRM, duplicate users, or sequence monotonicity violations are detected.
           - Emits an Underpowered / Immature Experiment warning if sample size or duration targets are unmet.
        2. Primary Metric (CVR): Two-sided significance & directional lift.
        3. Guardrail Metric (AOV): Non-inferiority test against business-approved margin.
        4. Heterogeneous Treatment Effects: Evaluates direct treatment-by-segment interaction test.
        5. Simulation Context: Clearly distinguishes synthetic generator assumptions from telemetry findings.
        """
        quality = self.experiment_quality_audit(
            planned_sample_size=planned_sample_size,
            planned_min_duration_days=planned_min_duration_days
        )
        metrics = self.overall_metrics()
        tests = self.statistical_tests(
            confidence_level=confidence_level,
            aov_margin_pct=aov_margin_pct
        )
        seg_df = self.segment_analysis(dimension='user_segment', confidence_level=confidence_level)
        interaction_res = self.segment_interaction_test()
        
        cvr_sig = tests['cvr']['significant']
        cvr_positive = tests['cvr']['lift'] > 0
        
        # AOV guardrail: evaluated via formal non-inferiority test
        aov_ni = tests['aov']['non_inferiority']
        aov_non_inferior = aov_ni['non_inferior']
        
        # Power user segment evaluation
        pu_row = seg_df[seg_df['user_segment'] == 'power_user']
        if len(pu_row) > 0:
            pu_lift = float(pu_row['lift'].iloc[0])
            pu_within_sig = bool(pu_row['significant_raw'].iloc[0])
        else:
            pu_lift, pu_within_sig = 0.0, False
            
        # Is power user drop statistically confirmed by both within-cohort test AND interaction test?
        pu_interaction_sig = interaction_res['power_user_interaction_sig']
        pu_statistically_hurt = (pu_lift < 0) and pu_within_sig and pu_interaction_sig

        # Decision Logic with Quality Gating
        if quality['overall_status'] == 'FAILED':
            verdict = "HALT_DATA_INTEGRITY_VIOLATION"
            badge_color = "#E41E3F"
            title = "HALT ROLLOUT: Telemetry / Data Integrity Violation"
            explanation = (
                f"Experiment cannot be evaluated due to critical quality audit failures: {quality['summary_message']} "
                "Resolve logging or assignment bugs before analyzing product metrics."
            )
        elif quality['overall_status'] == 'WARNING':
            verdict = "INCONCLUSIVE_IMMATURE_EXPERIMENT"
            badge_color = "#F59E0B"
            title = "PAUSE DECISION: Experiment Maturity Targets Unmet"
            explanation = (
                f"The experiment has not yet reached planned sample size or duration targets: {quality['summary_message']} "
                "Premature rollout decisions risk false positives (peeking bias) and seasonal distortion."
            )
        elif cvr_sig and cvr_positive and aov_non_inferior and not pu_statistically_hurt:
            verdict = "SHIP_TO_100"
            badge_color = "#31A24C"
            title = "SHIP TO 100% OF USERS (Unanimous Winner)"
            explanation = (
                f"Treatment delivered a statistically significant conversion lift of +{tests['cvr']['lift']*100:.2f}% "
                f"(95% CI: [{tests['cvr']['lift_ci_low']*100:+.2f}%, {tests['cvr']['lift_ci_high']*100:+.2f}%]). "
                f"AOV guardrail is confirmed non-inferior within the {aov_margin_pct*100:.1f}% margin, with no segment degradation."
            )
        elif cvr_sig and cvr_positive and pu_statistically_hurt:
            verdict = "CONDITIONAL_ROLLOUT"
            badge_color = "#F7B928"
            if aov_non_inferior:
                title = "CONDITIONAL ROLLOUT: Segmented Rollout Recommended"
                guardrail_note = f"AOV guardrail is confirmed non-inferior within the {aov_margin_pct*100:.1f}% limit."
            else:
                title = "CONDITIONAL ROLLOUT: Segmented Rollout + AOV Monitoring"
                guardrail_note = f"AOV guardrail non-inferiority is unconfirmed at the {aov_margin_pct*100:.1f}% limit (lower CI bound: ${aov_ni['lower_bound_ci']:+.2f} < -${aov_ni['margin_dollars']:.2f})."
                
            explanation = (
                f"Primary conversion rose +{tests['cvr']['lift']*100:.2f}% overall (95% CI: [{tests['cvr']['lift_ci_low']*100:+.2f}%, {tests['cvr']['lift_ci_high']*100:+.2f}%]), "
                f"driven by strong gains in new (+{seg_df.loc[seg_df['user_segment']=='new_user', 'lift'].iloc[0]*100:.1f}%) and mobile (+{seg_df.loc[seg_df['user_segment']=='mobile', 'lift'].iloc[0]*100:.1f}%) users. "
                f"However, power users experienced a statistically confirmed negative interaction (lift: {pu_lift*100:.2f}%, interaction p = {interaction_res['power_user_interaction_p']:.4f}). "
                f"{guardrail_note} Roll out treatment only to new and mobile segments while maintaining control for power users until friction is addressed."
            )
        elif cvr_sig and cvr_positive and not aov_non_inferior:
            verdict = "HOLD_GUARDRAIL_UNCONFIRMED"
            badge_color = "#F59E0B"
            title = "HOLD ROLLOUT: AOV Guardrail Non-Inferiority Not Established"
            explanation = (
                f"Although CVR increased (+{tests['cvr']['lift']*100:.2f}%), AOV failed the non-inferiority test at the "
                f"{aov_margin_pct*100:.1f}% decline limit (lower bound: ${aov_ni['lower_bound_ci']:+.2f} < -${aov_ni['margin_dollars']:.2f}). "
                "Higher conversion may be offset by basket deflation. Verify basket margin before proceeding."
            )
        elif cvr_sig and not cvr_positive:
            verdict = "DO_NOT_SHIP_NEGATIVE"
            badge_color = "#E41E3F"
            title = "DO NOT SHIP: Statistically Significant Degradation"
            explanation = "Treatment resulted in a statistically significant decline in checkout conversion rate. Revert to control immediately."
        else:
            verdict = "INCONCLUSIVE_NO_SIGNIFICANCE"
            badge_color = "#5D6C7B"
            title = "INCONCLUSIVE: No Statistically Significant Effect Detected"
            explanation = (
                f"Conversion rate difference is not statistically significant (p = {tests['cvr']['p_value']:.4f} > {1-confidence_level:.2f}). "
                "Data lacks sufficient evidence to justify rollout."
            )

        return {
            'verdict': verdict,
            'badge_color': badge_color,
            'title': title,
            'explanation': explanation,
            'quality': quality,
            'tests': tests,
            'metrics': metrics,
            'interaction': interaction_res,
            'segment_df': seg_df,
            'cvr_sig': cvr_sig,
            'cvr_positive': cvr_positive,
            'aov_non_inferior': aov_non_inferior,
            'power_user_hurt': pu_statistically_hurt,
            'power_user_lift': pu_lift,
            'rpu_lift': tests['rpu']['lift']
        }
