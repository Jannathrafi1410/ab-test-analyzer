# E-Commerce A/B Testing & Funnel Analytics Platform: A Simulated Case Study

[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://python.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.22+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![Statsmodels](https://img.shields.io/badge/Statsmodels-0.14+-4C51BF)](https://www.statsmodels.org/)
[![Status](https://img.shields.io/badge/Type-Simulated_Portfolio_Case_Study-0064E0)](#ground-truth-disclosure-and-simulated-nature)

> **Portfolio Scope & Synthetic Data Disclosure:**  
> This project is an interactive **portfolio dashboard and statistical analysis framework** demonstrating an end-to-end A/B test evaluation using **synthetic experiment data**. It showcases rigorous experimentation workflows, automated data-quality audits, non-inferiority testing, interaction modeling, and quality-gated decision logic. The data is simulated and does not represent real production user data or empirical evidence of checkout behavior.

---

## How to Run

### Quickstart (Local Python)

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/ab-test-funnel.git
cd ab-test-funnel

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: .\venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. (Optional) Run the automated unit & statistical test suite
python -m unittest test_analytics.py

# 5. Launch the Streamlit interactive dashboard
streamlit run app.py
```

The application will be accessible in your browser at `http://localhost:8501`.

### Run with Docker

```bash
# Build the container image
docker build -t ab-test-dashboard .

# Run the container
docker run -p 8501:8501 ab-test-dashboard
```

---

## Dashboard Architecture & Visual Demo

The dashboard is organized into five structured analytical tabs designed around an executive decision workflow:

```
+---------------------------------------------------------------------------------------------------------+
| EXPERIMENTATION ANALYTICS PLATFORM  [CASE-STUDY-09] [SIMULATED EXPERIMENT DATA]                         |
| Simulated Product Analyst Case Study: One-Click Checkout Redesign A/B Test                              |
+---------------------------------------------------------------------------------------------------------+
| [Executive Overview]  [Statistical Tests]  [Funnel Analysis]  [Segment Deep-Dive]  [Final Recommendation]|
+---------------------------------------------------------------------------------------------------------+
|  KPI SCORECARD:                                                                                         |
|  +--------------------+  +--------------------+  +--------------------+  +--------------------+         |
|  | CVR (Primary)      |  | AOV (Guardrail)    |  | RPU (Economic)     |  | Projected Annual   |         |
|  | 20.93% (+8.82%)    |  | $64.42 (-$1.11)    |  | $13.48 (+6.98%)    |  | +$52.7M Run-Rate   |         |
|  | [Stat Sig Lift]    |  | [Non-Inf Unconf]   |  | [+$0.88 / user]    |  | 95% CI: $17M-$88M  |         |
|  +--------------------+  +--------------------+  +--------------------+  +--------------------+         |
|                                                                                                         |
|  TELEMETRY & QUALITY AUDIT (6/6 Checks Passed)       DECISION VERDICT                                   |
|  - SRM Chi-Square: Valid (p = 0.6355)                 CONDITIONAL ROLLOUT:                              |
|  - User Identity & Contamination: 0 collisions        Roll out to New & Mobile traffic only;            |
|  - Funnel Monotonicity: 0 sequence errors             protect Power Users & monitor AOV guardrail.      |
|  - Event Bounds & Durations: Clean bounds                                                               |
|  - Outcome Completeness: 100% complete                                                                  |
|  - Maturity: 28.0 / 14 Days, 50,000 / 40,000 Users                                                      |
+---------------------------------------------------------------------------------------------------------+
```

---

## Ground-Truth Disclosure and Simulated Nature

### What Was Built Into the Generator (`generate_data.py`)
The generator pre-calibrates step completion probabilities across cohorts to model a realistic e-commerce scenario:
1. **50,000 Synthetic Users:** Randomly assigned 50/50 to:
   - `control`: Legacy 3-Step Checkout Flow
   - `treatment`: One-Click Checkout Redesign
2. **Pre-Programmed Conversion Targets by Cohort:**
   - **New Users (30%):** Built-in $+26.0\%$ conversion lift ($10.0\% \rightarrow 12.6\%$).
   - **Mobile Sessions (30%):** Built-in $+23.9\%$ conversion lift ($15.5\% \rightarrow 19.2\%$).
   - **Desktop Sessions (20%):** Built-in $+13.8\%$ conversion lift ($19.5\% \rightarrow 22.2\%$).
   - **Power Users (20%):** Built-in $-8.0\%$ conversion drop ($37.5\% \rightarrow 34.5\%$).
3. **Order Value (AOV Guardrail):** Generated via log-normal distributions without an engineered treatment effect, producing an observed sample difference of $-\$1.11$ (95% CI: $[-\$2.98, +\$0.75]$).

### Important Methodological Distinction on Causal Root Causes
The narrative that **Power Users** experienced checkout friction due to habitual preferences for saved cards and addresses is a **plausible design hypothesis** that motivated the simulation setup. 

However, **this dataset does not measure or establish that cause**. The telemetry records only funnel milestone completions, timestamps, and order values—it does not log address switcher clicks, credit card toggles, or cancellation reasons. In an actual product organization, confirming this causal explanation requires instrumenting card/address switching telemetry and conducting qualitative usability interviews before building remediation features.

---

## Concise Data Dictionary

### Event-Level Telemetry Schema (`ab_test_data.csv`, 250,000 rows)

| Column Name | Data Type | Nullable | Description / Domain |
|---|---|---|---|
| `user_id` | String | No | Unique synthetic user identifier (`U_000000` to `U_049999`). |
| `session_id` | String | No | Unique session identifier (`S_000000` to `S_049999`). |
| `variant` | String | No | Experimental assignment: `'control'` (3-step flow) or `'treatment'` (one-click flow). |
| `user_segment` | String | No | Behavioral cohort: `'new_user'`, `'mobile'`, `'desktop'`, `'power_user'`. |
| `country` | String | No | Geographic market: `'US'`, `'UK'`, `'DE'`, `'IN'`. |
| `funnel_step` | Integer | No | Funnel milestone index: `1` (Landing), `2` (Cart), `3` (Checkout), `4` (Payment), `5` (Purchase). |
| `step_completed` | Boolean | No | Milestone flag: `True` if user completed the milestone; `False` if dropped off. |
| `timestamp` | Datetime | No | Event timestamp across the 28-day experiment window. |
| `order_value` | Float | Yes | Purchase total in USD. Populated *only* when `funnel_step == 5` and `step_completed == True`; `NaN` otherwise. |
| `session_duration_sec` | Integer | No | Total elapsed session duration in seconds (ranges 35 to 500s). |

### Derived User-Level Metrics

| Metric | Level | Formula | Analytical Purpose |
|---|---|---|---|
| **Conversion Rate (CVR)** | User-level | $\frac{\text{Purchasers}}{\text{Total Assigned Users}}$ | Primary metric testing whether One-Click Checkout drives purchase completion. |
| **Average Order Value (AOV)** | Purchaser-level | $\frac{\sum \text{Order Value}}{\text{Purchasers}}$ | Guardrail metric verifying that frictionless checkout does not deflate basket size. |
| **Revenue Per User (RPU)** | User-level | $\frac{\sum \text{Order Value}}{\text{Total Assigned Users}}$ | Net economic impact combining conversion lift with basket integrity. |

---

## Experiment Quality & Telemetry Audit Suite

Prior to evaluating business hypotheses, the platform runs an automated 6-point data integrity audit:

```
[Experiment Quality Audit]
├── 1. Sample Ratio Mismatch (SRM)      --> Pearson Chi-Square test on variant counts (p < 0.01 threshold)
├── 2. User Identity & Contamination    --> Checks null IDs, cross-variant collisions, duplicate step rows
├── 3. Funnel Monotonicity              --> Verifies step k completed requires step k-1 completed
├── 4. Event Bounds & Anomalies         --> Verifies exact 5 events/user, non-negative session durations
├── 5. Outcome Data Completeness        --> Verifies purchases have valid order values, non-purchases have none
└── 6. Maturity & Sample Adequacy       --> Evaluates actual duration >= 14 days and actual sample >= 40,000 users
```

**Quality Gating:** The automated decision engine refuses to issue a rollout recommendation if critical data integrity violations (SRM, contamination, sequence violations) are detected, or issues a pause warning if sample size or duration targets are unmet.

---

## Statistical Framework & Methodological Rigor

### 1. Primary Success Metric: Two-Proportion Z-Test (CVR)
Tests whether treatment significantly altered purchase conversion:
$$Z = \frac{p_t - p_c}{\sqrt{\hat{p}(1-\hat{p})\left(\frac{1}{n_t} + \frac{1}{n_c}\right)}}$$
- **Observed Control CVR:** $19.23\%$ ($4,818 / 25,053$)
- **Observed Treatment CVR:** $20.93\%$ ($5,221 / 24,947$)
- **Absolute Risk Difference:** $+1.70\text{ percentage points}$ ($95\%\text{ CI: } [+1.00\text{pp}, +2.40\text{pp}]$)
- **Relative Lift:** $+8.82\%$ ($95\%\text{ CI via Delta Method: } [+5.17\%, +12.48\%]$)
- **Result:** $Z = 4.737, p = 2.17 \times 10^{-6} \rightarrow$ **Statistically Significant Winner.**

---

### 2. Guardrail Metric: Non-Inferiority Testing for AOV
> **Critical Statistical Insight:** A standard two-sided test failing to detect significance ($p = 0.24$) **does not equal** "guardrail intact." Absence of evidence is not evidence of absence. A test can lack power to detect a small decline while allowing an economically unacceptable drop.

We define a **business-approved AOV decline limit** $\delta$ (e.g. $3.0\%$ or $\$1.97$) and run a formal **Non-Inferiority Test**:
- **Hypotheses:**
  $$H_0: \mu_{\text{treatment}} - \mu_{\text{control}} \le -\delta \quad (\text{Treatment AOV is inferior by more than } \delta)$$
  $$H_1: \mu_{\text{treatment}} - \mu_{\text{control}} > -\delta \quad (\text{Treatment AOV is non-inferior; loss is bounded within } \delta)$$
- **Test Statistic & Welch-Satterthwaite df:**
  $$t_{\text{NI}} = \frac{\Delta_{\text{AOV}} - (-\delta)}{\text{SE}_{\Delta}} = \frac{-\$1.11 - (-\$1.97)}{\$0.952} = 0.895$$
- **One-Sided 95% Lower Confidence Bound (LCB):**
  $$\text{LCB} = \Delta_{\text{AOV}} - t_{0.95, df} \times \text{SE}_{\Delta} = -\$1.11 - (1.645 \times \$0.952) = -\$2.68$$
- **Conclusion at 3.0% Margin:** Because the lower bound ($-\$2.68$) is worse than $-\$1.97$, the test **fails to demonstrate non-inferiority** ($p_{\text{NI}} = 0.1853 \ge 0.05$). The business cannot rule out an AOV drop of up to $4.1\%$ at this sample size.

---

### 3. Prospective Power vs. The Post-Hoc Power Fallacy
The platform explicitly partitions power diagnostics into:
1. **The Observed Effect and Uncertainty:** Measured by point estimates and confidence intervals ($+8.82\%$, 95% CI: $[+5.17\%, +12.48\%]$).
2. **Pre-Planned Prospective Design Targets:**
   - Pre-planned target $MDE = 1.0\text{ percentage point}$ lift
   - Target statistical power $= 80\%$ at $\alpha = 0.05$
   - Required sample size: $24,858$ users per variant ($49,716$ total)
   - Actual enrolled: $25,053$ Control vs $24,947$ Treatment ($100.6\%$ adequacy ratio $\rightarrow$ **Target Met**).
3. **Analytical Caution on Post-Hoc Power:**
   Calculating power after an experiment using the observed effect size is a known methodological fallacy (Hoenig & Heisey, 2001). Observed post-hoc power is a redundant monotonic transformation of the p-value that adds zero evidential value. Decisions are governed by the **confidence bounds** and whether the test met its **pre-planned sample-size target**.

---

### 4. Heterogeneous Treatment Effects & Formal Interaction Testing
Reporting positive lifts in new users (+24%) and a negative lift in power users (-7.7%) within cohorts does not by itself prove that treatment effects differ between cohorts.

To confirm Heterogeneous Treatment Effects (HTE), we estimate a **Treatment-by-Segment Logistic Interaction Model**:
$$\text{logit}(P(\text{purchase}=1)) = \beta_0 + \beta_{\text{treat}}\text{Treat} + \sum \beta_{\text{seg}}\text{Seg} + \sum \gamma_{\text{seg}}(\text{Treat} \times \text{Seg})$$

- **Omnibus Likelihood Ratio Test:** $\chi^2(3) = 46.39, p = 4.69 \times 10^{-10} \rightarrow$ **Interaction confirmed.**
- **Interaction Term for Power Users:** $\beta = -0.3082, z = -4.756, p = 1.97 \times 10^{-6}$ (Odds Ratio: $0.735$, 95% CI: $[0.647, 0.835]$).
- **Primary vs. Exploratory Segmentation:**
  - `user_segment` is classified as **Pre-Planned Primary Stratification**.
  - `country` is classified as **Exploratory Post-Hoc Slices**.
  - Multiple comparisons corrections (Bonferroni FWER, Benjamini-Hochberg FDR) adjust for family-wise error rates, but exploratory findings must be confirmed in follow-up experiments rather than treated as conclusive product decisions.

---

### 5. Transparent Business Impact Modeling
Rather than displaying a single static number based on fixed assumptions, the business impact calculator exposes:
1. **Editable Traffic Input:** Monthly checkout volume $V_{\text{monthly}}$ (default $5,000,000$).
2. **Side-by-Side Formula:**
   $$\Delta \text{Annual Revenue} = 12 \times V_{\text{monthly}} \times \Delta \text{RPU}$$
3. **Range Based on 95% Confidence Interval of Incremental RPU ($\Delta \text{RPU} = +\$0.88$, 95% CI: $[+\$0.29, +\$1.47]$):**
   - **Point Estimate:** $+60\text{M} \times \$0.88 = \mathbf{+\$52.7\text{M} / \text{year}}$
   - **Conservative Bound (95% CI Low):** $+60\text{M} \times \$0.29 = \mathbf{+\$17.3\text{M} / \text{year}}$
   - **Optimistic Bound (95% CI High):** $+60\text{M} \times \$1.47 = \mathbf{+\$88.2\text{M} / \text{year}}$


## Resume Bullet Points

```text
E-Commerce A/B Testing & Funnel Analytics Platform | Python, Streamlit, Plotly, Statsmodels
• Architected an end-to-end A/B test analysis framework evaluating a 50K-user checkout redesign across a 5-stage funnel.
• Engineered a 6-point pre-analysis telemetry audit covering Sample Ratio Mismatch (SRM), cross-variant contamination, and funnel sequence monotonicity.
• Implemented two-proportion Z-tests, Welch's t-tests, and formal guardrail Non-Inferiority testing for Average Order Value (AOV) against a business-approved decline margin.
• Estimated logistic regression interaction models to statistically confirm Heterogeneous Treatment Effects (HTE, p < 0.0001), uncovering significant power-user friction (-7.7% lift).
• Modeled transparent annualized revenue impact using RPU confidence bounds (+$17.3M to +$88.2M/yr) and authored a 3-phase quality-gated rollout architecture.
```


