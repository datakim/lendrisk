# API reference

## Cash-flow features

`cashflow_features(data, *, as_of=None, windows=(30, 90), missing_days="raise")`
returns a float pandas Series for one merchant. Input columns are `date`,
`revenue`, optionally `operating_cost`. See [conventions](conventions.md).

Each window prefixes columns with `30d_`, `90d_`, etc. Columns: `observed_days`,
`coverage`, `revenue_sum`, `revenue_mean`, `revenue_cv`, `zero_revenue_share`,
`top_day_share`, `revenue_growth`; with costs, also `operating_cash_sum`,
`operating_margin`, `negative_operating_cash_share`.

`merchant_features(data, *, merchant_column="merchant_id", **kwargs)` returns
one row per merchant indexed by its ID. Keyword arguments go to
`cashflow_features`. For comparable cohorts, pass a common as_of date.

## Products

`RevenueAdvance(principal, factor_rate, holdback_rate, minimum_payment=None,
milestones=())` creates an immutable contract specification. Principal > 0,
factor_rate >= 1, holdback_rate in (0,1].

`MinimumPayment(amount, every_days=30)` configures a positive periodic floor.
`RepaymentMilestone(day, cumulative_fraction)` configures a positive day number
and fraction in (0,1]. Milestone days must be unique and cumulative fractions
nondecreasing.

`advance.simulate(data, *, opening_cash=0.0)` returns `SimulationResult`.
Its `schedule` contains date, revenue, operating_cost, net_operating_cash,
revenue_payment, minimum_topup, payment, cumulative_paid, remaining_balance,
cash_balance, milestone_checked, and milestone_shortfall.

`result.summary()` returns principal, target_repayment, total_paid,
remaining_balance, repaid, payoff_date, payoff_days, horizon_days,
recovery_ratio, effective_annual_return, minimum_cash_balance,
negative_cash_days, total_minimum_topup, milestones_observed, and
milestones_breached. Missing/unobserved payoff and return are None; cash
metrics are None without cost data.

`TermLoan(principal, annual_rate, term_months)` exposes `monthly_payment` and
`schedule(start_date=...)`. The schedule contains date, period, payment,
interest, principal_paid, remaining_balance.

## Stress

`RevenueShock(name, revenue_multiplier, start_date=None, end_date=None,
cost_elasticity=0.0)` represents a deterministic scenario. Multipliers >= 0,
cost elasticity in [0,1]. Date boundaries are inclusive.
`shock.apply(data)` returns a new daily DataFrame.

`compare_scenarios(product, data, scenarios, *, opening_cash=0.0)` returns a
DataFrame of simulation summaries indexed by scenario name, including
`baseline`. Names must be unique and cannot equal `baseline`.

## Credit diagnostics

`credit_metrics(y_true, pd_hat)` returns n_observations, event_rate,
mean_predicted_pd, roc_auc, gini, ks, brier_score, log_loss.

`expected_loss(pd_hat, lgd, ead)` returns a float or broadcast NumPy array.

`population_stability_index(expected, actual, *, bins=10, pseudocount=0.5)`
returns `(psi, table)`. Integer bins >= 2; explicit bins are strictly increasing
interior cutpoints. Table columns: lower, upper, expected_count, actual_count,
expected_share, actual_share, psi_contribution.

## Dated returns

`xnpv(rate, amounts, dates)` requires effective annual rate > -1.
`xirr(amounts, dates)` requires conventional dated cash flows. Both use ACT/365.

## Data and integration

`make_merchant_cashflows(days=365, start_date="2025-01-01", daily_revenue=1000,
operating_cost_ratio=0.65, volatility=0.2, seed=42)` returns illustrative daily
synthetic data. Volatility is the lognormal noise sigma, bounded at 5.

`lendrisk.scorecard.make_scorecard(variable_names, *, estimator=None, **kwargs)`
returns an unfitted upstream OptBinning Scorecard. Requires the scorecard extra.
Default estimator: scikit-learn LogisticRegression(max_iter=1000). Additional
keywords go to Scorecard. See [upstream API](https://gnpalencia.org/optbinning/scorecard.html).
