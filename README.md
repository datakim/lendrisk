# lendrisk

**Cash-flow features, lending simulations, and credit-risk diagnostics for Python.**

[한국어 안내](README.ko.md) · [Project plan](docs/project-plan.md) · [Calculation conventions](docs/conventions.md) · [API reference](docs/api.md)

`lendrisk` is an installable, local Python library for analysts working on credit
scoring, small-business lending, merchant cash advances (MCA), and revenue-based
financing (RBF). It connects daily business cash flows to contractual payment
paths and model diagnostics. Data stays in your Python process; package calls
do not contact a service or require an account.

Version **0.1.0a1** is an initial alpha. Its main use case is comparing financing
terms against explicitly supplied revenue scenarios. It also works alongside
OptBinning through an optional adapter.

## Install

Install the tagged alpha directly from GitHub:

```bash
python -m pip install "git+https://github.com/datakim/lendrisk.git@v0.1.0a1"
```

For development:

```bash
git clone https://github.com/datakim/lendrisk.git
cd lendrisk
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
```

Python 3.10+ is supported. Core dependencies are NumPy and pandas. A PyPI release
has **not** been published yet; `pip install lendrisk` will become available
after that separate release step.

## Quick start

```python
from lendrisk import (
    RevenueAdvance,
    RevenueShock,
    cashflow_features,
    compare_scenarios,
    make_merchant_cashflows,
)

# Entirely synthetic data. Future rows are an illustrative scenario,
# not a revenue forecast fitted by lendrisk.
data = make_merchant_cashflows(days=540, seed=42)
history, future = data.iloc[:180], data.iloc[180:]

features = cashflow_features(history, as_of=history["date"].max())
advance = RevenueAdvance(
    principal=30_000,
    factor_rate=1.12,  # Total contractual repayment = 33,600
    holdback_rate=0.10,  # Withhold 10% of eligible daily revenue
)
result = advance.simulate(future, opening_cash=5_000)
print(result.summary())
print(result.schedule.head())

comparison = compare_scenarios(
    advance,
    future,
    [
        RevenueShock("sales_down_20pct", revenue_multiplier=0.8),
        RevenueShock("sales_down_40pct", revenue_multiplier=0.6),
    ],
    opening_cash=5_000,
)
print(comparison[["repaid", "payoff_days", "remaining_balance", "minimum_cash_balance"]])
```

## Included in the alpha

| Module | What it provides |
| --- | --- |
| Cash flow | Point-in-time 30/90-day features, growth, volatility, concentration, coverage, operating margins, merchant panels |
| Revenue-linked financing | Fixed fee / repayment multiple, revenue holdback, periodic minimum payment top-ups, cumulative payment milestones |
| Term loans | Equal monthly installment schedules with explicit nominal-rate conventions |
| Stress analysis | Named revenue shocks, date intervals, fixed/variable cost assumptions, baseline comparisons |
| Credit diagnostics | Tie-aware AUC/Gini/KS, Brier score, log loss, PSI tables, elementwise PD × LGD × EAD |
| Dated returns | ACT/365 fixed XNPV and conventional-cash-flow XIRR |
| Examples | Seeded synthetic merchant paths, credit diagnostics, optional OptBinning scorecard |

All outputs use ordinary pandas objects or documented result objects. Monetary
values use one consistent currency unit chosen by the caller. The alpha uses
floating-point analytics without currency rounding; it does not provide a
settlement ledger.

## Minimum payments and milestones

These are distinct contract provisions. A payment floor triggers a top-up at
the end of each complete calendar-day block. A milestone checks progress and
reports a shortfall without adding an automatic payment.

```python
from lendrisk import MinimumPayment, RepaymentMilestone, RevenueAdvance

advance = RevenueAdvance(
    principal=30_000,
    factor_rate=1.12,
    holdback_rate=0.10,
    minimum_payment=MinimumPayment(amount=2_000, every_days=30),
    milestones=(RepaymentMilestone(day=180, cumulative_fraction=0.30),),
)
```

Configure these from the actual agreement. The object does not classify a
contract's legal type. The first daily observation is the day after funding.
No minimum top-up is charged for an incomplete final block. Simulation assumes
scheduled payments execute, even if the resulting business cash balance turns
negative. That negative balance is a liquidity signal, not a simulated default
or a validated estimate of default probability.

## Using an existing credit model

```python
from lendrisk import credit_metrics, expected_loss, population_stability_index

metrics = credit_metrics([0, 0, 1, 1], [0.05, 0.20, 0.60, 0.85])
losses = expected_loss(pd_hat=[0.05, 0.20], lgd=0.45, ead=[10_000, 20_000])
psi, table = population_stability_index(
    expected=[0.05, 0.10, 0.20, 0.40],
    actual=[0.10, 0.20, 0.35, 0.70],
    bins=[0.10, 0.25, 0.50],
)
```

For optional optimal binning / scorecard modeling:

```bash
python -m pip install -e ".[scorecard]"
```

```python
from lendrisk.scorecard import make_scorecard

scorecard = make_scorecard(variable_names=list(X_train.columns))
scorecard.fit(X_train, y_train)
probability_of_default = scorecard.predict_proba(X_test)[:, 1]
```

The adapter returns an upstream OptBinning `Scorecard`. Keep feature extraction,
binning, fitting, and validation within the appropriate point-in-time and
training boundaries. The alpha includes analytical utilities, not pretrained
credit scores or automatic approval policies.

## Examples and verification

```bash
python examples/cashflow_lending.py
python examples/credit_diagnostics.py
python -m pytest --cov=lendrisk --cov-report=term-missing
ruff check .
ruff format --check .
mkdocs build --strict
python -m build
```

Optional scorecard example: `python examples/optbinning_scorecard.py` after
installing the `scorecard` extra.

Tests include closed-form payment cases, cash conservation, horizon censoring,
missing-data and date handling, tied-score diagnostics, and comparisons against
NumPy Financial and scikit-learn. See [CONTRIBUTING.md](CONTRIBUTING.md) for
development and release steps.

## References

Design research draws on [OptBinning](https://gnpalencia.org/optbinning/),
[scorecardpy](https://github.com/ShichenXie/scorecardpy),
[skorecard](https://github.com/ing-bank/skorecard),
[NumPy Financial](https://numpy.org/numpy-financial/latest/), and
[scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html).
Payment-mechanics research uses
[Stripe Capital documentation](https://docs.stripe.com/capital/how-capital-for-platforms-works)
and [Shopify Capital documentation](https://help.shopify.com/en/manual/finance/shopify-capital/united-states).
The implementation is original; the optional scorecard delegates to OptBinning.
See [research notes](docs/research.md) for the decisions these references inform.

MIT licensed. Contributions with a concrete use case, documented assumptions,
and independently checkable numerical examples are welcome.
