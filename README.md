# lendrisk

**Native credit binning, scorecards, and cash-flow lending analytics for Python.**

[Project plan](docs/project-plan.md) · [API reference](docs/api.md) · [Calculation conventions](docs/conventions.md) · [Upstream provenance](docs/upstream.md)

`lendrisk` is an installable Python library for credit-model developers and
analysts working on business loans, merchant cash advances (MCA), and
revenue-based financing (RBF). It includes its own constrained binary binning
engine, WoE transformation, logistic scorecards, and revenue-linked repayment
analysis. Package calls execute locally without sending data to a service.

The binary optimization core adapts selected OptBinning code and changes the
solver formulation. See [NOTICE](NOTICE) and [provenance](docs/upstream.md).
OptBinning is not a runtime dependency. NumPy, pandas, scikit-learn, and OR-Tools
provide numerical operations, data structures, logistic regression, and CP-SAT.

Version **0.1.0a1** is an initial alpha with numerical features and binary
classification targets. Categorical/continuous/multiclass binning, advanced
trend shapes, sample weights, and regulatory approval policies are future work.

## Install

```bash
python -m pip install "git+https://github.com/datakim/lendrisk.git@v0.1.0a1"
```

Python 3.10+ is supported. PyPI publication has not been configured yet;
`pip install lendrisk` will become available after that separate release step.

For development:

```bash
git clone https://github.com/datakim/lendrisk.git
cd lendrisk
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,docs]"
```

## Native optimal binning

```python
from lendrisk import OptimalBinning

x = [10, 10, 20, 20, 30, 30, 40, 40]
y = [0, 0, 0, 1, 0, 1, 1, 1]  # Event/default = 1

binning = OptimalBinning(
    max_n_prebins=4,
    max_n_bins=3,
    monotonic_trend="ascending",
).fit(x, y)

print(binning.status_)
print(binning.splits_)
print(binning.table())
woe = binning.transform([15, 25, 35])
```

The optimizer maximizes IV over contiguous prebins subject to bin-count,
size, event/nonevent count, monotonicity, and event-rate-gap constraints.
Quantile or supplied prebin boundaries define the search space. Missing and
special codes have dedicated buckets. Infeasible or timed-out solves without
a feasible partition raise an explicit error rather than returning a fallback.

## Native scorecards

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lendrisk import LogisticScorecard, credit_metrics

# Synthetic classification example, not a credit performance benchmark.
X, y = make_classification(n_samples=600, n_features=4, n_redundant=0, random_state=42)
X = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

model = LogisticScorecard(pdo=20, base_score=600, base_odds=50).fit(X_train, y_train)
pd_hat = model.predict_proba(X_test)[:, 1]
points = model.score_points(X_test)
print(credit_metrics(y_test, pd_hat))
print(model.table())
```

Higher points mean lower default risk. `base_odds` is the good:bad odds at
`base_score`; adding `pdo` points doubles those odds. A `BinningProcess` can
also be used inside a scikit-learn Pipeline. Fit bins and models exclusively
on the training partition; supply point-in-time feature data.

## Cash-flow lending and stress analysis

```python
from lendrisk import (
    RevenueAdvance,
    RevenueShock,
    cashflow_features,
    compare_scenarios,
    make_merchant_cashflows,
)

# Future rows are a supplied synthetic scenario, not a fitted forecast.
data = make_merchant_cashflows(days=540, seed=42)
history, future = data.iloc[:180], data.iloc[180:]
features = cashflow_features(history, as_of=history["date"].max())

advance = RevenueAdvance(principal=30_000, factor_rate=1.12, holdback_rate=0.10)
result = advance.simulate(future, opening_cash=5_000)
print(result.summary())

comparison = compare_scenarios(
    advance,
    future,
    [RevenueShock("sales_down_20pct", 0.8), RevenueShock("sales_down_40pct", 0.6)],
    opening_cash=5_000,
)
print(comparison[["repaid", "payoff_days", "remaining_balance", "minimum_cash_balance"]])
```

Daily DataFrames require `date`, `revenue`, and optionally `operating_cost`.
Use one row per calendar day, including explicit zero-sales days. With costs,
post-funding cash starts at opening_cash + principal. Without costs, liquidity
metrics remain unknown. Simulations assume scheduled payments execute; a
negative cash balance indicates required liquidity under the supplied scenario.
Unrecovered balances remain outstanding at the end of the supplied horizon.

`MinimumPayment(amount, every_days)` configures a block-end payment top-up;
`RepaymentMilestone(day, cumulative_fraction)` checks progress without forcing
a payment. Configure contract terms explicitly. A financing object's legal
classification is not inferred. Floating-point analytical outputs do not apply
currency rounding or provide a settlement ledger.

## Included modules

| Module | Functions and classes |
| --- | --- |
| Binning | `OptimalBinning`, `BinningProcess`, WoE/IV tables, solver status and objective gap |
| Scorecards | `LogisticScorecard`, probabilities, PDO-scaled points, per-bin point contributions |
| Cash flow | Point-in-time features, growth, volatility, coverage, operating margins, merchant panels |
| Products | `RevenueAdvance`, payment floors/milestones, monthly `TermLoan` schedules |
| Stress | Named revenue shocks, date intervals, fixed/variable cost assumptions |
| Diagnostics | AUC/Gini/KS, Brier/log loss, PSI tables, elementwise PD × LGD × EAD |
| Returns | ACT/365 fixed XNPV/XIRR for conventional cash flows |

## Verification

```bash
python examples/native_scorecard.py
python examples/cashflow_lending.py
python examples/credit_diagnostics.py
python -m pytest --cov=lendrisk --cov-report=term-missing
ruff check .
ruff format --check .
mkdocs build --strict
python -m build
python -m twine check dist/*
```

Tests independently enumerate small binning problems, compare shared-search-space
IV with upstream OptBinning when installed, check monotonic constraints and score
scaling, and validate repayment invariants and diagnostics against reference
implementations. Optional upstream comparison: install `.[dev,reference]`.

## References and license

Reviewed references: [OptBinning](https://github.com/guillermo-navas-palencia/optbinning),
[scorecardpy](https://github.com/ShichenXie/scorecardpy),
[skorecard](https://github.com/ing-bank/skorecard),
[NumPy Financial](https://numpy.org/numpy-financial/latest/),
[scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html),
[Stripe Capital](https://docs.stripe.com/capital/how-capital-for-platforms-works),
and [Shopify Capital](https://help.shopify.com/en/manual/finance/shopify-capital/united-states).

Apache License 2.0. Adapted upstream code retains author attribution in
[NOTICE](NOTICE). Contributions should include a concrete analytical use case,
explicit assumptions, and independently checkable numerical examples.
