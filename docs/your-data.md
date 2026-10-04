# Bring your own data

Choose the recipe that matches your task. The package accepts pandas objects;
it does not connect to bank accounts, payment providers, or a remote service.

## Daily merchant cash flow

Use one row per calendar day for one merchant, in one consistent currency.
Example values below describe three consecutive days:

```csv
date,revenue,operating_cost
2026-01-01,1000,650
2026-01-02,900,640
2026-01-03,1100,680
```

| Column | Required? | Meaning |
| --- | --- | --- |
| `date` | Yes | Calendar date, without timezone or a time component |
| `revenue` | Yes | Nonnegative eligible daily revenue, finite and in currency units |
| `operating_cost` | For liquidity metrics | Nonnegative operating outflow; excludes the financing payment calculated by the simulator |

Load a file named `merchant_daily.csv`:

```python
import pandas as pd
from lendrisk import cashflow_features

daily = pd.read_csv("merchant_daily.csv", parse_dates=["date"])
features = cashflow_features(daily, as_of="2026-01-03", windows=(3,))
print(features)
```

This three-day example has revenue of 3,000, operating cash of 1,030, and full
coverage. The library's default 30/90-day windows need more history to obtain
full coverage. Growth needs two complete adjacent windows.

For financing, supply a **future daily path** separately from historical
observations. It may be a forecast or a named scenario, but the package does not
create that forecast for you:

```python
from lendrisk import RevenueAdvance

future = pd.read_csv("future_daily.csv", parse_dates=["date"])
result = RevenueAdvance(30_000, 1.12, 0.10).simulate(future, opening_cash=5_000)
print(result.summary())
```

Cash metrics are unknown without `operating_cost`. All scheduled payments are
assumed collected. [Read the financing tutorial](tutorials/revenue-financing.md).

### Multiple merchants

Add a `merchant_id` column. Dates can repeat across merchants, but not within one merchant.

```python
from lendrisk import merchant_features

panel = pd.read_csv("merchants_daily.csv", parse_dates=["date"])
features_by_merchant = merchant_features(panel, as_of="2026-06-30")
print(features_by_merchant.head())
```

Use a common cutoff date for a comparable cohort. Include enough observations
for the windows you plan to evaluate.

## Credit feature data

Use one row per observation with numerical feature columns and a binary target.
Example schema:

```csv
application_date,debt_ratio,revenue_cv,default_12m
2024-01-01,0.25,0.12,0
2024-01-02,0.72,0.41,1
```

The two example rows show the format only; a meaningful model needs a larger
cohort with both classes and enough observations per bin. `default_12m` is a
caller-defined label, not a definition imposed by the library.

```python
import pandas as pd
from lendrisk import LogisticScorecard, credit_metrics

credit = pd.read_csv("credit.csv", parse_dates=["application_date"])
columns = ["debt_ratio", "revenue_cv"]
train = credit[credit["application_date"] < "2024-07-01"]
test = credit[credit["application_date"] >= "2024-07-01"]
model = LogisticScorecard().fit(train[columns], train["default_12m"])
predicted = model.predict_proba(test[columns])[:, 1]
print(credit_metrics(test["default_12m"], predicted))
```

Only include features available at application time and observations with
fully observed outcome horizons. Use the same feature names and order during
prediction. Exclude identifiers, dates, and target columns from `X`.

## Resolve common input errors

| Symptom | What to check |
| --- | --- |
| Missing calendar days | Distinguish no-sales days from missing records. `cashflow_features(..., missing_days="zero")` explicitly treats gaps as zero revenue/cost; use it only if that matches the data. Simulation paths require every day. |
| Duplicate dates | Aggregate transactions to one daily record before calling the package. |
| NaN or infinite monetary values | Clean them before feature extraction or simulation. Zero revenue is valid; missing money is not zero by default. |
| Empty training cohort / only one class | Check the date split and label maturity; fitting needs both classes. |
| `BinningOptimizationError` | Inspect minimum counts/sizes, bin limits, and time limit. The solver reports infeasibility rather than quietly changing your constraints. |
| Feature-schema error at prediction | Keep the training column names and their exact order. |

[Calculation conventions](conventions.md) explains date, money, missing-value,
and solver behavior in detail.
