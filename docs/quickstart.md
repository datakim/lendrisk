# Your first financing comparison

**Question:** what happens to repayment and merchant cash if daily sales fall?
This example needs no dataset or account. It runs locally on synthetic data.

## 1. Install

Use Python 3.10 or newer, with Git available:

```bash
python -m pip install "git+https://github.com/datakim/lendrisk.git@v0.1.0a2"
```

If you use Jupyter, run the same command with `%pip` in a notebook cell so the
package is installed in that notebook's environment. PyPI publication is not
configured yet; use the GitHub tag or the release wheel.

## 2. Run

Copy the entire block into a Python file or notebook:

```python
from lendrisk import RevenueAdvance, RevenueShock, compare_scenarios, make_merchant_cashflows

data = make_merchant_cashflows(days=540, seed=42)
future = data.iloc[180:]
advance = RevenueAdvance(principal=30_000, factor_rate=1.12, holdback_rate=0.10)
report = compare_scenarios(
    advance,
    future,
    [RevenueShock("sales_down_20pct", 0.8), RevenueShock("sales_down_40pct", 0.6)],
    opening_cash=5_000,
)
print(report[["repaid", "payoff_days", "remaining_balance", "minimum_cash_balance"]].round(0))
```

## 3. Read the output

| Scenario | Repaid? | Payoff day | Remaining receivable | Minimum end-of-day cash |
| --- | --- | --- | ---: | ---: |
| `baseline` | True | 337 | 0 | 35,363 |
| `sales_down_20pct` | False | — | 4,755 | 35,151 |
| `sales_down_40pct` | False | — | 11,967 | −4,444 |

![The actual package outputs, plotted as remaining receivable and merchant cash.](assets/financing-scenarios.png)

- **Factor 1.12:** 30,000 funded creates a receivable of 33,600.
- **Holdback 0.10:** each day's payment is 10% of revenue, capped at the unpaid balance.
- **Opening cash 5,000:** funding adds 30,000, so merchant cash begins at 35,000.
  Each day adds revenue, subtracts operating costs, and subtracts financing payments.
- **Unpaid at day 360:** this horizon ends before full repayment; it is not an automatic default label.
- **Cash below zero:** the path needs more liquidity if all scheduled payments execute.

The generator supplies a synthetic path; it does not forecast a merchant's
sales. Revenue shocks retain the original cost path by default. A different
cost response may change the liquidity result substantially.

## 4. Take the next step

[Understand the daily schedule](tutorials/revenue-financing.md){ .md-button .md-button--primary }
[Load your own CSV](your-data.md){ .md-button }
[Build a scorecard](tutorials/scorecards.md){ .md-button }

Prefer a notebook? [Open the financing notebook in Colab](https://colab.research.google.com/github/datakim/lendrisk/blob/main/notebooks/01_revenue_financing.ipynb).
It includes the installation cell and charts.
