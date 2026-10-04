![lendrisk: Understand credit. Follow the cash.](assets/banner.svg)

# Credit risk tools that follow the cash

**lendrisk** is an installable Python toolkit for explainable credit scoring,
merchant cash-flow analysis, and revenue-linked financing. Work with pandas
DataFrames, inspect every calculation, and keep your workflow in Python.

[Try the quickstart](quickstart.md){ .md-button .md-button--primary }
[Use your own data](your-data.md){ .md-button }

## Start with a question

| Your question | Follow this guide |
| --- | --- |
| What changes if a merchant's sales fall? | [Revenue financing](tutorials/revenue-financing.md) |
| How do I turn numerical variables into credit scores? | [Binning and scorecards](tutorials/scorecards.md) |
| What input columns and dates do I need? | [Bring your own data](your-data.md) |
| What do PD, WoE, factor rate, and holdback mean? | [Plain-language glossary](glossary.md) |

## See the financing outcome

![Remaining receivable and merchant cash across baseline and sales declines of 20% and 40%.](assets/financing-scenarios.png)

A 30,000 advance with a 1.12 repayment factor and a 10% daily revenue share pays
off on day 337 of the baseline path. A 40% sales decline leaves 11,967 unpaid at
day 360 and produces a negative cash balance. These are deterministic simulations
of **synthetic** revenue paths; operating costs are unchanged by the shocks.

The [quickstart](quickstart.md) reproduces this result. The
[financing tutorial](tutorials/revenue-financing.md) explains the contract,
daily schedule, and interpretation.

## Build a readable credit score

![Numerical candidate bins become monotonic default-rate groups with weight of evidence.](assets/native-binning.png)

Native optimal binning turns numerical inputs into risk groups. A logistic
scorecard then returns default probabilities and score points. Fit on the
training cohort and evaluate on a separate cohort; the chart shows training data
from a synthetic example.

[Follow the scorecard tutorial](tutorials/scorecards.md){ .md-button }
[Open the notebooks](https://github.com/datakim/lendrisk/tree/main/notebooks){ .md-button }

## Know the current scope

The alpha supports numerical features and binary credit targets, daily cash-flow
features, revenue-linked repayments with explicit floors and milestones,
installment schedules, stress comparisons, credit metrics, PSI, and dated returns.

The native binning core adapts selected Apache-licensed OptBinning source.
[Provenance](upstream.md) documents the changes. [Conventions](conventions.md)
explains the calculation assumptions. Use the [API reference](api.md) once you
know which workflow you need.
