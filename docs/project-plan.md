# Project plan: lendrisk

Planning date: 2026-10-04. Implementation version: 0.1.0a1.

## Purpose and users

Provide reusable, installable Python tools for credit scoring and cash-flow
lending. Primary users are credit-model developers, fintech data scientists,
business-finance analysts, and researchers using notebooks or batch pipelines.

The package connects three questions: how to turn numerical variables into
constrained credit-model features; how to fit and explain a scorecard; and how
financing terms interact with a business's daily cash-flow path.

## Architecture

1. Maintain a native binary binning engine based on selected adapted OptBinning
   model-data code and a modified CP-SAT partition formulation.
2. Build native multi-variable WoE transformations and logistic scorecards on
   generic scikit-learn components, with explicit points scaling.
3. Keep cash-flow features, financing contracts, scenarios, and credit
   diagnostics in independent modules with pandas/NumPy outputs.
4. Execute all package calculations locally without service authentication.
5. Preserve upstream provenance and Apache 2.0 license requirements.
6. Write documentation, comments, examples, and errors in English.

## Implemented alpha scope

| Module | Inputs and outputs | Verification |
| --- | --- | --- |
| binning | Numerical feature + binary target → constrained intervals, WoE/IV table | Exhaustive small-problem oracle, upstream objective comparison, explicit infeasibility |
| scorecard | Named feature table + binary labels → PD and score points | Training-boundary fit, Pipeline integration, PDO scaling, point reconstruction |
| cashflow | Daily revenue/costs → point-in-time merchant features | Date cutoff, partial-window coverage, missing dates |
| products | Contract + revenue path → repayment schedule and summary | Payment caps, cash conservation, floors and milestones |
| stress | Baseline path + named shocks → comparison table | Input immutability, cost elasticity, fixed horizons |
| metrics/returns | Labels/predictions/distributions or dated cash flows → analytical diagnostics | Independent numerical references, ties, date conventions |

## Follow-on roadmap

These entries are planned and are not part of the current implementation.

| Priority | Extension | Entry criteria |
| --- | --- | --- |
| Next | Categorical binning, CART prebins, sample weights | Independent test cases and clear handling of unseen categories |
| Next | Advanced trend constraints and maximum bin event/nonevent counts | Solver-size benchmarks and optimality checks |
| Next | Out-of-time scorecard validation and stability reports | Defined feature cutoff and performance-window contracts |
| Later | Transaction aggregation, refunds, settlement delays, existing debt | Reproducible real-world data contracts |
| Later | Vintage/delinquency/recovery analytics | Explicit event dates, censoring and default definitions |
| Later | Product-term optimization and probabilistic recovery | Validated prediction models and documented objective functions |

## Release criteria

Publish source, tests, examples, research notes, and Apache 2.0 notices in
`datakim/lendrisk`. Pin an alpha tag for direct GitHub installation with pip.
Check supported Python versions in CI and verify a wheel in a clean environment
without OptBinning installed. PyPI publication is a separate account/project
configuration step.

The alpha makes no claim of full upstream feature parity or production credit
model performance. Scenario analysis uses caller-supplied paths; it does not
manufacture default labels, forecast probabilities, or automatic approvals.
