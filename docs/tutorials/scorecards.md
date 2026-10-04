# Turn numerical credit variables into explainable scores

This guide starts with one variable, then builds and evaluates a complete
scorecard. All data is synthetic. The first example gives a debt ratio a
deliberately increasing event probability so the binning behavior is easy to see.

## 1. Learn readable bins

```python
import numpy as np
from lendrisk import OptimalBinning

rng = np.random.default_rng(42)
debt_ratio = rng.uniform(0.05, 0.95, 1_000)
default = rng.binomial(1, 0.02 + 0.42 * debt_ratio**2)
bins = OptimalBinning(max_n_bins=5, monotonic_trend="ascending").fit(debt_ratio, default)
print(bins.status_)
print(bins.table()[["bin", "count", "event_rate", "woe"]].round(3))
```

![Candidate default rates, optimized monotonic bins, and weight of evidence.](../assets/native-binning.png)

- `max_n_bins=5` allows up to five regular bins; it does not force five.
- `monotonic_trend="ascending"` requires each next bin's default rate to be at least the previous rate.
- `event_rate` is the observed fraction of labels equal to 1 within each bin.
- **WoE** compares the bin's share of all non-defaults with its share of all defaults.
  Positive values indicate relative concentration of non-defaults; negative values indicate defaults.

`OPTIMAL` means the solver proved optimality in its candidate prebin search
space, at the documented objective precision. It does not mean the feature
or future model is perfect. A time-limited feasible solution is labeled
`FEASIBLE`; no feasible solution raises `BinningOptimizationError`.

Transform new numerical values with the learned boundaries:

```python
new_values = [0.10, 0.50, 0.90, np.nan]
print(bins.transform(new_values, metric="indices"))
print(bins.transform(new_values, metric="woe"))
```

Missing and configured special values have their own report buckets. If a
reserved bucket had no training observations, its WoE is neutral (zero).

## 2. Split the data before fitting a scorecard

This separate classification dataset contains four artificial numerical features.
The labels are about evenly balanced and have no specified real-world horizon;
they demonstrate the workflow rather than calibrated credit probabilities.

```python
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lendrisk import LogisticScorecard, credit_metrics

X, y = make_classification(n_samples=600, n_features=4, n_redundant=0, random_state=42)
X = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
model = LogisticScorecard(pdo=20, base_score=600, base_odds=50).fit(X_train, y_train)
pd_hat = model.predict_proba(X_test)[:, 1]
points = model.score_points(X_test)
print(credit_metrics(y_test, pd_hat))
print(pd.DataFrame({"model_pd": pd_hat, "score_points": points}).head().round(3))
```

The scorecard learns bins **inside `fit` on the training cohort**, transforms
those variables into WoE, and fits logistic regression. Evaluation applies the
learned transformation to the test cohort. Do not bin the entire dataset before
splitting. For a production credit problem, prefer a time-based validation cohort
that reflects the underwriting date and the outcome horizon.

Read the diagnostics by question:

| Diagnostic | Question answered |
| --- | --- |
| ROC AUC / Gini / KS | Does the score separate events from non-events? |
| Brier score / log loss | How close are predicted probabilities to observed labels? |
| Event rate vs. mean predicted PD | Is average predicted risk aligned with this cohort's observed event share? |

No single diagnostic establishes calibration or suitability. The API provides
calculated values; choose acceptance criteria for your population and task.

## 3. Understand the score scale

![The default point scale: 580, 600, and 620 points correspond to odds of 25, 50, and 100 to one.](../assets/score-scale.png)

At `base_score=600`, `base_odds=50` means 50 non-defaults per default: a model PD
of 1/51, about 1.96%. `pdo=20` means 20 extra points double those good:bad odds.
At 620 points the odds are 100:1 and model PD is about 0.99%.

This is a mathematical mapping of model probabilities. A conventional point
range or attractive score scale does not establish that predictions are calibrated.

## 4. Inspect the point contributions

```python
contributions = model.table()[["variable", "bin", "count", "woe", "points"]]
print(contributions.head(10).round(3))
print("Add this intercept once:", round(model.intercept_points, 3))
```

Each applicant selects one bin per input variable. The score is the sum of
those bin contributions plus the intercept once. The table gives the exact
unrounded calculation used by the model. It explains point construction;
adverse-action reason codes are outside the current alpha scope.

## 5. Bring a defined credit target

For your own data, use numerical feature columns and a binary target:
**1 means the event you define; 0 means non-event**. State the event horizon,
ensure labels have matured, and exclude information unavailable at application
time. Preserve the same feature names and column order at prediction time.

[Input recipe](../your-data.md#credit-feature-data) · [Glossary](../glossary.md) ·
[Upstream scope and differences](../upstream.md) ·
[Open the scorecard notebook](https://colab.research.google.com/github/datakim/lendrisk/blob/main/notebooks/02_native_scorecard.ipynb)
