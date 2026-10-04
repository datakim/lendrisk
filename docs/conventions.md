# Calculation and data conventions

## Daily data

`date` and `revenue` are required. `operating_cost` is optional. Use one row per
timezone-naive calendar day at midnight, including explicit zero-sales days.
Transactions must be aggregated before calling the library. Revenue and costs
must be finite and nonnegative. Select the eligible revenue basis for the
agreement yourself; refund processing, FX conversion, tax treatment, and payment
processor settlement delays are outside the alpha.

Rows are copied and sorted. Duplicates, NaN, infinity, and internal gaps raise
errors. Feature extraction can explicitly use `missing_days="zero"`, which
fills missing revenue **and supplied costs** with zero. Use this only when that
assumption is justified. Simulation always requires a complete supplied path.

`as_of` excludes later rows from feature values. A window of N days includes
as_of and its previous N-1 days. Windows can be partially observed; observed
counts and coverage expose that condition. There is no padding outside the
observed range. Growth requires two complete adjacent windows and a positive
previous total. Undefined ratios are NaN, including growth from zero revenue.
Revenue CV uses population standard deviation (ddof=0). Operating cash is the
supplied revenue minus supplied operating costs, not an accounting reconstruction.

## Revenue-linked financing

Funding happens the day before the first supplied daily row. For principal P,
repayment multiple F, and revenue share h, total receivable T = P × F and the
revenue payment is min(revenue × h, remaining receivable). F=1.12 denotes a 12%
fixed fee, not an annual interest rate. No interest accrues after funding.

A `MinimumPayment(A, D)` floor is evaluated at days D, 2D, ... measured from
funding. The top-up equals max(A minus payments in that block, 0), capped by the
remaining balance. An incomplete final block has no top-up. Blocks are fixed
calendar-day spans, not calendar months or business days.

A `RepaymentMilestone(day, fraction)` checks cumulative payments against
fraction × T on that exact day. It reports a shortfall without forcing payment.
Only milestones within the supplied horizon are observed. A breach does not
assign default status. Future checkpoints are not counted as passed.

Simulation assumes the calculated payments execute. Without operating costs,
cash-balance outputs are unknown. With costs, opening post-funding cash is
opening_cash + P and daily cash changes by revenue - operating_cost - payment.
opening_cash can reflect an existing deficit. Include inventory purchases or
use of proceeds in costs if relevant. Payments are not capped by available
cash; a negative balance identifies required liquidity under these assumptions.
All supplied dates are retained after payoff to show continuing business cash.

At the horizon, unrecovered balances remain outstanding. No terminal recovery
or estimated loss is fabricated. Floating-point residuals within 1e-12 of the
initial receivable are cleared; monetary values are not rounded to cents.

## Term loans and returns

Term loans have equal monthly end-of-period installments and nominal annual
interest divided by 12. Due dates are anchored to the original funding date:
January 31 → February month-end → March 31. The final installment clears the
remaining principal. Fees, holidays, late charges, prepayments, and actual-day
interest accrual are not included.

XNPV discounts at an effective annual rate with ACT/365 fixed year fractions.
XIRR supports a net initial outflow followed by later nonnegative inflows, so
the conventional cash-flow root is unique. Same-day flows are netted and sorted.
Multiple-sign-change cash flows raise rather than selecting an arbitrary root.
Simulation reports return only on full recovery; partial-recovery return is
None. This does not compute a legal or regulatory APR.

## Credit metrics and stability

Event=1 means default; higher probability means higher risk. AUC uses average
ranks for ties. KS evaluates cumulative distributions at distinct score
thresholds, so tied observations cannot be separated to inflate KS. AUC/Gini/KS
are NaN for single-class cohorts. Brier is binary mean squared error, not a
standalone calibration-only measure. Log loss clips endpoints for numerical
evaluation. Metrics require finite probabilities in [0,1]. No sample weights
are supported in this alpha.

PSI fits quantile cuts on the reference sample and uses those fixed cuts for
the actual sample. Exterior bins extend to infinity. Explicit bins are finite
interior cutpoints. Counts are smoothed with an additive pseudocount (default
0.5), then normalized within each sample. The returned table uses NumPy
histogram intervals [lower, upper), with the final upper boundary inclusive.
Repeated quantiles collapse; a constant reference has a narrow center and two
tails. PSI does not provide a universal significance or alarm threshold.

`expected_loss` computes elementwise PD × LGD × EAD with broadcasting and no
discounting, maturity adjustment, regulatory capital, or IFRS 9 staging. PD/LGD
must be user-supplied fractions in [0,1]; no model estimates them implicitly.

## Scope

Scenario paths have no assigned probability. Synthetic data is illustrative.
Package validation verifies contracts and arithmetic, not model suitability
for a lending population. Trained models, approval rules, and production
servicing are separate from the analytical utilities delivered in this alpha.

## Native binary binning and scorecards

The native binning engine accepts numerical variables and binary event labels.
Candidate boundaries come from reference-training quantiles or explicit
user_splits. It maximizes regular-observation IV over contiguous prebins, subject
to configured size/class-count constraints and optional monotonicity. Report
Missing/Special buckets do not count toward the constrained regular-bin count.
At least one event and nonevent are required in each selected regular bin.

IV objective contributions are rounded at 1e8 scale for CP-SAT. OPTIMAL refers
to that integer objective over the specified candidate search space. The
absolute objective_gap_ refers to this quantized IV objective, not a population
risk guarantee. Auto trend solves both directions independently; it reports
FEASIBLE if an alternative direction has not been proven optimal/infeasible.
If neither solve returns a feasible partition, fitting raises an explicit error.

Reporting and transformation use additive smoothing, default 0.5, across
occupied final regular/missing/special buckets. WoE = ln(non-event share/event
share). Empty reserved buckets receive neutral WoE=0 and overall training event
rate. Smoothed table IV differs from the unsmoothed regular optimization iv_.
Intervals are left-closed/right-open, with unbounded exterior intervals. Missing
values and special codes retain their own indices at transform time.

BinningProcess fits independent native bins per numerical DataFrame column and
returns a WoE DataFrame. LogisticScorecard fits those bins and logistic regression
inside its fit call, so the caller must pass only training data. Higher score
points represent lower risk. base_odds is good:bad odds at base_score, and pdo
points doubles those odds. Per-bin point contributions plus intercept_points
reconstruct the unrounded score. All fitted schemas require the same column
names and order at prediction time.
