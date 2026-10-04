# Research and design decisions

Reviewed on 2026-10-04. These are design references, not claims that no competing
package exists. Selected upstream binary model-data and solver setup code is adapted as documented in [provenance](upstream.md).

| Primary reference | Observed scope | Decision for lendrisk |
| --- | --- | --- |
| [OptBinning](https://gnpalencia.org/optbinning/) and [repository](https://github.com/guillermo-navas-palencia/optbinning) | Constrained optimal binning, scorecards, monitoring, explanations | Adapt a focused binary optimization core; maintain native binning and scorecards alongside cash-flow lending |
| [scorecardpy](https://github.com/ShichenXie/scorecardpy) | Scorecard development and WoE-related workflows | Keep conventional risk diagnostics available as small independent functions |
| [skorecard](https://github.com/ing-bank/skorecard) | Credit scorecard tooling connected to scikit-learn and OptBinning | Provide native estimators and feature tables for existing Python workflows |
| [NumPy Financial PMT](https://numpy.org/numpy-financial/latest/_api_stubs/numpy_financial.pmt.html) | Equal-period payment calculation with explicit period-rate conventions | State nominal annual rate / 12; test monthly payments against an independent reference |
| [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) | Classification and probabilistic prediction diagnostics | Compare AUC/Brier/log loss against upstream, document event direction and tied scores |
| [Stripe Capital](https://docs.stripe.com/capital/how-capital-for-platforms-works) | Revenue withholding, fixed fee, product-dependent periodic minimum payments | Separate revenue payments from block-end floor top-ups; avoid a universal MCA preset |
| [Shopify Capital US](https://help.shopify.com/en/manual/finance/shopify-capital/united-states) | Revenue-linked repayments and cumulative minimum-payment checkpoints | Check cumulative milestones separately from automatic payment floors |

MCA and RBF are supported through configurable payment mechanics. The package
does not assign legal classifications. Provider rules can change and vary by
product and jurisdiction; callers supply the agreement terms they are analyzing.

The initial differentiation is the explicit link between daily revenue,
operating costs, financing terms, and horizon-aware recovery/liquidity results.
This is a product direction chosen from the reviewed examples, not an exhaustive
market-gap finding.

The 0.1.0a2 numerical review follows the
[scikit-learn estimator inheritance and tag conventions](https://scikit-learn.org/stable/developers/develop.html).
The XIRR implementation retains the
[365-day dated-return convention](https://support.microsoft.com/en-us/excel/functions/xirr-function)
and uses [NumPy logaddexp](https://numpy.org/doc/stable/reference/generated/numpy.logaddexp.html)
to compare present values without exponentiating extreme discount factors.
Closed-form Decimal references and representability checks validate the change;
the algorithm does not copy Microsoft's iterative implementation.
