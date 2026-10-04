# Changelog

## 0.1.0a2 — 2026-10-04

- Correct scikit-learn transformer inheritance and tags for numerical, missing,
  and binary-target inputs. Failed refits now invalidate prior learned state.
- Compute conventional XIRR using log present values, avoiding incorrect returns
  caused by intermediate overflow/underflow and the old fixed search limit.
- Preserve representable XNPV values when an intermediate discount factor
  overflows or underflows.
- Preserve simulation summaries when an annualized return cannot be represented;
  expose `return_status` and a specific `UnrepresentableReturnError` from XIRR.
- Test the declared minimum dependency versions as well as current versions.
- Visual README with an original SVG identity and reproducible financing,
  native-binning, and score-scale charts.
- A first-run guide, two walkthroughs, input recipes, glossary, and executed notebooks.
- Searchable MkDocs documentation with light/dark themes and copyable code.
- Documentation example and notebook execution checks in CI.

## 0.1.0a1 — 2026-10-04

Initial alpha: native constrained numerical/binary binning, multi-variable WoE
transformation, logistic scorecards and PDO scaling; point-in-time cash-flow
features; revenue-linked repayment with floors/milestones; term-loan schedules;
deterministic stress scenarios; credit diagnostics; PSI and dated returns.

The optimization core adapts selected OptBinning code under Apache 2.0, with
source attribution and documented solver changes. Runtime dependencies do not
include OptBinning. All documentation and examples use English.

Includes independently checkable numerical tests, CI, synthetic examples,
English project/research/convention documents, and wheel/source packaging.
Distributed through a GitHub version tag; PyPI publication is separate.
