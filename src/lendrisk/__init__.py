"""Local, composable analytics for cash-flow lending and credit risk."""

from .cashflow import cashflow_features, merchant_features
from .datasets import make_merchant_cashflows
from .metrics import credit_metrics, expected_loss, population_stability_index
from .products import (
    MinimumPayment,
    RepaymentMilestone,
    RevenueAdvance,
    SimulationResult,
    TermLoan,
)
from .returns import xirr, xnpv
from .stress import RevenueShock, compare_scenarios

__version__ = "0.1.0a1"
__all__ = [
    "MinimumPayment",
    "RepaymentMilestone",
    "RevenueAdvance",
    "RevenueShock",
    "SimulationResult",
    "TermLoan",
    "cashflow_features",
    "compare_scenarios",
    "credit_metrics",
    "expected_loss",
    "make_merchant_cashflows",
    "merchant_features",
    "population_stability_index",
    "xirr",
    "xnpv",
]
