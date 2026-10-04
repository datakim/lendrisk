"""Contractual payment paths. Negative cash signals a liquidity shortfall, not default."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._validation import daily_frame, day, fraction, integer, number
from .returns import xirr


@dataclass(frozen=True)
class MinimumPayment:
    """Payment floor per complete block of calendar days, topped up at block end."""

    amount: float
    every_days: int = 30

    def __post_init__(self) -> None:
        number(self.amount, "amount", strict=True)
        integer(self.every_days, "every_days")


@dataclass(frozen=True)
class RepaymentMilestone:
    """Check cumulative payments against a fraction of the total receivable."""

    day: int
    cumulative_fraction: float

    def __post_init__(self) -> None:
        integer(self.day, "day")
        fraction(self.cumulative_fraction, "cumulative_fraction", positive=True)


@dataclass
class SimulationResult:
    """Payment schedule and explicit horizon status for one supplied scenario."""

    schedule: pd.DataFrame
    principal: float
    target_repayment: float
    funding_date: pd.Timestamp
    opening_cash: float | None = None

    def summary(self) -> dict[str, object]:
        """Report recovery and liquidity; IRR is only reported after full repayment."""
        paid = float(self.schedule["payment"].sum())
        remaining = float(self.schedule["remaining_balance"].iloc[-1])
        repaid = remaining == 0.0
        paid_rows = self.schedule.loc[self.schedule["payment"] > 0]
        payoff_date = paid_rows["date"].iloc[-1] if repaid else None
        annual_return = None
        if repaid:
            annual_return = xirr(
                [-self.principal, *paid_rows["payment"].tolist()],
                [self.funding_date, *paid_rows["date"].tolist()],
            )
        has_cash = self.opening_cash is not None
        return {
            "principal": self.principal,
            "target_repayment": self.target_repayment,
            "total_paid": paid,
            "remaining_balance": remaining,
            "repaid": repaid,
            "payoff_date": payoff_date,
            "payoff_days": (payoff_date - self.funding_date).days if repaid else None,
            "horizon_days": (self.schedule["date"].iloc[-1] - self.funding_date).days,
            "recovery_ratio": paid / self.target_repayment,
            "effective_annual_return": annual_return,
            "minimum_cash_balance": float(self.schedule["cash_balance"].min())
            if has_cash
            else None,
            "negative_cash_days": int(self.schedule["cash_balance"].lt(0).sum())
            if has_cash
            else None,
            "total_minimum_topup": float(self.schedule["minimum_topup"].sum()),
            "milestones_observed": int(self.schedule["milestone_checked"].sum()),
            "milestones_breached": int(self.schedule["milestone_shortfall"].gt(0).sum()),
        }


@dataclass(frozen=True)
class RevenueAdvance:
    """Fixed receivable paid from a share of supplied daily eligible revenue.

    factor_rate is a repayment multiple: 1.10 means principal plus a 10% fixed
    fee. No interest accrues. Configure floors/milestones explicitly from the
    contract; the class does not infer a legal product type or forecast revenue.
    """

    principal: float
    factor_rate: float
    holdback_rate: float
    minimum_payment: MinimumPayment | None = None
    milestones: tuple[RepaymentMilestone, ...] = ()

    def __post_init__(self) -> None:
        number(self.principal, "principal", strict=True)
        number(self.factor_rate, "factor_rate", minimum=1)
        fraction(self.holdback_rate, "holdback_rate", positive=True)
        if not np.isfinite(self.principal * self.factor_rate):
            raise ValueError("total repayment exceeds floating-point range")
        if self.minimum_payment is not None and not isinstance(
            self.minimum_payment, MinimumPayment
        ):
            raise ValueError("minimum_payment must be a MinimumPayment")
        if any(not isinstance(m, RepaymentMilestone) for m in self.milestones):
            raise ValueError("milestones must contain RepaymentMilestone objects")
        ordered = sorted(self.milestones, key=lambda m: m.day)
        if len({m.day for m in ordered}) != len(ordered):
            raise ValueError("milestone days must be unique")
        if any(
            b.cumulative_fraction < a.cumulative_fraction
            for a, b in zip(ordered, ordered[1:], strict=False)
        ):
            raise ValueError("cumulative milestone fractions must not decrease")

    def simulate(self, data: pd.DataFrame, *, opening_cash: float = 0.0) -> SimulationResult:
        """Fund on the day before the path starts; assume scheduled payments execute.

        With operating_cost supplied, cash starts at opening_cash + principal,
        then changes by revenue - operating_cost - payment each day. Without
        costs, liquidity is unknown. The full path is retained after payoff.
        """
        opening_cash = number(opening_cash, "opening_cash", minimum=-np.inf)
        frame = daily_frame(data)
        target = float(self.principal * self.factor_rate)
        remaining, paid, block_paid = target, 0.0, 0.0
        cash = opening_cash + self.principal
        thresholds = {m.day: target * m.cumulative_fraction for m in self.milestones}
        rows = []
        for elapsed, record in enumerate(frame.to_dict("records"), start=1):
            withheld = min(record["revenue"] * self.holdback_rate, remaining)
            topup = 0.0
            block_paid += withheld
            if self.minimum_payment is not None and elapsed % self.minimum_payment.every_days == 0:
                topup = min(
                    max(self.minimum_payment.amount - block_paid, 0.0), remaining - withheld
                )
                block_paid = 0.0
            payment = withheld + topup
            remaining = max(remaining - payment, 0.0)
            if remaining <= target * 1e-12:
                payment += remaining
                withheld += remaining
                remaining = 0.0
            paid += payment
            has_cost = "operating_cost" in record
            net = record["revenue"] - record["operating_cost"] if has_cost else np.nan
            if has_cost:
                cash += net - payment
            rows.append(
                {
                    "date": record["date"],
                    "revenue": record["revenue"],
                    "operating_cost": record.get("operating_cost", np.nan),
                    "net_operating_cash": net,
                    "revenue_payment": withheld,
                    "minimum_topup": topup,
                    "payment": payment,
                    "cumulative_paid": paid,
                    "remaining_balance": remaining,
                    "cash_balance": cash if has_cost else np.nan,
                    "milestone_checked": elapsed in thresholds,
                    "milestone_shortfall": max(thresholds.get(elapsed, 0.0) - paid, 0.0),
                }
            )
        return SimulationResult(
            pd.DataFrame(rows),
            float(self.principal),
            target,
            frame["date"].iloc[0] - pd.Timedelta(days=1),
            opening_cash if "operating_cost" in frame else None,
        )


@dataclass(frozen=True)
class TermLoan:
    """Equal end-of-month installments at nominal annual_rate / 12, no extra fees."""

    principal: float
    annual_rate: float
    term_months: int

    def __post_init__(self) -> None:
        number(self.principal, "principal", strict=True)
        number(self.annual_rate, "annual_rate")
        integer(self.term_months, "term_months")

    @property
    def monthly_payment(self) -> float:
        rate = self.annual_rate / 12.0
        if rate == 0:
            return self.principal / self.term_months
        denominator = -np.expm1(-self.term_months * np.log1p(rate))
        return float(self.principal * rate / denominator)

    def schedule(self, *, start_date: object) -> pd.DataFrame:
        """Fund on start_date; each due date is anchored to that original date."""
        start = day(start_date, "start_date")
        balance = float(self.principal)
        rows = []
        for month in range(1, self.term_months + 1):
            interest = balance * self.annual_rate / 12.0
            principal_paid = (
                balance
                if month == self.term_months
                else min(self.monthly_payment - interest, balance)
            )
            balance = max(balance - principal_paid, 0.0)
            rows.append(
                {
                    "date": start + pd.DateOffset(months=month),
                    "period": month,
                    "payment": interest + principal_paid,
                    "interest": interest,
                    "principal_paid": principal_paid,
                    "remaining_balance": balance,
                }
            )
        return pd.DataFrame(rows)
