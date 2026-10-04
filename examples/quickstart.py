"""The README example: a 360-day revenue-financing stress comparison."""

from lendrisk import RevenueAdvance, RevenueShock, compare_scenarios, make_merchant_cashflows


def main() -> None:
    data = make_merchant_cashflows(days=540, seed=42)
    future = data.iloc[180:]  # A supplied synthetic path, following 180 historical days.
    advance = RevenueAdvance(principal=30_000, factor_rate=1.12, holdback_rate=0.10)
    report = compare_scenarios(
        advance,
        future,
        [RevenueShock("sales_down_20pct", 0.8), RevenueShock("sales_down_40pct", 0.6)],
        opening_cash=5_000,
    )
    print(report[["repaid", "payoff_days", "remaining_balance", "minimum_cash_balance"]].round(0))


if __name__ == "__main__":
    main()
