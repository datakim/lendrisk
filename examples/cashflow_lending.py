"""Run: python examples/cashflow_lending.py (all data is synthetic)."""

from lendrisk import (
    RevenueAdvance,
    RevenueShock,
    cashflow_features,
    compare_scenarios,
    make_merchant_cashflows,
)


def main() -> None:
    data = make_merchant_cashflows(days=540, seed=42)
    history, scenario_path = data.iloc[:180], data.iloc[180:]
    print("Point-in-time features:")
    print(cashflow_features(history, as_of=history["date"].max()).round(4))
    advance = RevenueAdvance(principal=30_000, factor_rate=1.12, holdback_rate=0.10)
    print("\nSynthetic scenario comparison (not a probabilistic forecast):")
    report = compare_scenarios(
        advance,
        scenario_path,
        [RevenueShock("sales_down_20pct", 0.8), RevenueShock("sales_down_40pct", 0.6)],
        opening_cash=5_000,
    )
    print(report[["repaid", "payoff_days", "remaining_balance", "minimum_cash_balance"]])


if __name__ == "__main__":
    main()
