"""Small arithmetic examples; probabilities here are manually supplied."""

from lendrisk import TermLoan, credit_metrics, expected_loss, population_stability_index


def main() -> None:
    print("Credit metrics:", credit_metrics([0, 0, 1, 1], [0.05, 0.20, 0.60, 0.85]))
    print("Expected loss by exposure:", expected_loss([0.05, 0.20], 0.45, [10_000, 20_000]))
    psi, table = population_stability_index(
        [0.05, 0.10, 0.20, 0.40], [0.10, 0.20, 0.35, 0.70], bins=[0.10, 0.25, 0.50]
    )
    print("\nPSI:", psi)
    print(table)
    print("\nTerm loan:")
    print(TermLoan(10_000, 0.12, 12).schedule(start_date="2025-01-31"))


if __name__ == "__main__":
    main()
