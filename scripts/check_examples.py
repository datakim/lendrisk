"""Execute documented Python blocks; supply synthetic files for the CSV recipes."""

import contextlib
import io
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from lendrisk import make_merchant_cashflows

ROOT = Path(__file__).resolve().parents[1]


def fixtures(directory: Path) -> None:
    pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "revenue": [1000, 900, 1100],
            "operating_cost": [650, 640, 680],
        }
    ).to_csv(directory / "merchant_daily.csv", index=False)
    make_merchant_cashflows(days=360).to_csv(directory / "future_daily.csv", index=False)
    panel = make_merchant_cashflows(days=181, start_date="2026-01-01")
    panel["merchant_id"] = "synthetic-merchant"
    panel.to_csv(directory / "merchants_daily.csv", index=False)
    rng = np.random.default_rng(42)
    ratio = rng.uniform(0.05, 0.95, 500)
    pd.DataFrame(
        {
            "application_date": pd.date_range("2024-01-01", periods=500),
            "debt_ratio": ratio,
            "revenue_cv": rng.uniform(0.1, 0.7, 500),
            "default_12m": rng.binomial(1, 0.1 + 0.3 * ratio),
        }
    ).to_csv(directory / "credit.csv", index=False)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = [
        ROOT / "README.md",
        ROOT / "docs/quickstart.md",
        ROOT / "docs/your-data.md",
        *sorted((ROOT / "docs/tutorials").glob("*.md")),
    ]
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="lendrisk-doc-examples-") as tmp:
        directory = Path(tmp)
        fixtures(directory)
        os.chdir(directory)
        try:
            for path in paths:
                blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
                scope = {"__name__": "__documentation_example__"}
                with contextlib.redirect_stdout(io.StringIO()):
                    for number, code in enumerate(blocks, start=1):
                        exec(compile(code, f"{path}:{number}", "exec"), scope)
                plt.close("all")
                print(f"{path.relative_to(ROOT)}: {len(blocks)} Python blocks executed")
        finally:
            os.chdir(original)


if __name__ == "__main__":
    main()
