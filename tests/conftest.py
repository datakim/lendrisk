import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def daily_path():
    def make(days=10, revenue=100.0, cost=60.0, start="2025-01-01"):
        frame = pd.DataFrame(
            {
                "date": pd.date_range(start, periods=days),
                "revenue": np.full(days, revenue, dtype=float),
            }
        )
        if cost is not None:
            frame["operating_cost"] = float(cost)
        return frame

    return make
