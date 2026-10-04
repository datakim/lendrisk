"""Optional integration on synthetic classification data, not a credit benchmark."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lendrisk import credit_metrics
from lendrisk.scorecard import make_scorecard


def main() -> None:
    X, y = make_classification(n_samples=600, n_features=4, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = make_scorecard(list(X.columns))
    model.fit(X_train, y_train)
    print(credit_metrics(y_test, model.predict_proba(X_test)[:, 1]))
    print(model.table(style="summary"))


if __name__ == "__main__":
    main()
