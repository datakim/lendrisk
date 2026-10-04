"""Native constrained bins and scorecard on synthetic classification data."""

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from lendrisk import LogisticScorecard, credit_metrics


def main():
    X, y = make_classification(n_samples=600, n_features=4, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=["x0", "x1", "x2", "x3"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    model = LogisticScorecard().fit(X_train, y_train)
    print(credit_metrics(y_test, model.predict_proba(X_test)[:, 1]))
    print("Score points:", model.score_points(X_test)[:5])
    print("Intercept points:", model.intercept_points)
    print(model.table())


if __name__ == "__main__":
    main()
