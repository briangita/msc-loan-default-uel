import numpy as np

from generate_data import make_dataset
from loan_default import evaluate, prepare


def auc(label):
    X, y, _ = prepare(make_dataset(label=label), "default")
    return evaluate(X, y)[0]["ROC-AUC"]


def test_learns_rule_labels():
    assert (auc("rule").drop("Majority baseline") > 0.7).all()


def test_cannot_learn_random_labels():
    assert (auc("random") < 0.65).all()


def test_keeps_continuous_drops_ids():
    df = make_dataset(n=200)
    df["income"] = df["loan_amount"] + np.linspace(0.1, 0.9, len(df))  # all unique decimals
    df["loan_id"] = range(len(df))
    X, _, _ = prepare(df, "default")
    assert "income" in X and "loan_id" not in X


if __name__ == "__main__":
    test_learns_rule_labels()
    test_cannot_learn_random_labels()
    test_keeps_continuous_drops_ids()
    print("ok")
