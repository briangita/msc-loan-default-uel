from generate_data import make_dataset
from loan_default import evaluate, prepare


def auc(label):
    X, y, _ = prepare(make_dataset(label=label), "default")
    return evaluate(X, y)[0]["ROC-AUC"]


def test_learns_rule_labels():
    assert (auc("rule").drop("Majority baseline") > 0.7).all()


def test_cannot_learn_random_labels():
    assert (auc("random") < 0.65).all()


if __name__ == "__main__":
    test_learns_rule_labels()
    test_cannot_learn_random_labels()
    print("ok")
