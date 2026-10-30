"""Loan default prediction pipeline (Asiimwe Brian Gita, MSc Data Science dissertation, 2025).

Usage:
    python loan_default.py generated_loans_rule.csv              # data from generate_data.py
    python loan_default.py data.csv --target loan_status         # any CSV with a binary target column
"""
import argparse

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from xgboost import XGBClassifier

SEED = 42
REPAYMENT_ORDER = {"monthly": 1, "irregular": 2, "bullet": 3}  # thesis 4.2.2: ordinal risk scale


def prepare(df, target, positive=None):
    df = df.dropna(subset=[target])  # unknown outcome is not 'repaid'
    y_raw = df[target]
    if y_raw.nunique() != 2:
        raise SystemExit(f"Target '{target}' must have exactly 2 values, found {y_raw.nunique()}: {y_raw.unique()[:10]}")
    if positive is None:
        positive = 1 if set(y_raw.unique()) == {0, 1} else y_raw.value_counts().idxmin()
    y = (y_raw.astype(str) == str(positive)).astype(int)
    X = df.drop(columns=[target])
    if "repayment_interval" in X:
        X["repayment_interval"] = X["repayment_interval"].str.lower().map(REPAYMENT_ORDER)
    # ponytail: drops ID-like columns (all values unique); free text such as loan_use needs NLP, not ordinal codes
    X = X.drop(columns=[c for c in X if X[c].nunique() == len(X)])
    return X, y, positive


def build_models(X):
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X if c not in num]
    # Fitted inside each pipeline, so imputation/scaling/encoding only ever see training data (thesis 3.3).
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), MinMaxScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), cat),
    ])
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "XGBoost": XGBClassifier(eval_metric="logloss", learning_rate=0.1, max_depth=3, n_estimators=100,
                                 subsample=0.8, colsample_bytree=0.8, random_state=SEED),
    }, pre


def evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    models, pre = build_models(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, cms = [], {}
    details = {"y_test": y_test, "probs": {}, "models": {}}
    for name, clf in models.items():
        model = make_pipeline(clone(pre), clf)
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_prob),
            "CV ROC-AUC": f"{cv_auc.mean():.3f} ± {cv_auc.std():.3f}",
        })
        cms[name] = confusion_matrix(y_test, y_pred)
        details["probs"][name] = y_prob
        details["models"][name] = model
    return pd.DataFrame(rows).set_index("Model").round(3), cms, details


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV file (e.g. from generate_data.py)")
    ap.add_argument("--target", default="default", help="binary target column (default: default)")
    ap.add_argument("--positive", help="target value meaning 'defaulted' (default: 1, else the minority class)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.target not in df:
        raise SystemExit(f"No column '{args.target}'. Columns: {list(df.columns)}. Pass --target.")

    X, y, positive = prepare(df, args.target, args.positive)
    print(f"Rows: {len(X)}  Features: {list(X.columns)}")
    print(f"Positive class '{positive}' (default) rate: {y.mean():.1%}\n")
    results, cms, _ = evaluate(X, y)
    print(results.to_string(), "\n")
    for name, cm in cms.items():
        (tn, fp), (fn, tp) = cm
        print(f"{name:20s} TN={tn:4d} FP={fp:4d} FN={fn:4d} TP={tp:4d}")


if __name__ == "__main__":
    main()
