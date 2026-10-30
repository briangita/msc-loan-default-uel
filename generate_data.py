"""Generate a Kiva-style synthetic loan dataset for loan_default.py.

Usage:
    python generate_data.py                  # rule-based labels -> generated_loans_rule.csv
    python generate_data.py --label random   # labels independent of features (sanity check: AUC ~0.5)
"""
import argparse

import numpy as np
import pandas as pd

SEED = 42


def make_dataset(n=1000, label="rule", seed=SEED):
    """Kiva-style loans for Uganda, Kenya, Rwanda and Ghana (thesis Table 1, section 4.2)."""
    rng = np.random.default_rng(seed)
    sector = rng.choice(["agriculture", "retail", "services", "education"], n)
    activity = [f"{s}_{rng.integers(1, 6)}" for s in sector]
    df = pd.DataFrame({
        "loan_amount": rng.normal(650, 290, n).clip(50, 3000).round(-1),
        "term_in_months": rng.choice([6, 8, 12, 14, 18, 24], n),
        "repayment_interval": rng.choice(["monthly", "irregular", "bullet"], n, p=[0.6, 0.25, 0.15]),
        "borrower_genders": rng.choice(["female", "male", "group"], n, p=[0.6, 0.25, 0.15]),
        "sector": sector,
        "activity": activity,
        "country": rng.choice(["Uganda", "Kenya", "Rwanda", "Ghana"], n),
    })
    if label == "random":
        # Labels independent of every feature: no model should beat chance (thesis 3.3).
        df["default"] = (rng.random(n) < 0.22).astype(int)
        return df
    # Noisy logistic rule built on the sector/repayment risk patterns cited in thesis 3.4
    # (CGAP 2020, FSD Uganda 2018). The noise keeps the task from being trivially learnable.
    logit = (
        pd.Series(sector).map({"agriculture": 0.9, "retail": 0.4, "services": -0.2, "education": -0.5}).to_numpy()
        + df["repayment_interval"].map({"monthly": 0.0, "irregular": 0.7, "bullet": 1.2}).to_numpy()
        + 1.2 * (df["loan_amount"].to_numpy() - 650) / 290 / 2
        + 0.03 * (df["term_in_months"].to_numpy() - 12)
        + rng.normal(0, 1.0, n)
    )
    # Shift the intercept so the default rate is ~22% (Ahlin & Jiang, 2008).
    df["default"] = (logit > np.quantile(logit, 0.78)).astype(int)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", choices=["rule", "random"], default="rule", help="how default labels are assigned")
    ap.add_argument("--n", type=int, default=1000, help="number of rows")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", help="output CSV (default: generated_loans_<label>.csv)")
    args = ap.parse_args()

    out = args.out or f"generated_loans_{args.label}.csv"
    df = make_dataset(args.n, args.label, args.seed)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out} (default rate {df['default'].mean():.1%})")


if __name__ == "__main__":
    main()
