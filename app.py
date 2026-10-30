"""Web front end for the loan default models.

Run:  ./.env/bin/streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from generate_data import make_dataset
from loan_default import evaluate, prepare

THESIS = ("Enhancing Financial Inclusion in Uganda through Data Science: A Study on Predictive Analytics "
          "for Loan Default Prediction for Microfinance Institutions")
# Categorical slots 1-3 of the reference palette (validated all-pairs); the baseline is a neutral reference.
MODELS = ["Majority baseline", "Logistic Regression", "Random Forest", "XGBoost"]
COLORS = ["#9a9893", "#2a78d6", "#eb6834", "#1baf7a"]
MODEL_COLOR = alt.Color("Model:N", scale=alt.Scale(domain=MODELS, range=COLORS), legend=alt.Legend(orient="bottom"))

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("Loan Default Prediction for Microfinance Institutions")
st.markdown(
    f"**MSc Data Science dissertation · University of East London**  \n"
    f"*{THESIS}*  \n"
    f"Asiimwe Brian Gita · Module UEL-DS-7010 · Dissertation submitted 20 April 2025"
)
st.caption("Compares Logistic Regression, Random Forest and XGBoost against a majority-class baseline, "
           "following the methodology in Chapters 3 and 4 of the dissertation.")


@st.cache_data
def run(df, target, positive):
    X, y, _ = prepare(df, target, positive)
    results, cms, details = evaluate(X, y)
    importances = {}
    for name in ("Random Forest", "XGBoost"):
        pre, clf = details["models"][name].steps[0][1], details["models"][name].steps[-1][1]
        names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
        importances[name] = pd.Series(clf.feature_importances_, index=names)
    return list(X.columns), results, cms, details["y_test"].to_numpy(), details["probs"], importances


# --- 1. Choose data -----------------------------------------------------------
with st.sidebar:
    st.header("1. Data")
    source = st.radio("Source", ["Generate synthetic loans", "Upload a CSV", "Sample: German Credit (real outcomes)"])

    if source == "Generate synthetic loans":
        n = st.slider("Number of loans", 200, 10000, 1000, step=100)
        label = st.radio("How defaults are assigned", ["rule", "random"],
                         format_func={"rule": "Risk rule + noise (learnable)",
                                      "random": "Random (no pattern, sanity check)"}.get)
        seed = st.number_input("Random seed", value=42, step=1)
        df, target = make_dataset(n, label, int(seed)), "default"
    elif source == "Upload a CSV":
        file = st.file_uploader("CSV with one row per loan and a binary default column", type="csv")
        if file is None:
            st.info("Upload a CSV to continue.")
            st.stop()
        df = pd.read_csv(file)
        guess = next((c for c in df.columns if c.lower() in ("default", "loan_status", "class", "target", "y")),
                     df.columns[-1])
        target = st.selectbox("Target column (did the loan default?)", df.columns, index=list(df.columns).index(guess))
    else:
        df, target = pd.read_csv("german_credit.csv"), "class"

    values = df[target].dropna().unique()
    if len(values) != 2:
        st.error(f"'{target}' needs exactly 2 distinct values, found {len(values)}. Pick another column.")
        st.stop()
    counts = df[target].value_counts()
    default_value = 1 if set(values) == {0, 1} else ("bad" if "bad" in values else counts.idxmin())
    positive = st.selectbox("Value meaning 'defaulted'", values, index=list(values).index(default_value))

    st.divider()
    st.caption(f"Developed for the MSc Data Science dissertation at the University of East London, "
               f"Asiimwe Brian Gita, 2025.")

# --- 2. Preview ---------------------------------------------------------------
st.subheader("Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Loans", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Default rate", f"{(df[target] == positive).mean():.1%}")
st.dataframe(df.head(50), width="stretch", height=250)
st.download_button("Download this dataset (CSV)", df.to_csv(index=False), "loans.csv", "text/csv")

# --- 3. Train and evaluate ----------------------------------------------------
st.subheader("Model results")
if not st.button("Train and evaluate models", type="primary"):
    st.caption("Uses an 80/20 stratified train/test split and 5-fold cross-validation on the training set.")
    st.stop()

try:
    with st.spinner("Training models..."):
        features, results, cms, y_test, probs, importances = run(df, target, positive)
except ValueError as e:  # e.g. too few defaults for stratified 5-fold CV
    st.error(f"Could not train on this dataset: {e}")
    st.stop()

st.caption(f"Features used: {', '.join(features)} · Test set: {len(y_test)} loans, {int(y_test.sum())} defaults")
st.dataframe(results, width="stretch")

# Metric comparison
metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
long = results[metrics].reset_index().melt("Model", var_name="Metric", value_name="Score")
st.markdown("**Test-set metrics by model**")
st.altair_chart(
    alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Metric:N", sort=metrics, title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("Model:N", sort=MODELS),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
        color=MODEL_COLOR,
        tooltip=["Model", "Metric", alt.Tooltip("Score:Q", format=".3f")],
    ).properties(height=320),
    width="stretch",
)

# ROC and precision-recall curves
roc_rows, pr_rows = [], []
for name, p in probs.items():
    if name == "Majority baseline":
        continue
    auc = results.loc[name, "ROC-AUC"]
    fpr, tpr, _ = roc_curve(y_test, p)
    roc_rows += [{"Model": name, "False positive rate": a, "True positive rate": b, "AUC": auc} for a, b in zip(fpr, tpr)]
    prec, rec, _ = precision_recall_curve(y_test, p)
    pr_rows += [{"Model": name, "Recall": a, "Precision": b} for a, b in zip(rec, prec)]

left, right = st.columns(2)
with left:
    st.markdown("**ROC curves** (dashed line = random guessing)")
    chance = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(
        strokeDash=[4, 4], color="#9a9893").encode(x="x:Q", y="y:Q")
    roc = alt.Chart(pd.DataFrame(roc_rows)).mark_line(strokeWidth=2, interpolate="step-after").encode(
        x=alt.X("False positive rate:Q", title="False positive rate (good loans flagged)"),
        y=alt.Y("True positive rate:Q", title="True positive rate (defaults caught)"),
        color=MODEL_COLOR,
        tooltip=["Model", alt.Tooltip("AUC:Q", format=".3f"),
                 alt.Tooltip("False positive rate:Q", format=".2f"), alt.Tooltip("True positive rate:Q", format=".2f")],
    )
    st.altair_chart((chance + roc).properties(height=340), width="stretch")
with right:
    st.markdown(f"**Precision–recall curves** (dashed line = default rate, {y_test.mean():.0%})")
    base = alt.Chart(pd.DataFrame({"y": [y_test.mean()]})).mark_rule(strokeDash=[4, 4], color="#9a9893").encode(y="y:Q")
    pr = alt.Chart(pd.DataFrame(pr_rows)).mark_line(strokeWidth=2, interpolate="step-before").encode(
        x=alt.X("Recall:Q", title="Recall (share of defaults caught)"),
        y=alt.Y("Precision:Q", title="Precision (flagged loans that defaulted)", scale=alt.Scale(domain=[0, 1])),
        color=MODEL_COLOR,
        tooltip=["Model", alt.Tooltip("Recall:Q", format=".2f"), alt.Tooltip("Precision:Q", format=".2f")],
    )
    st.altair_chart((base + pr).properties(height=340), width="stretch")

# Feature importance
st.markdown("**Which features drive the predictions** (tree-model feature importance)")
for col, (name, imp) in zip(st.columns(2), importances.items()):
    data = imp.sort_values(ascending=False).head(15).rename_axis("Feature").reset_index(name="Importance")
    col.altair_chart(
        alt.Chart(data, title=name).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4,
                                             color=COLORS[MODELS.index(name)]).encode(
            x=alt.X("Importance:Q"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
        ).properties(height=max(160, 24 * len(data))),
        width="stretch",
    )

# Predicted probability distribution
st.markdown("**XGBoost predicted default probability, by actual outcome**  \n"
            "<small>Good separation means repaid loans cluster near 0 and defaults near 1. "
            "The 0.5 line is the cut-off behind the precision and recall figures.</small>", unsafe_allow_html=True)
dist = pd.DataFrame({"Predicted probability of default": probs["XGBoost"],
                     "Actual outcome": np.where(y_test == 1, "Defaulted", "Repaid")})
hist = alt.Chart().mark_bar(color=COLORS[3], cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
    x=alt.X("Predicted probability of default:Q", bin=alt.Bin(extent=[0, 1], step=0.05)),
    y=alt.Y("count():Q", title="Loans"),
    tooltip=[alt.Tooltip("Predicted probability of default:Q", bin=alt.Bin(extent=[0, 1], step=0.05)),
             alt.Tooltip("count():Q", title="Loans")],
).properties(height=180)
cut = alt.Chart().mark_rule(strokeDash=[4, 4], color="#52514e").encode(x=alt.datum(0.5))
st.altair_chart(alt.layer(hist, cut, data=dist).facet(
    row=alt.Row("Actual outcome:N", sort=["Repaid", "Defaulted"], title=None)).resolve_scale(y="independent"),
    width="stretch")

# Confusion matrices
st.markdown("**Confusion matrices on the test set**")
for col, (name, cm) in zip(st.columns(len(cms)), cms.items()):
    (tn, fp), (fn, tp) = cm
    cells = pd.DataFrame({"Actual": ["Repaid", "Repaid", "Defaulted", "Defaulted"],
                          "Predicted": ["Repaid", "Defaulted", "Repaid", "Defaulted"],
                          "Loans": [tn, fp, fn, tp]})
    heat = alt.Chart(cells, title=name).encode(
        x=alt.X("Predicted:N", sort=["Repaid", "Defaulted"], axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Actual:N", sort=["Repaid", "Defaulted"]),
        tooltip=["Actual", "Predicted", "Loans"],
    )
    col.altair_chart(
        (heat.mark_rect(stroke="white", strokeWidth=2).encode(
            color=alt.Color("Loans:Q", scale=alt.Scale(scheme="blues"), legend=None))
         + heat.mark_text(fontSize=15).encode(
            text="Loans:Q",
            color=alt.condition(alt.datum.Loans > cm.max() / 2, alt.value("white"), alt.value("#0b0b0b"))))
        .properties(height=200),
        width="stretch",
    )
    col.caption(f"Caught {tp} of {tp + fn} defaults, wrongly flagged {fp} good loans")

with st.expander("How to read these results"):
    st.markdown(
        "- **ROC-AUC**: how well the model ranks defaulters above non-defaulters. 0.5 = chance, 1.0 = perfect.\n"
        "- **Accuracy** can look high even for the baseline that never predicts a default, so compare against it.\n"
        "- **Recall**: share of real defaults caught. **Precision**: share of flagged loans that really defaulted.\n"
        "- **CV ROC-AUC**: average ± spread over 5 folds, more stable than a single test split.\n"
        "- **Feature importance** shows what the models rely on, not what causes default.\n"
        "- With random labels, every model should score about 0.5. If not, something is leaking."
    )

st.divider()
st.caption(f"© 2025 Asiimwe Brian Gita. Developed for the MSc Data Science dissertation \"{THESIS}\", "
           f"University of East London.")
