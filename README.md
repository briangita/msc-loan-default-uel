# Loan Default Prediction for Microfinance Institutions in Uganda

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://msc-loan-default-uel-liabsckn4jrdhxzygsvgz5.streamlit.app/)

**Live app:** https://msc-loan-default-uel-liabsckn4jrdhxzygsvgz5.streamlit.app/

Code that goes with the MSc Data Science dissertation:

> **Enhancing Financial Inclusion in Uganda through Data Science: A Study on Predictive Analytics for Loan Default Prediction for Microfinance Institutions.**
> Asiimwe Brian Gita, University of East London (UEL-DS-7010), 20 April 2025.

This repository rebuilds the modelling pipeline described in Chapters 3 and 4 of the dissertation. It also adds the validation checks that the dissertation itself recommends (§3.3, §3.5, §4.5.2, §5.2). You can run it on the dissertation's simulated data or on any CSV that has a binary default column.

---

## 1. The thesis in brief

### Problem
Microfinance institutions (MFIs) in Uganda lend to people the formal banking system leaves out. FinScope (2018) found that 42% of Ugandan adults have no formal financial services. Default rates that stay above about 10% (Ahlin & Jiang, 2008) push MFIs towards conservative lending, which excludes creditworthy borrowers. Traditional credit scoring depends on formal income and credit history, and those records rarely exist in Uganda's informal economy.

### Research question
Can supervised machine learning models, trained on basic loan and borrower attributes, tell defaulters apart from non-defaulters well enough to support fairer and more consistent lending decisions by Tier III MFIs in Uganda?

### Theoretical framing
| Theory | Role in the study |
|---|---|
| Financial Inclusion Theory (Beck et al., 2007, 2009) | Explains why wider access to credit matters for poverty reduction (SDG 1, SDG 8) |
| Credit Scoring Theory (Thomas, 2000) | Gives the framework for quantifying borrower risk |
| Data-Driven Decision-Making (Provost & Fawcett, 2013) | Argues for replacing loan officers' intuition with empirical models |

### Data
The study works from the structure of the public **Kiva Microfunds** loan dataset, filtered to Uganda, Kenya, Rwanda and Ghana. The dataset actually modelled has 1,000 synthetic loan records generated to mirror Kiva's fields and Sub-Saharan microfinance patterns (§4.2). Kiva records do **not** say whether a loan was repaid, so the dissertation **synthesised** a binary `default` label at a rate of about 22%, taken from East African MFI literature. The dissertation stresses this point: every metric measures how well a model recovers the *simulated* label, not real repayment behaviour (§3.2, §4.1, §5.2).

### Models
1. **Logistic Regression**: the interpretable baseline traditionally used in credit scoring.
2. **Random Forest**: a bagged tree ensemble that captures non-linear interactions.
3. **XGBoost**: a regularised gradient-boosted tree ensemble.

### Results reported in the dissertation (Table 2)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.855 | 0.765 | 0.696 | 0.729 | 0.913 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | 0.890 | 0.805 | 0.740 | 0.770 | 0.940 |

The dissertation itself warns that the perfect Random Forest score is a *diagnostic signal*, not proof of generalisation (§4.5.2). It also describes the XGBoost figures as "estimated" with "incomplete training" (§4.1, §4.5). See [§6](#6-how-these-results-compare-with-the-dissertation) for what this code finds.

### Conclusion of the dissertation
The study demonstrates a working *method* for building and comparing credit-risk models on microfinance-style data. Because the outcome was simulated, it does **not** show validated real-world default prediction. Future work needs loan-level repayment records, validation on a later time period, calibration and cost-sensitive threshold choice, fairness audits, and a pilot with human review before any deployment (§5.3).

---

## 2. Repository contents

| File | Purpose |
|---|---|
| [generate_data.py](generate_data.py) | Builds a Kiva-style synthetic loan dataset and writes it to CSV |
| [loan_default.py](loan_default.py) | Trains and evaluates the models on any CSV |
| [app.py](app.py) | Streamlit web app: generate or upload a dataset and view model results |
| [test_loan_default.py](test_loan_default.py) | Sanity check: models must learn rule-based labels and must *fail* on random labels |
| `generated_loans_rule.csv` | Synthetic data with labels from a noisy risk rule |
| `generated_loans_random.csv` | Synthetic data with labels unrelated to any feature |
| `german_credit.csv` | UCI Statlog German Credit data: 1,000 real loans with observed good/bad outcomes |
| `.env/` | Local conda environment (Python 3.11, pandas, scikit-learn, XGBoost) |

---

## 3. Setup

The environment is already in `.env/`. To recreate it:

```bash
conda create -p ./.env -c conda-forge python=3.11 pandas scikit-learn xgboost streamlit
```

Use conda-forge rather than pip on macOS. XGBoost needs the OpenMP runtime (`libomp`), which pip does not install, and conda-forge includes it.

In VS Code, choose `./.env/bin/python` as the interpreter (Cmd+Shift+P → *Python: Select Interpreter*). `.vscode/settings.json` already points there.

---

### Deploying online
The app is deployed on [Streamlit Community Cloud](https://share.streamlit.io) at https://msc-loan-default-uel-liabsckn4jrdhxzygsvgz5.streamlit.app/. To redeploy elsewhere: connect the GitHub repository, choose `app.py` as the main file and **Python 3.12** under *Advanced settings*. Libraries install from [requirements.txt](requirements.txt). The pinned XGBoost 3.4.1 on PyPI needs Python 3.12 or newer.

## 4. Usage

### Web app
```bash
./.env/bin/streamlit run app.py        # opens http://localhost:8501
```
In the sidebar, choose to generate synthetic loans (size, label mode, seed), upload your own CSV and pick its target column and "defaulted" value, or load the German Credit sample. Then press **Train and evaluate models**. The results page shows:
- a decision-threshold slider that updates precision, recall, F1, accuracy and the confusion matrices;
- the metrics table and a grouped bar chart of the test-set metrics;
- ROC curves and precision–recall curves for the three models;
- feature importance for Random Forest and XGBoost;
- the distribution of XGBoost's predicted default probabilities for repaid and defaulted loans, with the chosen cut-off;
- a confusion-matrix heatmap for each model.

### Generate data
```bash
./.env/bin/python generate_data.py                       # -> generated_loans_rule.csv
./.env/bin/python generate_data.py --label random        # -> generated_loans_random.csv
./.env/bin/python generate_data.py --n 5000 --seed 7 --out big.csv
```

### Train and evaluate
```bash
./.env/bin/python loan_default.py generated_loans_rule.csv
./.env/bin/python loan_default.py german_credit.csv --target class --positive bad
./.env/bin/python loan_default.py your_data.csv --target loan_status --positive 1
```

| Option | Meaning |
|---|---|
| `csv` | Input file (required) |
| `--target` | Name of the binary outcome column (default: `default`) |
| `--positive` | Target value meaning "defaulted". Defaults to `1` for 0/1 targets, otherwise the minority class |

### Run the check
```bash
./.env/bin/python test_loan_default.py     # prints "ok"
```

---

## 5. How the model works

### 5.1 Synthetic data (`generate_data.py`)
Follows the dissertation's feature set (Table 1, §4.2):

| Feature | Type | Values |
|---|---|---|
| `loan_amount` | numeric | Normal(mean 650, SD 290) USD, clipped to 50–3000 |
| `term_in_months` | numeric | 6, 8, 12, 14, 18, 24 |
| `repayment_interval` | ordinal | monthly (60%), irregular (25%), bullet (15%) |
| `borrower_genders` | nominal | female (60%), male (25%), group (15%) |
| `sector` | nominal | agriculture, retail, services, education |
| `activity` | nominal | 5 sub-activities per sector |
| `country` | nominal | Uganda, Kenya, Rwanda, Ghana |

**Label modes**
- `rule` (default): a noisy logistic risk score built from the sector and repayment patterns cited in §3.4 (CGAP 2020; FSD Uganda 2018):
  - agriculture is riskiest and education safest;
  - bullet repayment is riskier than irregular, which is riskier than monthly;
  - larger and longer loans add a little risk;
  - Gaussian noise is added so the label cannot be recovered exactly.

  The top 22% of scores are labelled `default = 1`.
- `random`: `default` is drawn independently of every feature at a rate of 22%. §3.3 of the dissertation predicts that no model should beat chance in this case. The mode exists to check that the pipeline isn't leaking the label.

### 5.2 Preprocessing (`loan_default.py → prepare`, `build_models`)
- **Target**: converted to 0/1 using `--positive`.
- **`repayment_interval`**: mapped to monthly=1, irregular=2, bullet=3, the ordinal risk scale from §4.2.2.
- **ID-like columns** (every value unique, excluding decimal columns such as income) are dropped. Free text such as Kiva's `loan_use` is not modelled.
- **Numeric columns**: median imputation, then min-max scaling (§3.3).
- **Categorical columns**: most-frequent imputation, then integer codes. This is the equivalent of the dissertation's `cat.codes` (§4.2.3), but categories not seen in training are handled safely.
- All preprocessing sits **inside a scikit-learn `Pipeline`**. It is fitted on the training fold only and then applied to held-out data, which prevents the leakage that §3.3 warns about.

### 5.3 Models (§4.4)
| Model | Configuration |
|---|---|
| Majority baseline | `DummyClassifier(strategy="most_frequent")`: always predicts "no default". Added as recommended in §4.5.2 |
| Logistic Regression | `max_iter=1000, random_state=42` |
| Random Forest | `n_estimators=100, random_state=42` |
| XGBoost | `learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42` |

The dissertation's `use_label_encoder=False` is left out because current XGBoost versions no longer accept it.

### 5.4 Evaluation (§3.5, §4.5)
- **80/20 stratified train/test split** with `random_state=42` (§4.4.1).
- **Test-set metrics**:
  - Accuracy: share of all predictions that are correct.
  - Precision: of the loans flagged as defaults, the share that really defaulted. Low precision means creditworthy borrowers get turned away.
  - Recall: of the real defaults, the share the model caught. Low recall means risky loans get approved.
  - F1: the harmonic mean of precision and recall.
  - ROC-AUC: how well the model ranks defaulters above non-defaulters across all thresholds. 0.5 is chance and 1.0 is perfect.
- **5-fold stratified cross-validated ROC-AUC** on the training set, reported as mean ± SD (§3.1, §3.5).
- **Confusion matrix** for each model (TN, FP, FN, TP), as recommended in §4.5.2.

Precision, recall and F1 use the default 0.5 probability threshold. A lender would choose a threshold based on what false approvals and false rejections cost (§2.3, §3.5). The web app has a threshold slider to explore this trade-off, but no cost-based threshold is chosen automatically.

---

## 6. How these results compare with the dissertation

### 6.1 Synthetic data with rule-based labels (`generated_loans_rule.csv`, 22% default)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Majority baseline | 0.780 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 ± 0.000 |
| Logistic Regression | 0.785 | 0.526 | 0.227 | 0.317 | 0.772 | 0.812 ± 0.051 |
| Random Forest | 0.790 | 0.531 | 0.386 | 0.447 | 0.780 | 0.812 ± 0.058 |
| XGBoost | 0.780 | 0.500 | 0.318 | 0.389 | 0.788 | 0.815 ± 0.052 |

### 6.2 Synthetic data with random labels (`generated_loans_random.csv`, 23.3% default realised from a 22% draw)
| Model | ROC-AUC (test) | CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.456 | 0.506 ± 0.043 |
| Random Forest | 0.605 | 0.476 ± 0.059 |
| XGBoost | 0.539 | 0.502 ± 0.059 |

Every model is at chance under cross-validation. The single test-split AUC of 0.605 for Random Forest is sampling noise on 200 rows, which is exactly why §5.2 warns against relying on one split.

### 6.3 Real outcomes: UCI German Credit (1,000 loans, 30% bad)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC |
|---|---|---|---|---|---|---|
| Majority baseline | 0.700 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 ± 0.000 |
| Logistic Regression | 0.760 | 0.620 | 0.517 | 0.564 | 0.786 | 0.781 ± 0.058 |
| Random Forest | 0.760 | 0.658 | 0.417 | 0.510 | 0.798 | 0.794 ± 0.044 |
| XGBoost | 0.790 | 0.688 | 0.550 | 0.611 | 0.813 | 0.798 ± 0.037 |

### 6.4 Interpretation
- **The perfect Random Forest score in Table 2 is not reproduced.** When the label contains realistic noise, Random Forest reaches an AUC of about 0.78, not 1.000. A perfect held-out score only happens if the label is an exact function of the features, or if information about the label leaks into the inputs. Both are possibilities the dissertation names in §4.5.2.
- **The ranking of the models only partly matches the dissertation.** The dissertation ranks Random Forest first, then XGBoost, then Logistic Regression. Here, on both datasets with learnable labels (6.1 and 6.3), XGBoost ≥ Random Forest ≥ Logistic Regression on test AUC. The two agree only that the tree ensembles edge out Logistic Regression, and those gaps are small compared with the cross-validation spread.
- **Accuracy is a misleading measure here.** On the synthetic data, accuracy is about 0.78 for every model, *including* the baseline that never predicts a default. Recall and AUC are the measures that show whether a model is useful.
- **The German Credit run is the strongest evidence in this repository.** It shows that the same pipeline works on loans with *observed* repayment outcomes and beats the baseline there. It is still not Ugandan MFI data.

---

## 7. Limitations

- **No Ugandan repayment outcomes.** The synthetic labels show only that the pipeline can learn a rule we wrote ourselves. Validation needs loan-level MFI records with origination dates, due dates, repayment events and a written definition of default (§5.3).
- **Small samples and one split.** With 1,000 rows, test metrics vary by several points from one split to the next. Cross-validation reduces the problem but does not remove it.
- **No calibration or cost-based threshold.** Predicted probabilities are not calibrated, and the 0.5 threshold ignores what each type of error costs a lender. The app's slider lets you move it by hand.
- **Not yet implemented** from the dissertation's methodology: hyperparameter grid search, SHAP explanations, fairness analysis by gender, sector or country, and NLP on `loan_use`.
- **Fairness.** `borrower_genders` is used as a feature. Any real deployment needs a proxy-discrimination audit, human oversight and a way for applicants to appeal (§3.5, §5.2).

---

## 8. Key references

As listed in the dissertation's reference list:

- Armendáriz, B., & Morduch, J. (2010). *The Economics of Microfinance* (2nd ed.). MIT Press.
- Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).
- Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring. *Journal of the Operational Research Society, 66*(4), 740-755.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems* (pp. 4765-4774).
- Thomas, L. C. (2000). A survey of credit and behavioural scoring: Forecasting financial risk of lending to consumers. *International Journal of Forecasting, 16*(2), 149-172.

- Ahlin, C., & Jiang, N. (2008). Can micro-credit bring development? *Journal of Development Economics, 86*(1), 1-21.
- Beck, T., Demirgüç-Kunt, A., & Levine, R. (2007). Finance, inequality and the poor. *Journal of Economic Growth, 12*(1), 27-49.
- Beck, T., Demirgüç-Kunt, A., & Honohan, P. (2009). Access to financial services: Measurement, impact, and policies. *The World Bank Research Observer, 24*(1), 119-145.
- Financial Sector Deepening Uganda. (2018). *FinScope Uganda 2018 survey*. https://fsduganda.or.ug/finscope-2018-case-for-deeper-inclusive-financial-sector-uganda/
- Provost, F., & Fawcett, T. (2013). *Data Science for Business: What You Need to Know About Data Mining and Data-Analytic Thinking*. O'Reilly Media.

Data source for the German Credit run: Statlog (German Credit Data), UCI Machine Learning Repository, `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data`
