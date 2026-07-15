# Telco Customer Churn Analysis - Statistical Analysis, Predictive ML & Power BI Dashboard

A full end-to-end analytics project that goes beyond training a model. The goal was threefold: first, understand **why** customers leave a telecom company using rigorous statistical hypothesis testing; second, build a system that can **predict who is likely to leave next** so a retention team can act before it's too late; and third, package those findings into a **self-serve Power BI dashboard** so non-technical stakeholders can monitor churn risk without opening a notebook.

The project covers the complete pipeline: data cleaning, exploratory analysis, statistical validation, preprocessing, model selection via cross-validation, threshold tuning, a reusable prediction function, and a live BI dashboard for business users.

---

## The Problem

Customer churn is expensive. Acquiring a new customer costs significantly more than retaining an existing one. Telecom companies typically have churn rates between 15–25% annually, and without a way to identify at-risk customers early, retention efforts end up as generic blanket discounts costly and ineffective.

This project builds a system that scores each customer with a churn probability *and* a dashboard that lets a retention manager see, at a glance, which customer segments need attention today.

---

## Dataset

The dataset is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

- **7,043 customers**
- **21 features** covering demographics, account information, and subscribed services
- **Target variable:** whether the customer churned (Yes / No)
- **Class imbalance:** approximately 73% retained, 27% churned

## Project Structure

```
├── Customer_Churn_Analysis_ML_Statistics.ipynb   # Main notebook (stats + ML pipeline)
├── Customer_Churn_Dashboard.pbix                 # Power BI executive dashboard
├── WA_Fn-UseC_-Telco-Customer-Churn.csv           # Dataset
├── customer_churn_model.pkl                       # Trained model + threshold
├── encoders.pkl                                   # Label encoders + scaler
└── images/                                        # Visualisations used in this README
```

---

## Part 1 — Statistical Analysis & Predictive ML

### 1. Data Cleaning
The dataset had no null values, but `TotalCharges` was stored as a string and contained blank spaces for 11 brand-new customers with zero tenure who had never been billed. These were identified, filled with 0, and the column was converted to float. `customerID` was dropped as it carries no predictive signal.

### 2. Exploratory Data Analysis
Tenure has a bimodal distribution many customers either leave very early or stay for a very long time. Monthly charges are roughly uniform, while total charges skew right, since they accumulate with tenure. Churned customers have noticeably shorter tenure and higher monthly charges on average — a strong signal before any modeling begins. Month-to-month contract customers churn at a dramatically higher rate than those on one- or two-year contracts, and customers without online security, tech support, or online backup also show elevated churn.

### 3. Statistical Hypothesis Testing
Charts can show patterns, but they can't tell us whether those patterns are statistically significant or just noise  so every feature was formally tested.

- **Categorical features:** Chi-square test of independence for significance, and Cramér's V to measure effect size (0–1 scale).
- **Numerical features:** Welch's independent t-test to compare means between churned and retained groups, and point-biserial correlation for direction and magnitude.

**Result:** Contract type has the highest effect size by a clear margin, followed by tenure, online security, and tech support. Every categorical feature tested came back statistically significant at p < 0.05. All three numerical features (tenure, monthly charges, total charges) showed highly significant differences between churned and retained groups, with p-values in the range of 10⁻⁴⁰ to 10⁻⁵⁰.

### 4. Preprocessing
Label encoding applied to all categorical columns (encoders saved to disk for reuse). `StandardScaler` fitted on training data only and applied to the three numerical columns. All preprocessing artifacts saved alongside the model so new data can be transformed identically.

### 5. Handling Class Imbalance
The target is imbalanced at roughly 73/27. Rather than oversampling with SMOTE, `class_weight="balanced"` was used in all models this penalizes misclassifying the minority churn class more heavily without generating synthetic data.

### 6. Model Training & Cross-Validation
Four models were trained and compared using 5-fold stratified cross-validation, scored on ROC-AUC (which measures how well the model separates churners from non-churners, regardless of threshold). **Gradient Boosting came out on top** with a mean ROC-AUC of ~0.84, narrowly ahead of Logistic Regression because it can capture non-linear interactions between features like contract type, tenure, and internet service type together.

### 7. Final Model & Threshold Tuning
```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=20,
    subsample=0.8,
    random_state=42
)
```
The default 0.5 threshold is rarely optimal for imbalanced problems, so the threshold that maximizes F1 score on the churn class was used instead meaningfully improving recall on churners, since missing an at-risk customer is more costly than a false alarm.

### 8. Feature Importance
Tenure, contract type, and monthly charges dominate the model's decisions consistent with what the statistical tests showed. The model learned the same relationships the hypothesis tests confirmed.

### Results Summary

| Metric | Value |
|---|---|
| ROC-AUC | ~0.84 |
| Accuracy (tuned threshold) | ~0.77 |
| Recall on churners | ~0.79 |
| Precision on churners | ~0.60 |

---

## Part 2 Power BI Executive Dashboard

The notebook above answers **who is going to churn and why**. The dashboard answers **what should the business do about it, today** built for retention managers and leadership who need a live, filterable view rather than a static report.

### Dashboard at a glance

| KPI | Value |
|---|---|
| Total Customers | 7,043 |
| Churned Customers | 1,869 |
| Retained Customers | 5,174 |
| **Overall Churn Rate** | **26.5%** |

The dashboard includes a churn-split pie chart, a customer treemap segmented by risk tier, contract- and tenure-based churn breakdowns, and monthly-charge comparisons between churned and retained customers — all filterable in real time.

### Key insights surfaced by the dashboard

- **Contract type is the biggest lever, by a wide margin.** Month-to-month customers churn at **42.7%**, one-year contract customers at **11.3%**, and two-year contract customers at just **2.8%** a **~15x gap** between the riskiest and safest contract types.
- **New customers are the most fragile.** Customers in their first year (0–12 months tenure) churn at **47.4%**, compared to just **9.5%** for customers with 48+ months tenure churn is overwhelmingly an early-lifecycle problem, not a long-term satisfaction one.
- **Churned customers pay more, not less.** Average monthly charges for churned customers are **$74.44**, versus **$61.27** for retained customers pointing to a pricing or perceived-value problem rather than customers leaving to save money elsewhere.
- **Risk compounds:** a new, month-to-month, higher-paying customer sits at the intersection of all three high-risk signals simultaneously exactly the segment the dashboard is built to surface first.

### Why this matters for the business

Rather than spreading a flat retention budget evenly across all 7,043 customers, the dashboard lets a retention team filter directly to the highest-risk segment new, month-to-month, higher-paying customers and prioritize contract-upgrade or loyalty offers there first. That's the difference between a reactive, blanket-discount strategy and a targeted, ROI-driven one.

### Built with
Power BI Desktop, DAX (custom measures for Churn Rate %, Retained/Churned Customer counts, and Average Monthly Charges by churn status), and calculated columns for Risk Type and Tenure Band segmentation.

---

## Combined Key Findings

1. **Contract type is the single strongest driver of churn** confirmed statistically (highest Cramér's V) *and* visually in the dashboard (15x gap between month-to-month and two-year contracts).
2. **Short tenure is tightly linked to churn.** Customers who leave tend to do so early — suggesting an onboarding and early-experience problem rather than a long-term satisfaction problem.
3. **Lack of value-added services** (online security, tech support, device protection) is associated with higher churn customers with more services have more reasons to stay.
4. **Higher monthly charges correlate with churn**, both in the model's feature importance and in the dashboard's direct comparison pointing to a pricing or value-perception issue worth investigating, particularly for fiber-optic customers.

---

## Technologies Used

- **Python 3** — pandas, numpy (data manipulation)
- **matplotlib, seaborn** — visualization
- **scipy** — statistical hypothesis testing (chi-square, t-test, point-biserial correlation)
- **scikit-learn** — preprocessing, model training, cross-validation, evaluation
- **pickle** — model serialization
- **Power BI, DAX, Power Query** — executive dashboard and business-facing reporting

---

## Making Predictions

Load the saved model and call `predict_churn` with any customer record:

```python
import pickle, pandas as pd

with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
with open("encoders.pkl", "rb") as f:
    preprocessing = pickle.load(f)

def predict_churn(customer_dict):
    row = pd.DataFrame([customer_dict])
    for col, encoder in preprocessing["encoders"].items():
        if col in row.columns:
            row[col] = encoder.transform(row[col])
    row[preprocessing["num_cols"]] = preprocessing["scaler"].transform(
        row[preprocessing["num_cols"]]
    )
    prob = model_data["model"].predict_proba(
        row[model_data["feature_names"]]
    )[0][1]
    label = "Churn" if prob >= model_data["threshold"] else "No Churn"
    print(f"Prediction : {label}")
    print(f"Churn prob : {prob:.2%}")
    return label, prob
```

---

## How to Run

**Clone the repository**
```bash
git clone https://github.com/Dishabhadauria30/-Telco-Customer-Churn-Statistical-Analysis-Predictive-ML.git
```

**Navigate to the project**
```bash
cd -Telco-Customer-Churn-Statistical-Analysis-Predictive-ML
```

**Install dependencies**
```bash
pip install -r requirements.txt
```

**Run the notebook**
```bash
jupyter notebook
```

**View the dashboard**
Open `Customer_Churn_Dashboard.pbix` in Power BI Desktop (free download from Microsoft) to explore the interactive version.

---

## What I Would Add Next

- **SHAP values** for per-customer explainability — not just which features matter globally, but why the model made a specific prediction for a specific person.
- **A Streamlit app** to make the prediction function directly accessible to a non-technical retention team, beyond the static dashboard.
- **Experiment tracking** with MLflow or Weights & Biases to make hyperparameter comparisons more systematic.
- **A direct SMOTE vs. class-weight ablation study** to quantify the tradeoff between the two imbalance-handling approaches.
- **Row-level drill-through in Power BI**, so a retention manager could click a risk segment and see the actual list of at-risk customers, not just the aggregate numbers.

