# Telco Customer Churn Prediction

A full end-to-end machine learning project that goes beyond simply training a model. The goal was to first understand *why* customers leave a telecom company using rigorous statistical hypothesis testing, and then build a system that can *predict who* is likely to leave next  so a retention team can act before it is too late.

The project covers the complete pipeline: data cleaning, exploratory analysis, statistical validation, preprocessing, model selection via cross-validation, threshold tuning, and a reusable prediction function.

---

## The Problem

Customer churn is expensive. Acquiring a new customer costs significantly more than retaining an existing one. Telecom companies typically have churn rates between 15 and 25 percent annually, and without a way to identify at risk customers early, retention efforts end up as generic blanket discounts that are costly and ineffective.

This project builds a system that scores each customer with a churn probability so the business can focus its retention budget on the customers most likely to leave.

---

## Dataset

The dataset is the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.

- 7,043 customers
- 21 features covering demographics, account information, and subscribed services
- Target variable: whether the customer churned (Yes / No)
- Class imbalance: approximately 73% retained, 27% churned

---

## Project Structure

```
├── Customer_Churn_Analysis_ML_Statistics.ipynb   # Main notebook
├── WA_Fn-UseC_-Telco-Customer-Churn.csv          # Dataset
├── customer_churn_model.pkl                       # Trained model + threshold
├── encoders.pkl                                   # Label encoders + scaler
└── images/                                        # Visualisations used in this README
```

---

## Step-by-Step Pipeline

### 1. Data Cleaning

The dataset had no null values but `TotalCharges` was stored as a string and contained blank spaces for 11 brand-new customers with zero tenure who had never been billed. These were identified, filled with 0, and the column was converted to float. The `customerID` column was dropped as it carries no predictive signal.

### 2. Exploratory Data Analysis

We start with distributions of the three numerical features  tenure, monthly charges, and total charges  and then look at how each categorical feature breaks down by churn status.

**Numerical feature distributions**

![Numerical Distributions](images/numerical_distributions.png)

Tenure has a bimodal distribution: many customers either leave very early or stay for a very long time. Monthly charges are roughly uniform, while total charges skew right because they accumulate with tenure.

**Numerical features split by churn**

![Violin Plot](images/violin_churn_split.png)

Churned customers have noticeably shorter tenure and higher monthly charges on average. This is already a strong signal before we even run any models.

**Correlation heatmap**

![Correlation Heatmap](images/correlation_heatmap.png)

Total charges is highly correlated with tenure, which makes sense as it is simply the running total of monthly bills. We keep both features but this is worth noting.

**Categorical features by churn status**

![Categorical Count Plots](images/categorical_countplots.png)

Month-to-month contract customers churn at a dramatically higher rate than those on one or two year contracts. Customers without online security, tech support, or online backup also show elevated churn. These patterns are consistent and visible across the dataset.

---

### 3. Statistical Hypothesis Testing

Charts can show patterns but they cannot tell us whether those patterns are statistically significant or just noise. We run formal hypothesis tests on every feature.

**For categorical features:** Chi-square test of independence to check significance, and Cramér's V to measure effect size on a 0 to 1 scale.

**For numerical features:** Welch's independent t-test to compare means between churned and retained groups, and point-biserial correlation to get direction and magnitude.

**Cramér's V effect sizes — categorical features**

![Cramers V](images/cramers_v_barchart.png)

Contract type has the highest effect size by a clear margin. Tenure, online security, and tech support follow. Every single categorical feature tested came back statistically significant at p < 0.05, meaning none of the patterns visible in the charts are noise.

For numerical features, all three (tenure, monthly charges, total charges) showed highly significant differences between churned and retained groups with p-values in the range of 10⁻⁴⁰ to 10⁻⁵⁰.

---

### 4. Preprocessing

- Label encoding applied to all categorical columns, with encoders saved to disk for reuse
- `StandardScaler` fitted on training data only and applied to the three numerical columns  essential for Logistic Regression and good practice generally
- All preprocessing artifacts saved alongside the model so new data can be transformed identically

---

### 5. Handling Class Imbalance

The target is imbalanced at roughly 73/27. Rather than oversampling (SMOTE), we use `class_weight="balanced"` in all models, which tells each model to penalise misclassifying the minority churn class more heavily. This avoids generating synthetic data while still correcting for the imbalance.

---

### 6. Model Training and Cross-Validation

Four models were trained and compared using 5-fold stratified cross-validation. The metric used is ROC-AUC, which measures how well the model separates churners from non-churners regardless of the decision threshold.

**Cross-validation comparison**

![Model Comparison](images/cv_model_comparison.png)

Gradient Boosting came out on top with a mean ROC-AUC of approximately 0.84, narrowly ahead of Logistic Regression. Unlike Logistic Regression, Gradient Boosting can capture non-linear interactions between features such as the combination of contract type, tenure, and internet service type.

---

### 7. Final Model and Threshold Tuning

The final Gradient Boosting model was trained with light hyperparameter tuning to reduce overfitting:

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

The default classification threshold of 0.5 is rarely optimal for imbalanced problems. We searched for the threshold that maximises the F1 score on the churn class. This meaningfully improved recall on churners which is the right trade-off here because missing an at-risk customer is more costly than a false alarm.

**Confusion matrix**

![Confusion Matrix](images/confusion_matrix.png)

**ROC curve**

![ROC Curve](images/roc_curve.png)

The model achieves an AUC of approximately 0.84, meaning it correctly ranks a churner above a non-churner 84% of the time.

---

### 8. Feature Importance

![Feature Importance](images/feature_importance.png)

Tenure, contract type, and monthly charges dominate the model's decisions consistent with what the statistical tests showed. The model has learnt the same relationships that the hypothesis tests confirmed.

---

## Results Summary

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.84 |
| Accuracy (tuned threshold) | ~0.77 |
| Recall on churners | ~0.79 |
| Precision on churners | ~0.60 |

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

## Key Findings

Contract type is the single strongest driver of churn. Customers on month-to-month contracts churn at rates several times higher than those on annual or two-year contracts, and this difference is statistically confirmed with the highest Cramér's V effect size of all features tested.

Short tenure is tightly linked to churn. Customers who leave tend to do so early. Those who make it past the first year or two show dramatically lower churn rates, suggesting an onboarding and early experience problem rather than a long-term satisfaction problem.

Lack of value-added services  online security, tech support, device protection — is associated with higher churn. Customers who subscribe to more services have more reasons to stay and a higher switching cost.

Higher monthly charges correlate with churn. Fiber optic customers pay more and churn more, which points to a pricing or value perception issue in that segment specifically.

---

## Technologies Used

- Python 3
- pandas, numpy — data manipulation
- matplotlib, seaborn — visualisation
- scipy — statistical hypothesis testing (chi-square, t-test, point-biserial correlation)
- scikit-learn — preprocessing, model training, cross-validation, evaluation
- pickle — model serialisation

---

## How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction

# Install dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn

# Download the dataset from Kaggle and place it in the project root
# https://www.kaggle.com/datasets/blastchar/telco-customer-churn

# Open the notebook
jupyter notebook Customer_Churn_Analysis_ML_Statistics.ipynb
```

---

## What I Would Add Next

There are a few natural extensions to this project. SHAP values would allow per-customer explainability  not just which features matter globally, but why the model made a specific prediction for a specific person. A Streamlit app would make the prediction function accessible to a non-technical retention team. Experiment tracking with MLflow or Weights and Biases would make hyperparameter comparisons more systematic. And comparing SMOTE oversampling against the class-weight approach directly would be a useful ablation study.


