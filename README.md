# 📡 Telco Customer Churn  Statistical Analysis & Predictive ML

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![SciPy](https://img.shields.io/badge/SciPy-Hypothesis%20Testing-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=for-the-badge)](https://seaborn.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)

> Most churn projects stop at "Random Forest gives 80% accuracy."
> This one starts a step earlier: **which features actually, statistically, matter** 
> and only then builds a model on top of that evidence.

---

##  Business Problem

A telecom company has a **26.54% customer churn rate** (95% CI: 25.51%–27.57%
across 7,043 customers)  roughly **1 in 4 customers leaves**. Acquiring a new
customer costs far more than retaining an existing one, so the business needs
to know:

1. **Which factors are *statistically proven* to drive churn** (not just visually correlated)?
2. **Which customers, right now, are at risk of leaving**  so retention offers can be targeted instead of blasted to everyone?

---

##  What Makes This Project Different
This project inserts a **statistical validation layer** in between the same
workflow an analyst would use to justify a recommendation to a business
stakeholder before it reaches a data scientist's model.

```
Data Cleaning → EDA → Statistical Hypothesis Testing → Preprocessing
              → Class-Imbalance Handling → Model Comparison (4 algorithms)
              → Final Model Evaluation → Predictive System
```

---

##  Machine Learning Results

Four models were compared using **5-fold Stratified Cross-Validation** (ROC-AUC):

| Model | Mean ROC-AUC |
|---|---|
| **Gradient Boosting** ✅ | **0.847** |
| Logistic Regression | 0.844 |
| Random Forest | 0.824 |
| Decision Tree | 0.648 |

**Final model: Gradient Boosting**  on the held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 0.80 |
| ROC-AUC | 0.839 |
| Precision (Churn class) | 0.65 |
| Recall (Churn class) | 0.52 |
| F1-score (Churn class) | 0.58 |

**Top 5 features driving the model's predictions:**
1. Contract (36.4%)
2. MonthlyCharges (15.1%)
3. Tenure (13.7%)
4. TotalCharges (11.9%)
5. OnlineSecurity (7.3%)

Contract type alone accounts for over a third of the model's predictive
power — and it's also the strongest statistical driver found in the
hypothesis-testing section. The statistics and the model agree, which is
exactly the kind of cross-validation a stakeholder wants to see.

---

##  Business Recommendations

1. **Target month-to-month contract customers first**  Contract is both the
   strongest statistical driver (Cramér's V = 0.41) and the top ML feature
   (36.4% importance). Offering discounted 1-year contract upgrades to
   month-to-month customers is the single highest-leverage retention action.

2. **Build a "new customer" retention program for the first 18 months** 
   churned customers average just 18 months of tenure vs. 38 for retained
   customers. Early-tenure customers are the highest-risk group.

3. **Bundle Online Security & Tech Support into base packages** both show
   strong, significant associations with churn (Cramér's V > 0.34). Customers
   without these add-ons churn at noticeably higher rates.

4. **Review pricing for high-Monthly-Charge customers** churned customers
   pay roughly ₹13 more per month on average (₹74.44 vs ₹61.27), a
   statistically significant difference (p < 0.001).

5. **Deprioritize gender-based segmentation**  statistically, gender has
   zero relationship with churn (p = 0.487). Any retention campaign
   segmented by gender is very likely wasted effort.

6. **Use the model's probability score, not just the label** the model
   outputs `predict_proba()`, so the business can rank customers by churn
   risk score and focus retention budget on the top percentile, rather than
   treating all "predicted churn" customers identically.

---

##  Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **Pandas / NumPy** | Data wrangling, feature engineering |
| **SciPy** | Chi-Square test, t-tests, Cramér's V, point-biserial correlation |
| **Scikit-learn** | Label encoding, train/test split, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, model evaluation |
| **Matplotlib / Seaborn** | Distribution plots, box plots, count plots, heatmaps, ROC curve |
| **Pickle** | Model & encoder persistence for the predictive system |
| **Jupyter Notebook** | Development & analysis narrative |

---

##  Handling Class Imbalance , A Design Choice Worth Explaining

The target is imbalanced (73.5% No-Churn vs 26.5% Churn). This project uses
`class_weight="balanced"` rather than SMOTE:

- No extra dependencies (`imbalanced-learn` not required)
- No synthetic data  avoids any risk of synthetic samples leaking
  unrealistic patterns into the model
- Achieves a comparable recall lift on the minority class

SMOTE is documented as a commented-out alternative in the notebook for
anyone who wants to compare both approaches a good discussion point in
interviews about why one imbalance-handling technique was chosen over
another.

---

##  Dataset

**Source:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
**Rows:** 7,043 customers | **Columns:** 21 (demographics, account info, services subscribed, churn label)

---

##  How to Run

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction_using_ML.ipynb
```

**requirements.txt**
```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
jupyter
```

---

##  Future Work

1. Hyperparameter tuning via `GridSearchCV` / `RandomizedSearchCV`
2. Compare SMOTE vs. `class_weight="balanced"` head-to-head
3. Address overfitting risk via tree-depth / learning-rate tuning + learning curves
4. Add SHAP values for individual-customer-level explainability
5. Deploy via a simple Streamlit app for interactive churn-risk scoring

---

## 👩‍💻 About the Author

**Disha Bhadauria** | Aspiring Data Analyst

**Skills demonstrated in this project:**
- Data cleaning & feature engineering (handling disguised missing values)
- Statistical hypothesis testing (Chi-Square, Cramér's V, t-tests, point-biserial correlation, confidence intervals)
- Supervised ML model comparison (4 algorithms, stratified CV)
- Class-imbalance handling with a justified, documented design choice
- Model persistence & a working predictive system
- Translating statistical and ML findings into business recommendations

[LinkedIn](#) • [Portfolio](#) • [Email](#)
