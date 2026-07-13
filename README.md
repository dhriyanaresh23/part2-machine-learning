# Part 2 – Supervised Machine Learning Model – Build, Train, and Evaluate

## Project Overview

This project focuses on building, training, and evaluating supervised machine learning models using the cleaned House Prices dataset generated in Part 1. Two predictive models were developed: a Linear Regression model for predicting continuous house prices and a Logistic Regression model for binary classification. The project includes preprocessing, feature encoding, train-test splitting, feature scaling, regression analysis, classification, ROC analysis, threshold sensitivity analysis, regularization experiments, and bootstrap confidence interval estimation.

---

## Objectives

The objectives of this project are:

- Load the cleaned dataset generated in Part 1.
- Define regression and classification target variables.
- Encode categorical variables appropriately.
- Perform a leak-free train-test split.
- Apply feature scaling using StandardScaler.
- Train and evaluate a Linear Regression model.
- Compare Linear Regression with Ridge Regression.
- Train and evaluate a Logistic Regression model.
- Handle class imbalance where applicable.
- Evaluate classification performance using multiple evaluation metrics.
- Plot and interpret the ROC Curve and AUC score.
- Perform decision-threshold sensitivity analysis.
- Compare different Logistic Regression regularization strengths.
- Estimate the confidence interval for AUC using bootstrap sampling.

---

## Dataset Information

**Dataset Name:** cleaned_data.csv

The dataset used in this project is the cleaned House Prices dataset created in Part 1 after performing data cleaning and exploratory data analysis.

### Regression Target

The regression target variable is:

- **SalePrice**

The objective of the regression model is to predict the continuous house sale price.

### Classification Target

A binary classification label was created by converting the SalePrice column into two classes using its median value.

```python
y_clf = (y_reg > y_reg.median()).astype(int)
```

Class Labels:

- **1** → House price greater than the median SalePrice.
- **0** → House price less than or equal to the median SalePrice.

This conversion creates a balanced binary classification problem suitable for Logistic Regression.

---

## Repository Structure

```text
part2-machine-learning/
│
├── part2_machine_learning.ipynb
├── cleaned_data.csv
├── README.md
│
└── outputs/
    ├── roc_curve.png
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- imbalanced-learn
- Jupyter Notebook

---

## Dependencies

Install the required Python libraries before running the notebook.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

---

## How to Run

### Step 1

Clone the repository.

```bash
git clone https://github.com/dhriyanaresh23/part2-machine-learning.git
```

### Step 2

Navigate to the project directory.

```bash
cd part2-machine-learning
```

### Step 3

Install the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

### Step 4

Launch Jupyter Notebook.

```bash
jupyter notebook
```

### Step 5

Open the notebook:

```text
part2_machine_learning.ipynb
```

### Step 6

Run all notebook cells from top to bottom.

The notebook automatically performs:

- Data preprocessing
- Feature encoding
- Train-test splitting
- Feature scaling
- Linear Regression
- Ridge Regression
- Logistic Regression
- ROC Curve generation
- Threshold sensitivity analysis
- Bootstrap confidence interval estimation
- Model evaluation

---

## Environment Variables

This project does not require any API keys, database credentials, or authentication tokens.

No sensitive information is stored in this repository.

If environment variables are required in future versions, they should be stored in a `.env` file that is excluded using `.gitignore`.

---

## Outputs Generated

The notebook generates the following outputs:

- ROC Curve (`roc_curve.png`)
- Regression evaluation metrics (MSE and R²)
- Linear Regression coefficient table
- Ridge Regression comparison
- Confusion Matrix
- Classification Report
- Accuracy, Precision, Recall, F1-score, and AUC score
- Decision-threshold sensitivity table
- Bootstrap confidence interval results
  
## Task 1 – Load cleaned_data.csv

### Objective

The objective of this task is to load the cleaned dataset generated in Part 1 and prepare the feature matrix along with the regression and classification target variables for supervised machine learning.

### Implementation

The cleaned dataset (`cleaned_data.csv`) generated in Part 1 was loaded into a Pandas DataFrame.

The **feature matrix (`X`)** was created by selecting all columns except the target column (`SalePrice`).

```python
X = df.drop("SalePrice", axis=1)
```

The **regression target (`y_reg`)** was defined as the continuous numeric column **SalePrice**.

```python
y_reg = df["SalePrice"]
```

The **classification target (`y_clf`)** was created by converting the continuous SalePrice values into binary labels using the median SalePrice.

```python
y_clf = (y_reg > y_reg.median()).astype(int)
```

### Label Definitions

**Feature Matrix (X)**

Contains all predictor variables except the target column (**SalePrice**).

**Regression Label (y_reg)**

The continuous target variable **SalePrice**, which is used to train the Linear Regression model for predicting house prices.

**Classification Label (y_clf)**

A binary target variable created using the median of SalePrice.

- **1** → SalePrice greater than the median.
- **0** → SalePrice less than or equal to the median.

The median was selected as the threshold because it creates an approximately balanced binary classification problem while being less sensitive to extreme house prices than the mean.

### Result

The dataset was successfully prepared for supervised machine learning.

- **Feature Matrix (X):** All predictor variables except SalePrice.
- **Regression Label (y_reg):** SalePrice.
- **Classification Label (y_clf):** Binary labels created using the median of SalePrice.

The prepared dataset was used in all subsequent preprocessing, model training, and evaluation tasks.

## Task 2 – Encode Categorical Columns

### Objective

The objective of this task is to convert categorical features into numerical representations suitable for machine learning while preserving meaningful relationships between feature values.

### Implementation

Categorical columns in the feature matrix (**X**) were encoded using two different techniques based on whether the categories had a natural order.

#### Label Encoding for Ordinal Features

Ordinal categorical features with a natural ordering were encoded by mapping their categories to integer values while preserving their ranking.

The following quality mapping was used:

- **Po = 0**
- **Fa = 1**
- **TA = 2**
- **Gd = 3**
- **Ex = 4**

This mapping preserves the natural progression of quality levels from Poor to Excellent, allowing the machine learning models to correctly interpret higher values as representing better quality.

#### One-Hot Encoding for Nominal Features

The remaining categorical features, which do not have a natural order, were encoded using:

```python
pd.get_dummies(drop_first=True)
```

A separate binary column was created for each category, and the first dummy column was dropped (`drop_first=True`) to avoid multicollinearity (the dummy variable trap).

### Why One-Hot Encoding Was Used

Nominal categorical variables, such as neighborhoods or house styles, do not have any natural ordering. If Label Encoding were applied to these variables, categories would be assigned arbitrary integer values (for example, Neighborhood A = 0, Neighborhood B = 1, Neighborhood C = 2). This would incorrectly imply that one category is greater than or less than another, creating a **false ordinal relationship** that does not actually exist.

One-Hot Encoding avoids this problem by creating a separate binary feature for each category. Each category is represented independently using values of 0 or 1, so the machine learning model treats all categories as distinct without assuming any ordering or numerical relationship between them. This provides a more accurate representation of nominal categorical features.

### Why Label Encoding Was Used

Label Encoding was applied only to ordinal features because these variables have a meaningful ranking. Mapping categories such as **Poor**, **Fair**, **Typical**, **Good**, and **Excellent** to increasing integer values preserves their natural order and provides meaningful numerical information to the models.

### Design Decision

A combination of Label Encoding and One-Hot Encoding was selected to ensure that ordinal information was preserved while preventing artificial numerical relationships in nominal variables. Dropping the first dummy variable also reduced redundancy and avoided multicollinearity during model training.

### Result

All categorical variables were successfully converted into numerical features suitable for supervised machine learning. After encoding, no categorical (`object` or `category`) columns remained in the feature matrix, and the encoded dataset was used for train-test splitting and model training.

## Task 3 – Leak-Free Train-Test Split and Scaling

### Objective

The objective of this task is to split the dataset into training and testing sets and apply feature scaling without introducing data leakage. Proper preprocessing ensures that the machine learning models are trained fairly and evaluated on unseen data.

### Implementation

The encoded feature matrix (`X`) and both target variables (`y_reg` and `y_clf`) were divided into training and testing sets using an 80:20 split.

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

A `StandardScaler` was then applied to standardize the feature values.

The scaler was **fitted only on the training features**:

```python
scaler.fit(X_train)
```

The fitted scaler was then used to transform both the training and testing feature sets:

```python
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Why the Scaler Was Fitted Only on the Training Data

The StandardScaler was fitted **only on the training data** to prevent **data leakage**.

If the scaler were fitted on the **entire dataset before the train-test split**, it would compute the mean and standard deviation using both the training and testing data. As a result, the scaler would **encode test-set statistics into the training process**, allowing information from the unseen test data to influence model training. This constitutes **data leakage** and can lead to overly optimistic evaluation results because the model has indirectly gained information about the test set.

By fitting the scaler only on the training data and then applying the same transformation to the test data, the testing dataset remains completely unseen during training, ensuring a fair and unbiased evaluation of model performance.

### Design Decision

Performing the train-test split before feature scaling ensures that the testing dataset remains completely independent throughout the preprocessing pipeline. This approach follows standard machine learning best practices and prevents information leakage from the testing set into the training process.

### Result

The dataset was successfully split into training (80%) and testing (20%) sets. Feature scaling was applied using StandardScaler, which was fitted only on the training data and then used to transform both the training and testing feature sets. This prevented data leakage and ensured a fair evaluation of the machine learning models.

## Task 4 – Regression Model - Linear Regression

### Objective

The objective of this task is to train and evaluate a Linear Regression model for predicting house prices, interpret the model coefficients, and compare its performance with Ridge Regression.

### Implementation

A **Linear Regression** model from `sklearn.linear_model` was trained using the scaled training features and the regression target (`y_reg_train`).

The trained model was then used to predict house prices for the testing dataset.

Model performance was evaluated using the following metrics:

- Mean Squared Error (MSE)
- Coefficient of Determination (R² Score)

The coefficients learned by the Linear Regression model were printed together with their corresponding feature names, and the three features with the largest absolute coefficient values were identified.

A **Ridge Regression** model with **alpha = 1.0** was then trained using the same training and testing datasets to compare its performance with the standard Linear Regression model.

### Results

#### Linear Regression Performance

| Metric | Value |
|---------|-------:|
| Mean Squared Error (MSE) | 2,480,296,419.08 |
| R² Score | 0.6766 |

#### Top Three Features with the Largest Absolute Coefficients

| Feature | Coefficient |
|---------|------------:|
| RoofMatl_CompShg | 76,395.77 |
| GarageQual_TA | 54,662.37 |
| RoofMatl_Tar&Grv | 52,786.44 |

#### Linear Regression vs Ridge Regression

| Model | MSE | R² Score |
|--------|-------------:|------:|
| Linear Regression | 2,480,296,419.08 | 0.6766 |
| Ridge Regression | 2,107,902,000.00 | 0.7252 |

### Interpretation

A large **positive coefficient** indicates that increasing the value of the corresponding scaled feature is associated with an increase in the predicted house price while keeping all other features constant.

A large **negative coefficient** indicates that increasing the value of the corresponding scaled feature is associated with a decrease in the predicted house price while keeping all other features constant.

The features **RoofMatl_CompShg**, **GarageQual_TA**, and **RoofMatl_Tar&Grv** have the largest absolute coefficient values, indicating that they have a strong influence on the predicted house price in the Linear Regression model.

Ridge Regression applies **L2 regularization**, which penalizes large coefficient values during model training. This reduces model complexity, minimizes overfitting, and improves the model's ability to generalize to unseen data.

The **alpha** parameter controls the strength of the regularization. A larger alpha value applies a stronger penalty to large coefficients, whereas a smaller alpha value makes Ridge Regression behave more similarly to standard Linear Regression.

### Result

The Ridge Regression model achieved a **lower Mean Squared Error (2.11 × 10⁹)** and a **higher R² score (0.7252)** than the Linear Regression model (**MSE = 2.48 × 10⁹**, **R² = 0.6766**). This indicates that applying L2 regularization improved the predictive performance of the regression model by reducing prediction error and increasing the proportion of variance explained.

## Task 5 – Classification Model - Logistic Regression

### Objective

The objective of this task is to train and evaluate a Logistic Regression model for binary classification, assess its performance using multiple evaluation metrics, analyse the ROC curve and AUC score.

### Implementation

The training dataset was first checked for class imbalance using the class distribution.

The training class distribution was:

| Class | Count | Percentage |
|------:|------:|-----------:|
| 0 | 571 | 48.89% |
| 1 | 597 | 51.11% |

Since both classes contained more than **35%** of the training samples, the dataset was considered balanced. Therefore, no imbalance handling technique such as **SMOTE** or **class_weight='balanced'** was required.

A Logistic Regression model from `sklearn.linear_model` was then trained using the scaled training features.

Predictions and prediction probabilities were generated for the testing dataset.

Model performance was evaluated using:

- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-score
- ROC Curve
- Area Under the ROC Curve (AUC)

### Results

#### Class Distribution

| Class | Before Treatment | After Treatment |
|------:|-----------------:|----------------:|
| 0 | 571 | 571 |
| 1 | 597 | 597 |

No imbalance treatment was applied because both classes exceeded the required 35% threshold.

#### Confusion Matrix

| Actual / Predicted | Class 0 | Class 1 |
|-------------------|---------:|---------:|
| Class 0 | 148 | 13 |
| Class 1 | 5 | 126 |

#### Classification Performance

| Metric | Value |
|---------|------:|
| Accuracy | 0.94 |
| Precision | 0.94 |
| Recall | 0.94 |
| F1-score | 0.94 |
| AUC Score | 0.9663 |

#### Decision Threshold Sensitivity

| Threshold | Precision | Recall | F1-score |
|----------:|----------:|-------:|---------:|
| 0.30 | 0.8707 | 0.9771 | 0.9209 |
| 0.40 | 0.8873 | 0.9618 | 0.9231 |
| 0.50 | 0.9065 | 0.9618 | 0.9333 |
| 0.60 | 0.9065 | 0.9618 | 0.9333 |
| 0.70 | 0.9197 | 0.9618 | 0.9403 |

The highest F1-score (**0.9403**) was achieved at a decision threshold of **0.70**.

### Interpretation

#### Precision Formula

Precision = TP / (TP + FP)

Precision measures the proportion of predicted positive instances that are actually positive.

#### Recall Formula

Recall = TP / (TP + FN)

Recall measures the proportion of actual positive instances that are correctly identified by the model.

#### Which Metric Is More Important?

For this house price classification problem, **Precision and Recall are both important**, since the objective is to correctly classify houses above and below the median sale price while minimizing both false positives and false negatives.

#### ROC Curve Interpretation

The ROC curve lies close to the upper-left corner of the graph, indicating that the Logistic Regression model achieves a high True Positive Rate while maintaining a low False Positive Rate across different classification thresholds.

#### AUC Interpretation

The model achieved an **AUC score of 0.9663**, indicating excellent discriminative ability. An AUC close to 1.0 means that the model can effectively distinguish between houses with sale prices above the median and those at or below the median.

### Result

The Logistic Regression model achieved an overall **Accuracy of 94%**, **Precision of 94%**, **Recall of 94%**, **F1-score of 94%**, and an **AUC score of 0.9663**. Since the dataset was already balanced, no class imbalance treatment was necessary. The ROC curve demonstrated excellent classification performance, and the model showed excellent ability to distinguish between the two classes.

## Task 5(b) – Decision-Threshold Sensitivity

### Objective

The objective of this task is to analyse how different decision thresholds affect the classification performance of the Logistic Regression model by comparing Precision, Recall, and F1-score across multiple threshold values.

### Implementation

The predicted probabilities for the positive class were generated using:

```python
model.predict_proba(X_test_scaled)[:, 1]
```

The decision threshold was varied from **0.30** to **0.70** in increments of **0.10**.

For each threshold, the predicted probabilities were converted into class predictions using:

```python
(proba >= threshold).astype(int)
```

The following evaluation metrics were computed at each threshold:

- Precision
- Recall
- F1-score

### Results

#### Decision Threshold Sensitivity

| Threshold | Precision | Recall | F1-score |
|----------:|----------:|-------:|---------:|
| 0.30 | 0.8707 | 0.9771 | 0.9209 |
| 0.40 | 0.8873 | 0.9618 | 0.9231 |
| 0.50 | 0.9065 | 0.9618 | 0.9333 |
| 0.60 | 0.9065 | 0.9618 | 0.9333 |
| 0.70 | 0.9197 | 0.9618 | 0.9403 |

The highest F1-score (**0.9403**) was achieved at a decision threshold of **0.70**.

### Interpretation

#### Precision Formula

Precision = TP / (TP + FP)

Precision measures the proportion of predicted positive instances that are actually positive.

#### Recall Formula

Recall = TP / (TP + FN)

Recall measures the proportion of actual positive instances that are correctly identified by the model.

#### Threshold that Maximises F1-score

Among the evaluated thresholds, **0.70** produced the highest F1-score (**0.9403**). This threshold provides the best balance between Precision and Recall for this dataset.

#### Decision Threshold Interpretation

Increasing the decision threshold improved Precision while maintaining a high Recall. Among the evaluated thresholds, **0.70** produced the highest F1-score, providing the best balance between Precision and Recall.

Increasing the threshold reduces false positive predictions but may increase false negatives. Conversely, lowering the threshold increases Recall but may reduce Precision by producing more false positives.

#### Which Metric Is More Important?

For this house price classification task, **Precision and Recall are both important** because the objective is to correctly classify houses above and below the median sale price while minimizing both false positive and false negative predictions.

#### Threshold Selection

If the objective is to improve **Precision**, the decision threshold should be **raised**, as this reduces false positive predictions. However, increasing the threshold may also increase the number of false negatives, causing some positive cases to be missed.

If the objective is to improve **Recall**, the decision threshold should be **lowered**, which increases the detection of positive cases but may also increase the number of false positives.

### Result

The decision-threshold analysis showed that changing the classification threshold directly affects Precision, Recall, and F1-score. A threshold of **0.70** achieved the highest F1-score, providing the best balance between Precision and Recall for this dataset.

## Task 6 – Regularization Experiment on Logistic Regression

### Objective

The objective of this task is to evaluate the effect of L2 regularization on Logistic Regression by comparing a baseline model (**C = 1.0**) with a strongly regularized model (**C = 0.01**). The models were compared using Precision, Recall, and AUC to determine whether stronger regularization improved classification performance.

### Implementation

A baseline Logistic Regression model with **C = 1.0** was first trained using the scaled training dataset.

A second Logistic Regression model with **C = 0.01** was then trained using the same training and testing datasets. Reducing the value of **C** increases the strength of L2 regularization, encouraging the model to learn smaller coefficient values and reducing the likelihood of overfitting.

Both models were evaluated using:

- Precision
- Recall
- Area Under the ROC Curve (AUC)

### Results

#### Regularization Comparison

| Model | Precision | Recall | AUC |
|--------|----------:|-------:|----:|
| Baseline (C = 1.0) | 0.9065 | 0.9618 | 0.9663 |
| Strong Regularization (C = 0.01) | 0.9407 | 0.9695 | 0.9805 |

### Interpretation

The **C** parameter controls the strength of regularization in Logistic Regression.

- A **larger C** applies weaker regularization, allowing the model coefficients to become larger and fit the training data more closely.
- A **smaller C** applies stronger L2 regularization, shrinking coefficient values and reducing model complexity, which can improve generalization to unseen data.

For this dataset, reducing **C** from **1.0** to **0.01** improved all evaluation metrics. The strongly regularized model achieved higher Precision, higher Recall, and a higher AUC score than the baseline model, indicating better classification performance.

### Result

The Logistic Regression model with **C = 0.01** outperformed the baseline model (**C = 1.0**) across all evaluation metrics. This indicates that stronger L2 regularization improved the model's ability to generalize and produced better overall classification performance on the test dataset.

## Task 7 – Bootstrap Confidence Interval for AUC Difference

### Objective

The objective of this task is to evaluate the reliability of the performance difference between the baseline Logistic Regression model (**C = 1.0**) and the strongly regularized Logistic Regression model (**C = 0.01**) using bootstrap sampling. A 95% confidence interval was computed to determine whether the observed AUC difference is statistically reliable.

### Implementation

A total of **500 bootstrap samples** were generated by randomly sampling the test dataset with replacement using:

```python
np.random.choice(len(y_clf_test), size=len(y_clf_test), replace=True)
```

For each bootstrap sample:

- The corresponding true class labels and predicted probabilities from both Logistic Regression models were selected.
- The AUC score for the baseline model (**C = 1.0**) and the strongly regularized model (**C = 0.01**) was computed.
- The AUC difference was calculated as:

**AUC Difference = AUC(C = 1.0) − AUC(C = 0.01)**

After completing all 500 bootstrap iterations, the following statistics were calculated:

- Mean AUC Difference
- 2.5th Percentile
- 97.5th Percentile

The 2.5th and 97.5th percentiles form the **95% confidence interval** for the AUC difference.

### Results

#### Bootstrap Summary

| Metric | Value |
|--------|------:|
| Number of Bootstrap Samples | 500 |
| Mean AUC Difference (C = 1.0 − C = 0.01) | -0.014824 |
| Lower Bound (2.5th Percentile) | -0.030090 |
| Upper Bound (97.5th Percentile) | -0.002388 |

### Interpretation

The mean AUC difference was calculated as:

**AUC(C = 1.0) − AUC(C = 0.01) = -0.014824**

The 95% confidence interval for the AUC difference is:

**[-0.030090, -0.002388]**

Since the **95% confidence interval excludes zero**, the observed difference in AUC is statistically significant and is unlikely to have occurred due to random sampling variation.

The entire confidence interval is **negative**, indicating that the Logistic Regression model with **C = 0.01** consistently achieved a higher AUC than the baseline model (**C = 1.0**) across different bootstrap samples.

Therefore, the improvement of the **C = 0.01** model is considered reliable and is expected to generalize consistently across different samples drawn from the same data distribution.

### Result

Bootstrap analysis using **500 bootstrap samples** produced a mean AUC difference of **-0.014824** with a **95% confidence interval of [-0.030090, -0.002388]**. Because the confidence interval **does not include zero**, the difference in model performance is statistically reliable. The results demonstrate that the Logistic Regression model with **C = 0.01** consistently outperformed the baseline model (**C = 1.0**) on this dataset.

# Part 2 Summary

In this part of the project, supervised machine learning models were successfully developed and evaluated using the cleaned House Prices dataset from Part 1.

The main outcomes of this part are:

- The cleaned dataset was prepared by defining the feature matrix (**X**), regression target (**y_reg**), and binary classification target (**y_clf**).
- Categorical features were encoded using Label Encoding for ordinal variables and One-Hot Encoding for nominal variables.
- A leak-free train-test split and feature scaling pipeline were implemented using `StandardScaler`.
- Linear Regression achieved an **R² score of 0.6766**, while Ridge Regression improved the performance with an **R² score of 0.7252**, demonstrating the effectiveness of L2 regularization.
- Logistic Regression achieved an **Accuracy of 94%** and an **AUC score of 0.9663**, indicating excellent classification performance.
- Decision-threshold analysis showed that a threshold of **0.70** produced the highest F1-score, providing the best balance between Precision and Recall.
- The regularization experiment showed that the Logistic Regression model with **C = 0.01** achieved better Precision, Recall, and AUC than the baseline model (**C = 1.0**).
- Bootstrap analysis using **500 bootstrap samples** confirmed that the improvement of the **C = 0.01** model was statistically reliable, as the **95% confidence interval for the AUC difference excluded zero**.

Overall, this part successfully demonstrated the complete supervised machine learning workflow, including data preparation, preprocessing, regression modelling, classification, model evaluation, regularization analysis, and statistical validation of model performance.
