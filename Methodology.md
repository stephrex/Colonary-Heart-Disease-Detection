# Methodology
## Coronary Heart Disease Prediction Using Machine Learning

---

## 1. Introduction

Coronary heart disease (CHD) is one of the leading causes of mortality worldwide, and early detection plays a major role in reducing patient risk. The objective of this research is to build a machine learning system that can predict whether a patient is likely to have coronary heart disease based on a set of clinical and demographic measurements. This document describes the methodology followed throughout the project, covering data gathering, data understanding, exploratory analysis, preprocessing, model development, and model evaluation.

The work was structured as a supervised binary classification problem in which a patient record is classified as either healthy (0) or showing presence of heart disease (1). Four models were trained and compared: Logistic Regression, Support Vector Machine (SVM), Random Forest, and an Artificial Neural Network (ANN).

---

## 2. Methodology Overview

The overall methodology followed the standard machine learning research pipeline. Each stage produced an output that fed into the next stage, ensuring that the modelling phase received clean, well-structured data. The stages were:

1. Data gathering
2. Dataset description and problem understanding
3. Exploratory data analysis (EDA)
4. Data preprocessing (cleaning, imputation, encoding, scaling)
5. Train–test split
6. Model development (Logistic Regression, SVM, Random Forest, ANN)
7. Model evaluation
8. Deployment of the trained models in an inference dashboard

---

## 3. Data Gathering

The dataset used in this research is the **UCI Heart Disease Dataset**, a well-known and widely used public dataset in medical machine learning research. The data was downloaded as a single CSV file (`heart_disease_uci.csv`) and placed inside the project's `Dataset/` folder. The dataset combines records from multiple medical institutions, with the Cleveland subset being the most complete and most commonly referenced in published studies.

The raw file contained 920 patient records described by 14 primary clinical attributes plus a unique identifier and a location field, giving 16 input columns and one target column.

---

## 4. Dataset Description

The dataset is multivariate and consists of demographic, clinical, and diagnostic features. The columns are summarised below.

| # | Column     | Description |
|---|-----------|-------------|
| 1 | id         | Unique patient identifier. |
| 2 | age        | Age of the patient in years. |
| 3 | sex        | Patient sex (Male / Female). |
| 4 | dataset    | Location where the record was collected. |
| 5 | cp         | Chest pain type (typical angina, atypical angina, non-anginal, asymptomatic). |
| 6 | trestbps   | Resting blood pressure (mm Hg). |
| 7 | chol       | Serum cholesterol (mg/dl). |
| 8 | fbs        | Fasting blood sugar > 120 mg/dl (True / False). |
| 9 | restecg    | Resting electrocardiographic result (normal, ST-T abnormality, LV hypertrophy). |
| 10 | thalch    | Maximum heart rate achieved. |
| 11 | exang     | Exercise-induced angina (True / False). |
| 12 | oldpeak   | ST depression induced by exercise relative to rest. |
| 13 | slope     | Slope of the peak exercise ST segment (upsloping, flat, downsloping). |
| 14 | ca        | Number of major vessels (0–3) coloured by fluoroscopy. |
| 15 | thal      | Thalassemia category (normal, fixed defect, reversible defect). |
| 16 | num       | Original target — disease severity 0–4. |

The original target variable `num` represents the severity of heart disease on a scale from 0 (no disease) to 4 (very severe disease). For this research, the problem was reduced to a binary classification task because the goal is presence detection rather than severity grading. A new binary `target` column was therefore created where a value of 0 indicated no heart disease and a value of 1 indicated presence of disease at any severity level.

---

## 5. Exploratory Data Analysis (EDA)

Before any preprocessing, exploratory data analysis was performed to understand the distribution of each variable and how it related to the target. This stage produced the visualisations that informed several preprocessing decisions later in the pipeline.

### 5.1 Target Distribution

The distribution of the binary target column was inspected to check whether the dataset was balanced. The data was found to be relatively balanced, with approximately **55.3 %** of records representing patients with heart disease and **44.7 %** representing healthy patients. Because the imbalance was mild, no resampling techniques (such as SMOTE) were required.

> **[INSERT IMAGE HERE]** — Target distribution plot (multi-class `num` countplot, and binary `target` countplot). Caption: *Distribution of heart disease cases in the dataset (multi-class severity and binary form).*

### 5.2 Age Distribution and Age vs Heart Disease

The age distribution showed that most patients in the dataset fell between **47 and 60 years**, with the median around 54 years. A boxplot comparing age against the target indicated that patients with heart disease tended to be slightly older than healthy patients, supporting the well-established clinical relationship between age and CHD risk.

> **[INSERT IMAGE HERE]** — Age distribution histogram with KDE overlay. Caption: *Age distribution of patients in the dataset.*

> **[INSERT IMAGE HERE]** — Boxplot of age grouped by target. Caption: *Age distribution between healthy and diseased patients.*

### 5.3 Chest Pain Type vs Heart Disease

A countplot grouped by chest pain type revealed that patients labelled as **asymptomatic** were the most likely to have heart disease, while patients with **atypical angina** and **non-anginal pain** were more likely to be healthy. This was an important observation because it shows that the absence of typical chest pain is not necessarily a sign of safety.

> **[INSERT IMAGE HERE]** — Countplot of chest pain type vs target. Caption: *Relationship between chest pain type and heart disease.*

### 5.4 Maximum Heart Rate (thalch)

A boxplot of maximum heart rate against the target showed that healthy patients tended to achieve a **higher maximum heart rate** than diseased patients. This is consistent with the medical understanding that reduced exercise capacity is associated with cardiovascular disease.

> **[INSERT IMAGE HERE]** — Boxplot of maximum heart rate vs target. Caption: *Maximum heart rate achieved by healthy vs diseased patients.*

### 5.5 Cholesterol Level (chol)

The cholesterol boxplot showed overlapping distributions between healthy and diseased patients, suggesting that cholesterol alone is **not a strong individual predictor** of heart disease in this dataset. Several zero values were also identified, which are clinically impossible and were later treated as missing values.

> **[INSERT IMAGE HERE]** — Boxplot of cholesterol vs target. Caption: *Serum cholesterol level distribution by target.*

### 5.6 Resting Blood Pressure (trestbps)

The blood pressure boxplot showed only a slight upward shift for diseased patients. As with cholesterol, a small number of zero readings were present and were treated as missing.

> **[INSERT IMAGE HERE]** — Boxplot of resting blood pressure vs target. Caption: *Resting blood pressure vs heart disease.*

### 5.7 Exercise-Induced Angina (exang)

The countplot for exercise-induced angina showed a **strong relationship** with the target: patients who experienced angina during exercise were considerably more likely to have heart disease.

> **[INSERT IMAGE HERE]** — Countplot of exang vs target. Caption: *Exercise-induced angina vs heart disease.*

### 5.8 ST Depression (oldpeak)

The oldpeak boxplot showed clearly **higher ST depression values for diseased patients**, indicating that this feature carries strong predictive power. This observation was later confirmed in the random forest feature importance analysis.

> **[INSERT IMAGE HERE]** — Boxplot of oldpeak vs target. Caption: *ST depression (oldpeak) vs heart disease.*

### 5.9 Summary of EDA Findings

The exploratory analysis confirmed the following:
- The dataset is mildly imbalanced and does not require resampling.
- Age, chest pain type, exercise-induced angina, maximum heart rate, and ST depression are the most clearly discriminative features.
- Cholesterol and resting blood pressure have weaker individual signals.
- Several clinically impossible values (zeros in cholesterol and blood pressure) need to be cleaned.
- A substantial number of missing values exist in `slope`, `ca`, and `thal`.

---

## 6. Data Preprocessing

The preprocessing pipeline transformed the raw CSV file into a clean, scaled, machine-readable dataset.

### 6.1 Dropping Unnecessary Columns

The columns `id` and `dataset` were dropped because they do not carry predictive information. `id` is only a row identifier, and `dataset` represents the collection site, which is not relevant to the clinical prediction task.

### 6.2 Missing Value Inspection

After loading the dataset, a missing value count was computed for every column. The result was as follows:

| Column | Missing values |
|---|---|
| trestbps | 59 |
| chol | 30 |
| fbs | 90 |
| restecg | 2 |
| thalch | 55 |
| exang | 55 |
| oldpeak | 62 |
| slope | 309 |
| ca | 611 |
| thal | 486 |

The columns `slope`, `ca`, and `thal` had the highest proportions of missing values. Because dropping these columns would lose useful clinical information, they were retained and imputed instead.

### 6.3 Replacing Invalid Tokens

The dataset contained the string `"?"` to indicate missing entries in some columns. These were replaced with `NaN` so that they could be detected by pandas and processed through the standard imputation pipeline. After this step, the affected columns (`trestbps`, `chol`, `thalch`, `oldpeak`, `ca`) were also explicitly converted to numeric types.

### 6.4 Handling Missing Values

A **hybrid imputation strategy** was used:

- **Numerical columns** were imputed using the **mean** of the column. The mean was chosen because the distributions of the numerical features were not extremely skewed after invalid-zero treatment, and mean imputation preserves the central tendency of the data.
- **Categorical columns** were imputed using the **mode** (most frequent value), which preserves the original class distribution.

### 6.5 Outlier and Invalid Value Treatment

After visual inspection of the boxplots and summary statistics, several patient records were found with `chol = 0` and `trestbps = 0`. These values are clinically impossible (a patient cannot have zero cholesterol or zero blood pressure), so they were treated as missing data. They were replaced with `NaN` and re-imputed using the **median** of the respective columns, which is more robust to extreme outliers than the mean.

### 6.6 Encoding Categorical Variables

Categorical variables (`sex`, `cp`, `restecg`, `slope`, `thal`) were transformed into numerical format using **one-hot encoding** with the first category dropped (`drop_first=True`). This avoids the dummy-variable trap while ensuring that the categorical information is preserved in a form that any machine learning algorithm can consume.

### 6.7 Feature and Target Separation

After encoding, the feature matrix `X` was formed by dropping the columns `target` (the binary classification target) and `num` (the original multi-class severity column, which would leak the answer into the features). The target vector `y` was set to the binary `target` column.

The final feature matrix consisted of **18 columns** (8 numerical and 10 one-hot binary columns).

### 6.8 Feature Scaling

The numerical features in the dataset were on very different scales — for example, age (28–77) compared to cholesterol (0–600+) compared to oldpeak (–2.6 to 6.2). To prevent the larger-magnitude features from dominating distance-based and gradient-based models, the entire feature matrix was scaled using **StandardScaler** from scikit-learn, which transforms each feature to have mean 0 and standard deviation 1.

The fitted scaler was saved to `Utils/scaler.pkl` so that the exact same scaling could be reapplied at inference time.

### 6.9 Saving the Cleaned Dataset

After all preprocessing steps were completed, the cleaned dataset was saved as `cleaned_heart_disease_data.csv` inside `Dataset/Processed Datasets/`. The scaled feature matrix and target vector were also saved as NumPy files (`X_scaled.npy` and `y.npy`) for reproducibility.

The cleaned dataset had:
- **920 rows** (no records were dropped)
- **18 features** after encoding
- **Target distribution: 55.3 % positive, 44.7 % negative**
- **No missing values**

---

## 7. Train–Test Split

The cleaned and scaled dataset was split into training and test sets using an **80 / 20 split**. The split was stratified on the target variable to ensure that both subsets contained the same class proportions, and a fixed `random_state=42` was used for reproducibility.

| Split | Number of records |
|---|---|
| Training set | 736 |
| Test set | 184 |

---

## 8. Model Development

Four machine learning models were trained on the same training set. The choice of models was deliberately diverse, ranging from a simple linear baseline to an ensemble model and finally a neural network. This allowed a fair comparison between model families.

### 8.1 Logistic Regression

Logistic Regression was used as the **baseline classifier** because of its simplicity, interpretability, and strong performance on binary classification problems with linearly separable structure.

**Hyperparameters:**
- `max_iter = 1000` (to ensure convergence on the scaled dataset)
- All other parameters left at scikit-learn defaults (`solver='lbfgs'`, L2 regularisation, `C=1.0`).

The trained model was saved to `Utils/logistic_regression_model.pkl`.

### 8.2 Support Vector Machine (SVM)

A Support Vector Machine was chosen for its ability to handle complex, non-linear decision boundaries through the kernel trick. It is particularly effective in moderate-dimensional datasets such as this one.

**Hyperparameters:**
- `kernel = 'rbf'` (Radial Basis Function kernel — captures non-linear separations)
- `probability = True` (enables probability estimates required for ROC AUC and the inference dashboard)
- Other parameters left at scikit-learn defaults (`C=1.0`, `gamma='scale'`).

The trained model was saved to `Utils/svm_model.pkl`.

### 8.3 Random Forest

The Random Forest classifier was selected because of its strong performance on tabular clinical data, its robustness to outliers, and its ability to provide feature importance estimates that can be used for interpretation.

**Hyperparameters:**
- `n_estimators = 200` (number of decision trees in the forest)
- `max_depth = 10` (maximum depth of each tree, to control overfitting)
- `random_state = 42` (reproducibility)

The trained model was saved to `Utils/random_forest_model.pkl`.

### 8.4 Artificial Neural Network (ANN)

A small feed-forward neural network was trained to capture more complex non-linear interactions between features.

**Architecture:**

| Layer | Type | Units / Rate | Activation |
|---|---|---|---|
| 1 | Input | 18 features | — |
| 2 | Dense | 64 | ReLU |
| 3 | Dropout | 0.3 | — |
| 4 | Dense | 32 | ReLU |
| 5 | Dropout | 0.3 | — |
| 6 | Dense (output) | 1 | Sigmoid |

The model contained **3,329 trainable parameters**.

**Training configuration:**
- Optimiser: **Adam**
- Loss: **Binary cross-entropy**
- Metric: **Accuracy**
- Epochs: **50**
- Batch size: **32**
- Validation split: **20 %** of the training set

The training process was monitored using the loss and accuracy curves on both the training and validation splits.

> **[INSERT IMAGE HERE]** — `Plots/ann_model_structure.png`. Caption: *Architecture of the trained Artificial Neural Network.*

> **[INSERT IMAGE HERE]** — `Plots/ann_training_loss_curve.png`. Caption: *Training and validation loss across 50 epochs.*

> **[INSERT IMAGE HERE]** — `Plots/ann_training_accuracy_curve.png`. Caption: *Training and validation accuracy across 50 epochs.*

By the final epoch, the network reached a **training accuracy of approximately 89.5 %** and a **validation accuracy of approximately 85.1 %**, with no sign of severe overfitting due to the dropout layers.

The trained model was saved as `Utils/ann_model.h5`.

---

## 9. Model Evaluation

After training, all four models were evaluated on the held-out test set of 184 records using the metrics that were originally specified in the project proposal: **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.

### 9.1 Evaluation Metrics

- **Accuracy** — proportion of correctly classified patients out of the total.
- **Precision** — proportion of positive predictions that were actually correct. High precision means few false positives (fewer healthy patients incorrectly flagged as diseased).
- **Recall (Sensitivity)** — proportion of actually diseased patients that were correctly identified. High recall is particularly important in medical screening because false negatives are dangerous.
- **F1 Score** — harmonic mean of precision and recall, balancing both concerns.
- **ROC-AUC** — area under the receiver operating characteristic curve, which measures the model's ability to rank diseased patients above healthy ones across all decision thresholds.

### 9.2 Results Summary

The evaluation results for all four models are summarised in the table below.

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.43 % | 80.56 % | 85.29 % | 82.86 % | 89.26 % |
| SVM | 82.61 % | 80.17 % | 91.18 % | 85.32 % | 90.67 % |
| **Random Forest** | **84.24 %** | **82.88 %** | **90.20 %** | **86.39 %** | **90.78 %** |
| ANN | 83.15 % | 81.42 % | 90.20 % | 85.58 % | 89.93 % |

The Random Forest classifier achieved the **highest overall performance**, leading on Accuracy, Precision, F1 Score, and ROC AUC. SVM achieved the highest Recall but at the cost of a slightly lower precision.

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/model_comparison_with_values.png`. Caption: *Performance comparison of all four models across the five evaluation metrics.*

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/model_performance_heatmap.png`. Caption: *Heatmap representation of the model performance metrics.*

### 9.3 Confusion Matrices

The confusion matrices below describe how each model distributed its predictions between true positives, true negatives, false positives, and false negatives on the test set.

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/Logistic Regression_confusion_matrix.png`. Caption: *Logistic Regression confusion matrix.*

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/SVM_confusion_matrix.png`. Caption: *SVM confusion matrix.*

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/Random Forest_confusion_matrix.png`. Caption: *Random Forest confusion matrix.*

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/ANN_confusion_matrix.png`. Caption: *ANN confusion matrix.*

### 9.4 Classification Reports

Per-class precision, recall, F1 score, and support values for each model are visualised in the classification report figures below.

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/Logistic Regression_classification_report.png`. Caption: *Logistic Regression classification report.*

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/SVM_classification_report.png`. Caption: *SVM classification report.*

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/Random Forest_classification_report.png`. Caption: *Random Forest classification report.*

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/ANN_classification_report.png`. Caption: *ANN classification report.*

### 9.5 ROC Curves

The ROC curve plots the true positive rate against the false positive rate at different decision thresholds. A higher area under the curve (AUC) indicates a better discriminator.

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/roc_curve_comparison.png`. Caption: *ROC curves for all four models with their corresponding AUC values.*

The ROC curves confirm the ranking observed in the metric table: Random Forest achieved the highest AUC (0.908), closely followed by SVM (0.907), with the ANN (0.899) and Logistic Regression (0.893) slightly behind.

### 9.6 Precision–Recall Curves

The precision–recall curve is particularly informative for medical classification tasks because it focuses on the positive class. The curves show how each model trades precision for recall as the decision threshold changes.

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/precision_recall_curve.png`. Caption: *Precision–Recall curves for all four models.*

### 9.7 Feature Importance

Because the Random Forest was the best-performing model, its feature importance values were extracted and plotted. This indicates which features contributed most to the prediction.

> **[INSERT IMAGE HERE]** — `Plots/Evaluation_Plots/random_forest_feature_importance.png`. Caption: *Feature importance ranking produced by the Random Forest classifier.*

The most influential features, as expected from the exploratory analysis, were related to **chest pain type, ST depression (oldpeak), number of major vessels (ca), maximum heart rate (thalch), and age**.

---

## 10. Discussion

The results show that classical machine learning models perform well on this dataset, with the Random Forest classifier producing the strongest overall results. Several observations follow from the analysis:

1. **Recall is high across all models** (85 – 91 %), which is desirable in a medical screening context where missed diagnoses are more costly than false alarms.
2. **The Random Forest model's advantage** comes primarily from its higher precision and accuracy, while its recall is statistically comparable to SVM and ANN. This makes it the best balance for deployment.
3. **The ANN did not outperform the ensemble model.** This is consistent with the general observation that tree-based ensembles tend to outperform small neural networks on tabular clinical datasets, especially when the dataset size is moderate (under 1000 samples).
4. **Cholesterol and resting blood pressure** were less predictive on their own than expected, but they still contributed to the ensemble through interactions with other features.
5. **The dataset is moderately balanced**, which avoided the need for resampling techniques and made the metric comparisons reliable.

---

## 11. Deployment

The trained models, the fitted scaler, and the cleaned dataset were packaged into a Streamlit-based web dashboard (`app.py`). The dashboard supports:

- **Single-patient inference** through an interactive form.
- **Batch inference** through CSV upload, with downloadable predictions.
- **Multi-model selection**, with Random Forest as the default recommended model.
- **Model insights view** that displays the comparison table and all evaluation plots.

This provides a usable interface for demonstrating the research and for downstream clinical exploration.

---

## 12. Summary

This research designed and evaluated four machine learning models for the prediction of coronary heart disease on the UCI Heart Disease Dataset. After thorough exploratory analysis and preprocessing, all four models achieved accuracies above 80 %, with the **Random Forest classifier producing the strongest overall results (Accuracy 84.24 %, F1 86.39 %, ROC AUC 90.78 %)**. The pipeline is fully reproducible: the cleaned dataset, fitted scaler, and trained models are all stored alongside the notebook, and the final models are exposed through a working inference dashboard.
