# Financial-Anomaly-Detection-
This project predicts weather a transaction is anomaly or normal

# Financial Anomaly Detection using Machine Learning

A machine learning project for detecting potentially fraudulent financial transactions using **Anomaly Detection** techniques. The project uses **Isolation Forest**, an unsupervised machine learning algorithm, to identify transactions that behave differently from normal transaction patterns.

## Project Overview

Financial fraud detection is a highly imbalanced classification problem because fraudulent transactions are usually much fewer than legitimate transactions.

Instead of relying only on traditional classification, this project explores **anomaly detection** to identify unusual transactions based on their feature patterns.

The main algorithm used in this project is:

* **Isolation Forest**

The trained model can be saved and deployed through a **Streamlit web application**, allowing users to enter transaction information and receive an anomaly/fraud prediction.

## Objectives

The main objectives of this project are:

* Detect unusual financial transactions.
* Apply an unsupervised anomaly detection algorithm.
* Handle highly imbalanced financial data.
* Preprocess and scale numerical features.
* Train an Isolation Forest model.
* Evaluate anomaly detection performance.
* Save the trained model for deployment.
* Build a simple Streamlit interface for predictions.

## Dataset

The project uses a financial transaction dataset containing transaction-related features and a target `Class` column.

The dataset contains highly imbalanced classes:

* `0` → Normal transaction
* `1` → Fraudulent transaction

Fraudulent transactions represent only a very small portion of the complete dataset, making this a challenging anomaly detection problem.

### Dataset Features

The dataset contains several anonymized numerical features along with:

* `Amount`
* `Class`

The anonymized features represent transformed transaction information.

For model training, the target `Class` column is not used as an input feature because the purpose of anomaly detection is to identify unusual transactions.

## Machine Learning Approach

### 1. Data Loading

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("creditcard.csv")
```

### 2. Data Exploration

Exploratory Data Analysis was performed to understand:

* Dataset size
* Feature distributions
* Missing values
* Duplicate records
* Class distribution
* Transaction amount distribution
* Potential outliers

### 3. Feature Selection

The target variable is separated from the input features.

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

The `Class` column is excluded from model training because it represents the actual fraud label.

### 4. Feature Scaling

Financial transaction features can have very different numerical ranges.

Scaling is therefore applied to appropriate numerical features.

For example:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Scaling helps the machine learning model work with features on comparable numerical scales.

## Isolation Forest

The main algorithm used in this project is **Isolation Forest**.

Isolation Forest is an unsupervised anomaly detection algorithm that works by isolating observations.

The basic idea is:

> Anomalous observations are usually easier to isolate than normal observations.

The model randomly selects features and split values to construct isolation trees. Transactions that require fewer splits to isolate are considered more likely to be anomalies.

### Model

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    n_estimators=100,
    max_samples="auto",
    contamination=0.5,
    max_features=1.0,
    random_state=42
)

model.fit(X_scaled)
```

### Important Hyperparameters

#### `n_estimators`

Number of isolation trees used by the model.

```text
n_estimators=100
```

More trees can generally provide more stable predictions, although they also increase computation time.

#### `max_samples`

Number of samples used to build each tree.

```text
max_samples="auto"
```

With `"auto"`, scikit-learn determines an appropriate sample size automatically.

#### `contamination`

Expected proportion of anomalies in the dataset.

```text
contamination=0.5
```

This parameter strongly affects the number of observations classified as anomalies and should be selected carefully based on the actual problem and validation results.

#### `max_features`

Number or proportion of features considered when building each tree.

```text
max_features=1.0
```

This means all available features can be considered.

## Prediction

Isolation Forest produces predictions such as:

```text
1  → Normal
-1 → Anomaly
```

Example:

```python
predictions = model.predict(X_scaled)
```

The model output can then be mapped to the application's desired labels.

For example:

```python
result = "Fraud/Anomaly" if prediction == -1 else "Normal"
```

## Model Evaluation

Because fraud detection datasets are usually highly imbalanced, **accuracy alone is not a reliable metric**.

The project evaluates the model using metrics such as:

* Precision
* Recall
* F1-score
* Confusion Matrix

### Confusion Matrix

The confusion matrix helps analyze:

* True Positives
* True Negatives
* False Positives
* False Negatives

For fraud detection, **Recall is particularly important** because failing to detect an actual fraudulent transaction can be costly.

However, precision is also important because classifying too many legitimate transactions as fraud creates a large number of false alerts.

Therefore, the model should be evaluated using a balance between precision and recall rather than accuracy alone.

## Model Saving

After training, the model and preprocessing objects can be saved using `joblib`.

```python
import joblib

joblib.dump(model, "isolation_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

These files can later be loaded by the Streamlit application.

```python
model = joblib.load("isolation_forest_model.pkl")
scaler = joblib.load("scaler.pkl")
```

## Streamlit Deployment

A Streamlit frontend can be used to provide an interactive interface for the trained model.

The application allows users to enter transaction information and receive a prediction from the trained anomaly detection model.

Example:

```text
User Input
    ↓
Data Preprocessing
    ↓
Feature Scaling
    ↓
Isolation Forest Model
    ↓
Prediction
    ↓
Normal / Anomaly
```

## Project Structure

```text
anomaly-detection/
│
├── data/
│   └── creditcard.csv
│
├── model/
│   ├── isolation_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── anomaly_detection.ipynb
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Google Colab
* Jupyter Notebook

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/anomaly-detection.git
```

Navigate to the project directory:

```bash
cd anomaly-detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Running the Streamlit Application

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

## Key Challenges

This project addresses several challenges associated with financial fraud detection:

### Highly Imbalanced Data

Fraudulent transactions are significantly less common than legitimate transactions.

### False Positives

A model that identifies too many normal transactions as anomalies can create unnecessary fraud alerts.

### False Negatives

Missing a fraudulent transaction can have significant financial consequences.

### Feature Scaling

Financial datasets can contain features with very different numerical ranges, making appropriate preprocessing important.

### Contamination Selection

The Isolation Forest `contamination` parameter has a major impact on the number of transactions classified as anomalies and should not be selected arbitrarily.

## Future Improvements

Potential improvements include:

* Hyperparameter tuning.
* Better contamination estimation.
* Comparing Isolation Forest with One-Class SVM.
* Implementing an Autoencoder.
* Comparing multiple anomaly detection algorithms.
* Handling class imbalance more carefully.
* Improving precision and recall.
* Adding ROC-AUC and PR-AUC analysis.
* Creating a more advanced Streamlit dashboard.
* Adding transaction-risk scores.
* Deploying the application online.

## Learning Outcomes

Through this project, I learned how to:

* Work with highly imbalanced financial datasets.
* Perform exploratory data analysis.
* Preprocess machine learning data.
* Apply feature scaling.
* Understand anomaly detection.
* Implement Isolation Forest.
* Tune machine learning hyperparameters.
* Evaluate models using precision, recall, F1-score, and confusion matrices.
* Save trained ML models using Joblib.
* Build a Streamlit interface for ML deployment.

## Conclusion

This project demonstrates how machine learning can be used to identify unusual financial transactions.

The **Isolation Forest** algorithm provides an effective approach for anomaly detection because it does not require every transaction to be explicitly labeled during model training.

However, fraud detection is a high-impact problem where model performance should be evaluated carefully. In particular, the trade-off between **precision, recall, false positives, and false negatives** must be considered before using the model in a real-world financial system.

## Author

**Muhammad Tayyab**

Data Science & Machine Learning Student

Interested in:

* Data Science
* Machine Learning
* Artificial Intelligence
* Anomaly Detection
* ML Model Deployment
* Generative AI

---

⭐ If you find this project useful, consider giving the repository a star.

