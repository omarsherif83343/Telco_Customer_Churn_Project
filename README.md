# 📊 Customer Churn Prediction using K-Nearest Neighbors (KNN)

A Machine Learning classification project that predicts whether a telecom customer is likely to leave the company (Churn) using the **K-Nearest Neighbors (KNN)** algorithm.

---

# 📁 Dataset

**Dataset:** Telco Customer Churn Dataset

The dataset contains customer information such as:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Internet Service
- Contract Type
- Payment Method
- Monthly Charges
- Total Charges
- Churn (Target)

---

# 🎯 Project Objective

The goal of this project is to build a Machine Learning model that predicts customer churn based on customer information.

---

# 🛠️ Project Workflow

## Stage 1 — Data Exploration (EDA)

- Load Dataset
- Display first rows
- Dataset Information
- Statistical Summary
- Check Missing Values
- Count Churned Customers
- Count Active Customers

---

## Stage 2 — Data Cleaning

- Handle missing values
- Convert TotalCharges to numeric
- Remove duplicate rows

---

## Stage 3 — Data Preprocessing

### Label Encoding

Applied on:

- Gender
- Partner
- Dependents
- PhoneService
- PaperlessBilling
- Churn

### One-Hot Encoding

Applied on:

- MultipleLines
- InternetService
- Contract
- PaymentMethod
- OnlineSecurity
- OnlineBackup
- DeviceProtection
- TechSupport
- StreamingTV
- StreamingMovies

---

## Stage 4 — Feature Scaling

StandardScaler was applied before training the KNN model.

---

## Stage 5 — Model Training

Algorithm:

- K-Nearest Neighbors (KNN)

Different K values were tested:

```python
1,3,5,7,9,11,13,15,17,19,21
```

The best K was selected automatically.

---

# 📈 Model Performance

**Best K**

```text
21
```

**Accuracy**

```text
77%
```

---

# 📊 K vs Accuracy

![K vs Accuracy](images/knn_accuracy.png)

---

# 🔥 Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

---

# 📋 Classification Report

The project evaluates the model using:

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

# 🧰 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn

---

# 📂 Project Structure

```text
Customer-Churn-KNN/
│
├── customer_churn_project.py
├── README.md
├── requirements.txt
├── telco-customer-churn.csv
└── images/
    ├── knn_accuracy.png
    └── confusion_matrix.png
```

---

# 🚀 Future Improvements

- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- Random Forest

Compare all models using:

- Accuracy
- Precision
- Recall
- F1-Score

---

# 👨‍💻 Author

**Omar Sherif Ahmed Mohamed**

Data Analyst | AI Engineer

GitHub: https://github.com/omarsherif83343

LinkedIn: https://www.linkedin.com/in/omar-sherif-860a852a3/
