# SMART PREDICTIVE MAINTENANCE SYSTEM
## 📌 Overview

The Smart Predictive Maintenance System is a Machine Learning based project developed to predict industrial machine failures before they occur.

The main objective of this project is to support smart industries and smart city infrastructure by monitoring machine health conditions and providing early failure warnings.

The system follows a two-stage prediction approach:

**Stage 1:**
Predicts whether a machine is healthy or requires maintenance.

**Stage 2:**
Identifies the specific failure type when a failure occurs.

The system classifies machines into:

🟢 Healthy Machine

🔴 Machine Failure Detected


For failed machines, the system identifies:
⚙️ Tool Wear Failure (TWF)

🌡️ Heat Dissipation Failure (HDF)

⚡ Power Failure (PWF)

🔧 Overstrain Failure (OSF)

# 🎯 Objective

The primary objective of this project is to build an intelligent predictive maintenance system that can detect machine failures early and assist industries in reducing unexpected downtime.

The project aims to:

⚙️ Monitor machine health parameters

📉 Reduce unexpected machine breakdowns

🚨 Provide early failure warnings

🔧 Improve maintenance planning

🏙 Support smart city industrial automation

🤖 Apply Machine Learning for real-time prediction


---

# 📂 Dataset

The project uses the AI4I 2020 Predictive Maintenance Dataset.

The dataset contains machine operational parameters such as:

- Machine Type
- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear


The dataset also contains different failure categories:

- TWF
- HDF
- PWF
- OSF
- RNF


---

# 📊 Data Preprocessing

Several preprocessing techniques were applied before model training.

## 🧹 Data Cleaning

Removed unnecessary columns:

- UDI
- Product ID


## 🔤 Encoding

Machine type values were converted:

L → 0

M → 1

H → 2


## 📉 Outlier Handling

Outliers were detected using the IQR method.

Instead of removing data points, capping technique was applied to maintain dataset size.


## ⚖️ Handling Imbalanced Data

The dataset contained fewer failure samples compared to healthy samples.

SMOTE (Synthetic Minority Oversampling Technique) was applied to balance the classes.


---

# 🤖 Machine Learning Models Used

Multiple classification algorithms were implemented and compared.

Models used:

🌳 Decision Tree Classifier

🌲 Random Forest Classifier

📈 Logistic Regression

🤝 Voting Classifier


The models were evaluated using:

- Accuracy Score
- Precision
- Recall
- F1 Score
- Confusion Matrix


---

# 🏆 Stage 1: Machine Failure Prediction

Problem Type:

Binary Classification


Classes:

0 → Healthy

1 → Failure


Best Performing Model:

Voting Classifier


Performance:

Accuracy: 96.9%


The model successfully identifies whether a machine requires maintenance or is operating normally.


---

# 🔎 Stage 2: Failure Type Prediction

Stage 2 is performed only when Stage 1 detects a failure.

Problem Type:

Multiclass Classification


Algorithm Used:

Random Forest Classifier


Feature Engineering was applied by creating:

## 🌡 Temperature Difference

Process Temperature - Air Temperature


## ⚡ Power Consumption

Torque × Rotational Speed


Final Features:

- Machine Type
- Temperature
- Speed
- Torque
- Tool Wear
- Temperature Difference
- Power


Performance:

Accuracy: 92.6%


---

# 📈 Model Evaluation

The models were evaluated using:

## 📊 Confusion Matrix

Used to analyze actual and predicted failure categories.


## 📝 Classification Report

Includes:

- Precision
- Recall
- F1 Score

# 💻 Streamlit Application

A user-friendly Streamlit dashboard was developed.

Features:

🏭 Machine Selection

⚙️ Parameter Input

🔍 Failure Prediction

📊 Failure Probability Visualization

⚠️ Warning Messages

🚦 Risk Level Detection

🔧 Maintenance Recommendation


The application provides real-time machine health monitoring.




# 🛠 Technologies Used

Python

Pandas

NumPy

Scikit-Learn

Imbalanced-learn

Matplotlib

Seaborn

Streamlit

# 🚀 Future Improvements

Improve prediction using advanced ML algorithms

Integrate real-time IoT sensor data

Deploy the application on cloud platforms

Add live machine monitoring dashboard

Implement deep learning based failure prediction


