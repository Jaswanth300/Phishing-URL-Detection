# 🔐 Phishing URL Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Internship%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This project implements a **machine learning-based phishing website detection system** using a real-world Kaggle dataset.  

It benchmarks multiple ML models and optimizes performance using hyperparameter tuning.

The system classifies websites as:

- **0 → Legitimate**
- **1 → Phishing**

---

## 📊 Dataset Information

- Source: Kaggle Phishing Website Dataset
- Total Features: 9 security-based attributes
- Target Column: `Result`
- Balanced dataset (Phishing vs Legitimate)

---

## 🧠 Models Implemented & Benchmarked

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 88.2% | 0.874 | 0.892 | 0.883 |
| 🔥 Tuned Random Forest | **90.1%** | 0.905 | 0.897 | 0.901 |

---

## ⚙️ Hyperparameter Tuning (GridSearchCV)

Best Parameters Found:
{'max_depth': None,
'min_samples_leaf': 1,
'min_samples_split': 5,
'n_estimators': 100}

Tuning improved model accuracy beyond baseline Random Forest.

---

## 🔍 Feature Importance (Random Forest)

| Feature | Importance |
|----------|------------|
| SFH | 40% |
| popUpWidnow | 15% |
| SSLfinal_State | 11% |
| Request_URL | 10% |
| URL_of_Anchor | 8% |
| URL_Length | 6% |
| web_traffic | 4% |
| age_of_domain | 2% |
| having_IP_Address | 1% |

### Insight:
Server Form Handler (SFH) is the strongest phishing indicator.

---

## 📈 Confusion Matrix

Confusion matrix generated and saved during model execution.

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- GridSearchCV
- Git & GitHub

---

## 🚀 Future Improvements

- Add SVM & XGBoost
- Deploy using Streamlit
- Add real-time URL parsing
- Convert into REST API
- Integrate browser extension for phishing detection

---

## 👨‍💻 Author

**Jaswanth**  
BTech – Artificial Intelligence & Data Science  
Cybersecurity & ML Enthusiast  

---

⭐ If you found this project useful, consider giving it a star!