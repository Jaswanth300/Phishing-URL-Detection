\# 🔐 Phishing URL Detection using Machine Learning



\## 📌 Project Overview

This project builds a machine learning-based phishing website detection system using a real-world Kaggle dataset. The goal is to classify websites as \*\*Legitimate (0)\*\* or \*\*Phishing (1)\*\* using multiple ML models and performance benchmarking.



---



\## 📊 Dataset

\- Source: Kaggle Phishing Website Dataset

\- Features: 9 security-related attributes

\- Target Column: `Result`

\- Classes:

&nbsp; - 0 → Legitimate

&nbsp; - 1 → Phishing



---



\## 🧠 Models Implemented



| Model | Accuracy | Precision | Recall | F1-Score |

|--------|----------|-----------|--------|----------|

| Logistic Regression | 88.2% | 0.874 | 0.892 | 0.883 |

| Tuned Random Forest | \*\*90.1%\*\* | 0.905 | 0.897 | 0.901 |



---



\## 🔍 Feature Importance (Random Forest)



Top Contributing Features:



\- SFH (40%)

\- popUpWidnow (15%)

\- SSLfinal\_State (11%)

\- Request\_URL

\- URL\_of\_Anchor



Least Contributing:

\- having\_IP\_Address (~1%)



---



\## ⚙️ Hyperparameter Tuning



GridSearchCV was used with:

\- n\_estimators

\- max\_depth

\- min\_samples\_split

\- min\_samples\_leaf



Best Parameters:{'max\_depth': None, 'min\_samples\_leaf': 1, 'min\_samples\_split': 5, 'n\_estimators': 100}


---



\## 📈 Performance Insights



\- Random Forest outperformed Logistic Regression.

\- Balanced precision and recall indicate no class bias.

\- Non-linear ensemble methods perform better on phishing detection features.



---



\## 🚀 Future Improvements



\- Add Support Vector Machine (SVM)

\- Implement XGBoost

\- Deploy using Streamlit

\- Integrate real-time URL feature extraction

\- Add API-based phishing detection



---



\## 🛠️ Tech Stack



\- Python

\- Pandas

\- Scikit-learn

\- Seaborn

\- Matplotlib

\- Git \& GitHub



---



\## 👨‍💻 Author

Jaswanth  

BTech AI \& Data Science  

Cybersecurity \& ML Enthusiast

