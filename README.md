# 🔐 Phishing URL Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Project-Production%20Ready-brightgreen)

---

## 📌 Project Overview

This project implements a complete end-to-end phishing website detection system using Machine Learning.

It includes:

- Model benchmarking (Logistic Regression, Random Forest, SVM)
- Hyperparameter tuning (GridSearchCV)
- Feature importance analysis
- Model comparison table
- Model serialization
- REST API deployment using Flask

The system classifies websites as:

- 0 → Legitimate
- 1 → Phishing

---

## 📊 Dataset

- Source: Kaggle Phishing Website Dataset
- Total Features: 9 structured security features
- Target Column: `Result`
- Balanced dataset

---

## 🧠 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 88.2% | 0.874 | 0.892 | 0.883 |
| Random Forest (Tuned) | 90.1% | 0.905 | 0.897 | 0.901 |
| 🔥 SVM (Best) | **90.6%** | 0.910 | 0.901 | 0.906 |

SVM with RBF kernel performed best.

---

## 🔍 Feature Importance (Random Forest)

Top contributing features:

- SFH (~40%)
- popUpWidnow (~15%)
- SSLfinal_State (~11%)
- Request_URL
- URL_of_Anchor

Least contributing:

- having_IP_Address (~1%)

---

## 🗂️ Project Structure

```
Phishing-URL-Detection/
│
├── data/
│   └── Website Phishing.csv
│
├── phishing_detector.py
├── app.py
├── phishing_model.pkl
├── confusion_matrix.png
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🌐 Running the Project

### 1️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Train the model

```
python phishing_detector.py
```

This will:
- Train models
- Perform benchmarking
- Save `phishing_model.pkl`

### 3️⃣ Run Flask API

```
python app.py
```

Server will start at:

```
http://127.0.0.1:5000/
```

---

## 🚀 API Usage Example

### Endpoint:

```
POST /predict
```

### Example Request:

```
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d "{\"SFH\":1,
\"popUpWidnow\":-1,
\"SSLfinal_State\":1,
\"Request_URL\":-1,
\"URL_of_Anchor\":-1,
\"web_traffic\":1,
\"URL_Length\":1,
\"age_of_domain\":1,
\"having_IP_Address\":0}"
```

### Example Response:

```
{
  "prediction": 0,
  "result": "Legitimate Website"
}
```

---

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Seaborn
- Matplotlib
- GridSearchCV
- Flask
- Git & GitHub

---

## 🚀 Future Improvements

- Add XGBoost
- Deploy API to cloud (Render/Railway)
- Add frontend interface
- Convert into browser extension

---

## 👨‍💻 Author

Jaswanth  
BTech – Artificial Intelligence & Data Science  
Cybersecurity & ML Enthusiast