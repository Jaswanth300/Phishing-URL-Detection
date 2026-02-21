import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# Load Kaggle dataset
df = pd.read_csv("data/Website Phishing.csv")

# Convert target column
df['Result'] = df['Result'].replace({-1: 1, 1: 0})

# Features and target
X = df.drop("Result", axis=1)
y = df["Result"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------- Logistic Regression ----------------
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy:", accuracy_score(y_test, log_pred))
print(classification_report(y_test, log_pred))


# ---------------- Random Forest (Hyperparameter Tuning) ----------------

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# ---------------- Support Vector Machine ----------------

svm_model = SVC(kernel='rbf', C=1.0)
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)

print("\n===== Support Vector Machine =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# Best model
best_rf = grid_search.best_estimator_

print("\nBest Parameters Found:")
print(grid_search.best_params_)

# Predict using best model
rf_pred = best_rf.predict(X_test)

print("\n===== Tuned Random Forest =====")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# ================= Model Comparison Summary =================

models = {
    "Logistic Regression": log_pred,
    "Random Forest (Tuned)": rf_pred,
    "SVM": svm_pred
}

print("\n========== Model Comparison Summary ==========\n")

print("{:<22} {:<10} {:<10} {:<10} {:<10}".format(
    "Model", "Accuracy", "Precision", "Recall", "F1-Score"
))

print("-" * 65)

for name, preds in models.items():
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("{:<22} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
        name, acc, prec, rec, f1
    ))


# Confusion Matrix for Random Forest
cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Random Forest")
plt.savefig("confusion_matrix.png")
plt.close()