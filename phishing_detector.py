import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# -------- Generate Balanced Dataset --------
def generate_dataset(n=500):
    data = []

    for _ in range(n):
        features = np.random.choice([-1, 0, 1], size=23)
        Result = np.random.choice([-1, 1])  # balanced labels
        row = list(features) + [Result]
        data.append(row)

    columns = [
        "having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol",
        "double_slash_redirecting","Prefix_Suffix","having_Sub_Domain",
        "SSLfinal_State","Domain_registeration_length","Favicon","port",
        "HTTPS_token","Request_URL","URL_of_Anchor","Links_in_tags","SFH",
        "Submitting_to_email","Abnormal_URL","Redirect","on_mouseover",
        "RightClick","popUpWidnow","Iframe","Result"
    ]

    df = pd.DataFrame(data, columns=columns)
    df.to_csv("data/phishing_data.csv", index=False)


# Generate dataset (run once)
generate_dataset(500)


# -------- Model Training --------
df = pd.read_csv("data/phishing_data.csv")

X = df.drop("Result", axis=1)
y = df["Result"].replace(-1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Phishing Detection")
plt.show()
