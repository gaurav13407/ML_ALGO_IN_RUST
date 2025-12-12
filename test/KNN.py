import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)
import time
from sklearn.metrics import roc_auc_score


PATH = r"../examples/Logistic_regression_5.csv"
df = pd.read_csv(PATH)

target_column = "Outcome"

X = df.drop(columns=[target_column])
y = df[target_column]

X = pd.get_dummies(X, drop_first=False)   # drop_first=False = full one-hot

# Convert to numpy
X = X.values
y = y.values


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


knn = KNeighborsClassifier(n_neighbors=5)

# Train time
start = time.time()
knn.fit(X_train, y_train)
train_time = time.time() - start

# Predict time
start = time.time()
y_pred = knn.predict(X_test)
predict_time = time.time() - start

roc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])

# -------------------------
# 5. Metrics
# -------------------------
print("Train Time:", train_time, "seconds")
print("Predict Time:", predict_time, "seconds")


print("ROC-AUC Score:", roc)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nf1_score:\n", f1_score(y_test, y_pred))

print("\nRecall :\n", recall_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

