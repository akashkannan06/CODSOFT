import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("creditcard.csv")

print("DATASET PREVIEW")
print(data.head())

X = data[["Amount", "Time"]]
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL ACCURACY")
print(round(accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions))

sample_transaction = pd.DataFrame({
    "Amount": [10000],
    "Time": [21]
})

prediction = model.predict(sample_transaction)

print("\nTRANSACTION STATUS")

if prediction[0] == 1:
    print("Fraudulent Transaction")
else:
    print("Legitimate Transaction")