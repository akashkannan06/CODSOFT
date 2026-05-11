import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

dataset = pd.read_csv("Titanic-Dataset.csv")

print("DATASET PREVIEW")
print(dataset.head())

print("\nMISSING VALUES")
print(dataset.isnull().sum())

dataset["Age"] = dataset["Age"].fillna(dataset["Age"].median())

dataset["Embarked"] = dataset["Embarked"].fillna(
    dataset["Embarked"].mode()[0]
)

dataset.drop("Cabin", axis=1, inplace=True)

gender_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

dataset["Sex"] = gender_encoder.fit_transform(dataset["Sex"])

dataset["Embarked"] = embarked_encoder.fit_transform(
    dataset["Embarked"]
)

selected_features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = dataset[selected_features]

y = dataset["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

classifier = RandomForestClassifier(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

classifier.fit(X_train, y_train)

predicted_values = classifier.predict(X_test)

model_accuracy = accuracy_score(y_test, predicted_values)

print("\nMODEL ACCURACY")
print(round(model_accuracy * 100, 2), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predicted_values))

new_passenger = pd.DataFrame({
    "Pclass": [1],
    "Sex": [0],
    "Age": [28],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [80],
    "Embarked": [2]
})

prediction = classifier.predict(new_passenger)

print("\nSURVIVAL PREDICTION")

if prediction[0] == 1:
    print("Passenger likely survived")
else:
    print("Passenger likely did not survive")