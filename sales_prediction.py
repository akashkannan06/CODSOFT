import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

data = pd.read_csv("Advertising.csv")

print("DATASET PREVIEW")
print(data.head())

X = data[["TV", "Radio", "Newspaper"]]
y = data["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

sample = pd.DataFrame({
    "TV": [150],
    "Radio": [25],
    "Newspaper": [30]
})

predicted_sales = model.predict(sample)

print("\nPREDICTED SALES")
print(round(predicted_sales[0], 2))