import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

movie_data = pd.read_csv("movies.csv", encoding="latin1")

print("DATASET PREVIEW")
print(movie_data.head())

print("\nMISSING VALUES")
print(movie_data.isnull().sum())

movie_data = movie_data.dropna()

movie_data["Genre"] = LabelEncoder().fit_transform(
    movie_data["Genre"].astype(str)
)

movie_data["Director"] = LabelEncoder().fit_transform(
    movie_data["Director"].astype(str)
)

movie_data["Actor 1"] = LabelEncoder().fit_transform(
    movie_data["Actor 1"].astype(str)
)

movie_data["Duration"] = (
    movie_data["Duration"]
    .astype(str)
    .str.replace(" min", "", regex=False)
)

movie_data["Duration"] = pd.to_numeric(
    movie_data["Duration"],
    errors="coerce"
)

movie_data["Votes"] = (
    movie_data["Votes"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

movie_data["Votes"] = pd.to_numeric(
    movie_data["Votes"],
    errors="coerce"
)

movie_data["Rating"] = pd.to_numeric(
    movie_data["Rating"],
    errors="coerce"
)

movie_data = movie_data.dropna()

X = movie_data[
    [
        "Genre",
        "Director",
        "Actor 1",
        "Duration",
        "Votes"
    ]
]

y = movie_data["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nMODEL PERFORMANCE")
print("Mean Absolute Error:", round(mae, 2))

sample_movie = pd.DataFrame({
    "Genre": [5],
    "Director": [10],
    "Actor 1": [15],
    "Duration": [120],
    "Votes": [5000]
})

predicted_rating = model.predict(sample_movie)

print("\nPREDICTED MOVIE RATING")
print(round(predicted_rating[0], 2))