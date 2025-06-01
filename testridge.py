import DatabaseManager
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

db = DatabaseManager.DatabaseManager(host="localhost", user="root", password="venomio", database="rrtest")
data_select_query = "SELECT * FROM drivers"
df = db.select(data_select_query, ())

# Features and target
X = df[["gender", "driving_experience", "age", "number_of_accidents"]]
y = df["safety_score"]

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Ridge regression with some alpha (regularization strength)
ridge = Ridge(alpha=1.0)

# Fit model
ridge.fit(X_train, y_train)

# Predict on test set
y_pred = ridge.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ridge Regression Results")
print("------------------------")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Example prediction for your test driver: gender=1, driving_experience=5, age=30, accidents=1
test_driver = np.array([[0, 55, 90, 3]])
predicted_score = ridge.predict(test_driver)[0]
print(f"\nPredicted safety score for test driver: {predicted_score:.2f}")
