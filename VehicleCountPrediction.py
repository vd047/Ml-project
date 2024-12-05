import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Example data: Replace with your dataset
data = {
    'sensor1': [30, 45, 50, 60, 80, 70, 55, 95, 100, 110],
    'sensor2': [25, 35, 55, 65, 85, 75, 60, 90, 105, 115],
    'sensor3': [20, 40, 60, 70, 90, 80, 65, 85, 110, 120],
    'vehicle_count': [50, 60, 80, 90, 120, 110, 85, 140, 150, 170]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['sensor1', 'sensor2', 'sensor3']]
y = df['vehicle_count']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Output
print("\nPredicted vs Actual vehicle counts:")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual: {actual}, Predicted: {predicted:.2f}")
