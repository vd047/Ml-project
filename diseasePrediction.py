import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example data: Replace with your dataset
data = {
    'age': [25, 30, 45, 35, 50, 23, 37, 49, 55, 65],
    'bmi': [18.5, 24.9, 22.2, 26.5, 30.1, 21.0, 28.4, 25.0, 31.2, 29.5],
    'blood_pressure': [120, 130, 125, 140, 135, 115, 128, 138, 145, 150],
    'has_disease': [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['age', 'bmi', 'blood_pressure']]
y = df['has_disease']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
