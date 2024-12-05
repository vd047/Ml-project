import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Example data: Replace with your dataset
data = {
    'text': [
        'Breaking news: Aliens have landed on Earth!',
        'Government announces new tax reform.',
        'Study shows the benefits of meditation.',
        'Celebrity scandal: Famous actor involved in controversy.',
        'The cure for cancer has been found!',
        'Economy is expected to grow next year.',
        'New technology revolutionizes the industry.',
        'Asteroid to pass close to Earth tomorrow.',
        'Health experts warn about the new flu strain.',
        'Man wins lottery for the third time!'
    ],
    'label': [
        'fake', 'real', 'real', 'fake', 'fake',
        'real', 'real', 'fake', 'real', 'fake'
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
X = df['text']
y = df['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create and train the model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
