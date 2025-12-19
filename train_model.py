import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load dataset (built-in, no internet needed)
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['quality'] = data.target

print("Dataset loaded!")
print(df.head())
print(f"Shape: {df.shape}")

# Split features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open('wine_model.pkl', 'wb'))
print("\nModel saved as wine_model.pkl")
