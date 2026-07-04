import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../diabetes.csv")

# Features (Questions)
X = df.drop("Outcome", axis=1)

# Target (Answer)
y = df["Outcome"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training patients:", len(X_train))
print("Testing patients:", len(X_test))
# Create the Machine Learning model
model = RandomForestClassifier(random_state=42)

# Teach the model
model.fit(X_train, y_train)

print("🎉 Model training completed!")
# Test the model
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)
# Save the trained model
joblib.dump(model, "../diabetes_model.pkl")
print(df.columns)

print("✅ Model saved successfully!")