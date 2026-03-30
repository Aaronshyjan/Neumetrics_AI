import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Set seed for reproducibility
np.random.seed(42)

X = []
y = []

# Realistic Keystroke Timing Ranges (approximate)
# Feature 0: Dwell Time (ms)
# Feature 1: Flight Time (ms) 
# Feature 2: Typing Speed (chars/sec)

# LOW cognitive load (Fluent, relaxed typing)
for _ in range(300):
    X.append([
        np.random.normal(80, 20),   # Dwell: 80ms
        np.random.normal(120, 40),  # Flight: 120ms
        np.random.normal(9, 2)      # Speed: 9 chars/s (~100 wpm)
    ])
    y.append(0)

# MEDIUM cognitive load (Standard typing, some thought)
for _ in range(300):
    X.append([
        np.random.normal(150, 40),  # Dwell: 150ms
        np.random.normal(400, 150), # Flight: 400ms
        np.random.normal(4, 1.5)    # Speed: 4 chars/s (~48 wpm)
    ])
    y.append(1)

# HIGH cognitive load (Stressed, hesitant, or difficult task)
for _ in range(300):
    X.append([
        np.random.normal(300, 100), # Dwell: 300ms
        np.random.normal(1000, 300),# Flight: 1000ms
        np.random.normal(1.5, 0.5)  # Speed: 1.5 chars/s (~18 wpm)
    ])
    y.append(2)

X = np.array(X)
y = np.array(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Training Accuracy: {accuracy * 100:.2f}%")

# Save model and accuracy
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
joblib.dump(model, os.path.join(BASE_DIR, "cognitive_model.pkl"))
joblib.dump(accuracy, os.path.join(BASE_DIR, "accuracy_score.pkl"))

print(f"✅ Model trained and saved successfully with {accuracy*100:.1f}% accuracy")

