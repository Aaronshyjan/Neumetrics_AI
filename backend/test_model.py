import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "cognitive_model.pkl")

if not os.path.exists(model_path):
    print("Model not found")
else:
    model = joblib.load(model_path)
    
    # Test data for "Low" (based on training script)
    # Dwell: 80, Flight: 120, Speed: 9
    test_low = np.array([[80, 120, 9]])
    prediction = model.predict(test_low)[0]
    print(f"Prediction for Low-like data: {prediction} (Type: {type(prediction)})")
    
    mapping = {0: "Low", 1: "Medium", 2: "High"}
    print(f"Mapped value: {mapping.get(prediction, 'Medium')}")
