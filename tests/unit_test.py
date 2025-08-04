import joblib
import pandas as pd  # <- important
import numpy as np

def test_prediction_output():
    model = joblib.load("artifacts/model.joblib")

    # Provide sample as a DataFrame with correct column names
    sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=[
        "sepal_length", "sepal_width", "petal_length", "petal_width"
    ])
    prediction = model.predict(sample)[0]

    # Write predicted species to shared output file
    with open("test_output.txt", "a") as f:
        f.write("\n🧠 Model Prediction\n")
        f.write(f"✅ Predicted species: {prediction}\n")

    # Ensure it's a valid string
    assert isinstance(prediction, str)
