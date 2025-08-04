import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def test_model_accuracy():
    df = pd.read_csv("data/iris.csv")
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = joblib.load("artifacts/model.joblib")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    # Saving accuracy output to file for CML
    with open("test_output.txt", "a") as f:
        f.write(f"\n📊 Model Evaluation\n✅ Accuracy: {acc:.2f}\n")

    assert acc >= 0.85, f"Model accuracy too low: {acc}"
