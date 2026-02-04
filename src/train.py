from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_models(X, y):
    """Train a Random Forest classifier for risk prediction."""
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X, y)

    os.makedirs("output/model", exist_ok=True)
    joblib.dump(model, "output/model/risk_model.pkl")

    return model
