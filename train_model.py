"""Train water quality classifier and save model."""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from config import TRAIN_DIR, TEST_DIR, MODELS_DIR

FEATURES = [
    "ph", "turbidity", "dissolved_oxygen", "temperature",
    "conductivity", "total_dissolved_solids", "nitrate",
    "fecal_coliform", "bod"
]
TARGET = "quality"

def load_data():
    train_path = os.path.join(TRAIN_DIR, "train.csv")
    test_path = os.path.join(TEST_DIR, "test.csv")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Run data_generator.py first to create train/test data.")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_df, test_df = load_data()
    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    print("Test Accuracy:", round(acc, 4))
    print("\nClassification Report:\n", classification_report(y_test, le.inverse_transform(y_pred)))
    print("Confusion Matrix:\n", confusion_matrix(y_test_enc, y_pred))
    joblib.dump(model, os.path.join(MODELS_DIR, "water_quality_model.joblib"))
    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder.joblib"))
    print("\nModel saved to", MODELS_DIR)

if __name__ == "__main__":
    train()
