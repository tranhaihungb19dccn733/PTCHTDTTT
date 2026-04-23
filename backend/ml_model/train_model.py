"""Training pipeline for the heart disease machine learning model."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart_disease_cleveland.csv"
MODEL_PATH = BASE_DIR / "random_forest.pkl"
FEATURE_COLUMNS = [
	"age",
	"sex",
	"congenital_heart_disease",
	"chest_pain_type",
	"resting_bp",
	"cholesterol",
	"fasting_blood_sugar",
	"resting_ecg",
	"max_heart_rate",
	"exercise_angina",
	"oldpeak",
	"slope",
	"num_major_vessels",
	"thal",
]
DEFAULT_FEATURE_VALUES = {
	"congenital_heart_disease": 0,
}
TARGET_CANDIDATES = ["target", "num", "heart_disease"]
RENAMES = {
	"cp": "chest_pain_type",
	"trestbps": "resting_bp",
	"chol": "cholesterol",
	"fbs": "fasting_blood_sugar",
	"restecg": "resting_ecg",
	"thalach": "max_heart_rate",
	"exang": "exercise_angina",
	"ca": "num_major_vessels",
}

CATEGORY_MAPPINGS = {
	"chest_pain_type": {1: 0, 2: 1, 3: 2, 4: 3},
	"slope": {1: 0, 2: 1, 3: 2},
	"thal": {3: 0, 6: 1, 7: 2},
}


def load_dataset() -> pd.DataFrame:
	if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
		raise FileNotFoundError(
			f"Khong tim thay du lieu huan luyen hop le tai {DATA_PATH}."
		)

	dataframe = pd.read_csv(DATA_PATH)
	dataframe = dataframe.rename(columns=RENAMES)
	dataframe = normalize_dataset(dataframe)

	for column, default_value in DEFAULT_FEATURE_VALUES.items():
		if column not in dataframe.columns:
			dataframe[column] = default_value

	target_column = next((column for column in TARGET_CANDIDATES if column in dataframe.columns), None)
	if target_column is None:
		raise ValueError(
			"Khong tim thay cot target. Hay dung mot trong cac cot: target, num, heart_disease."
		)

	missing_columns = [column for column in FEATURE_COLUMNS if column not in dataframe.columns]
	if missing_columns:
		raise ValueError(f"Thieu cot dac trung: {missing_columns}")

	dataframe = dataframe[FEATURE_COLUMNS + [target_column]].dropna().copy()
	dataframe[target_column] = (dataframe[target_column] > 0).astype(int)
	dataframe = dataframe.rename(columns={target_column: "target"})
	return dataframe


def normalize_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
	normalized = dataframe.copy()

	for column in FEATURE_COLUMNS:
		if column in normalized.columns:
			normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

	for column, mapping in CATEGORY_MAPPINGS.items():
		if column in normalized.columns:
			normalized[column] = normalized[column].replace(mapping)

	return normalized


def train_model() -> None:
	dataframe = load_dataset()

	features = dataframe[FEATURE_COLUMNS]
	target = dataframe["target"]

	X_train, X_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=42,
		stratify=target,
	)

	model = RandomForestClassifier(
		n_estimators=300,
		max_depth=10,
		min_samples_split=4,
		random_state=42,
	)
	model.fit(X_train, y_train)

	predictions = model.predict(X_test)
	accuracy = accuracy_score(y_test, predictions)

	with MODEL_PATH.open("wb") as model_file:
		pickle.dump(model, model_file)

	print(f"Model saved to: {MODEL_PATH}")
	print(f"Accuracy: {accuracy:.4f}")
	print(classification_report(y_test, predictions, digits=4))


if __name__ == "__main__":
	train_model()

