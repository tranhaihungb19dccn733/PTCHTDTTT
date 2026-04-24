"""FastAPI backend cho hệ thống dự đoán bệnh tim."""

from __future__ import annotations

import csv
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score

try:
	from .kbs_engine import HeartDiseaseKBS
	from .ml_model.train_model import FEATURE_COLUMNS, load_dataset
except ImportError:
	from kbs_engine import HeartDiseaseKBS
	from ml_model.train_model import FEATURE_COLUMNS, load_dataset


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "ml_model" / "random_forest.pkl"
USER_TRAINING_DATA_PATH = APP_DIR / "ml_model" / "data" / "heart_disease_extended_inputs.csv"

USER_TRAINING_FIELDS = [
	"submitted_at",
	"age",
	"sex",
	"weight",
	"height",
	"smoking",
	"family_history",
	"congenital_heart_disease",
	"diagnosed_diabetes",
	"physical_activity_level",
	"symptom_severity",
	"chest_pain_type",
	"resting_bp",
	"cholesterol",
	"fasting_blood_sugar",
	"resting_ecg",
	"max_heart_rate",
	"shortness_of_breath",
	"leg_swelling",
	"palpitations",
	"exercise_angina",
	"oldpeak",
	"slope",
	"num_major_vessels",
	"thal",
	"expert_risk_level",
	"expert_score",
	"ml_probability",
	"final_risk_level",
	"final_risk_percent",
	"confirmed_target",
]

OPTIONS = {
	"sex": [
		{"value": 1, "label": "Nam"},
		{"value": 0, "label": "Nữ"},
	],
	"smoking": [
		{"value": 0, "label": "Không hút thuốc"},
		{"value": 1, "label": "Đang hút thuốc"},
	],
	"family_history": [
		{"value": 0, "label": "Không có"},
		{"value": 1, "label": "Có"},
	],
	"congenital_heart_disease": [
		{"value": 0, "label": "Không"},
		{"value": 1, "label": "Có"},
	],
	"diagnosed_diabetes": [
		{"value": 0, "label": "Không"},
		{"value": 1, "label": "Có"},
	],
	"physical_activity_level": [
		{"value": 0, "label": "Thấp"},
		{"value": 1, "label": "Trung bình"},
		{"value": 2, "label": "Cao"},
	],
	"symptom_severity": [
		{"value": 0, "label": "Bình thường, hầu như không có triệu chứng"},
		{"value": 1, "label": "Có dấu hiệu nhẹ hoặc thoáng qua"},
		{"value": 2, "label": "Có nhiều triệu chứng nghi ngờ bệnh tim"},
		{"value": 3, "label": "Triệu chứng rõ rệt, nghi ngờ đã mắc bệnh"},
	],
	"chest_pain_type": [
		{"value": 0, "label": "Đau thắt ngực điển hình"},
		{"value": 1, "label": "Đau thắt ngực không điển hình"},
		{"value": 2, "label": "Đau không do thắt ngực"},
		{"value": 3, "label": "Không có triệu chứng"},
	],
	"fasting_blood_sugar": [
		{"value": 0, "label": "< 120 mg/dl"},
		{"value": 1, "label": "> 120 mg/dl"},
	],
	"resting_ecg": [
		{"value": 0, "label": "Bình thường"},
		{"value": 1, "label": "ST-T bất thường"},
		{"value": 2, "label": "Phì đại thất trái"},
	],
	"exercise_angina": [
		{"value": 0, "label": "Không"},
		{"value": 1, "label": "Có"},
	],
	"shortness_of_breath": [
		{"value": 0, "label": "Không"},
		{"value": 1, "label": "Có"},
	],
	"leg_swelling": [
		{"value": 0, "label": "Không"},
		{"value": 1, "label": "Có"},
	],
	"palpitations": [
		{"value": 0, "label": "Không"},
		{"value": 1, "label": "Có"},
	],
	"slope": [
		{"value": 0, "label": "Dốc lên"},
		{"value": 1, "label": "Ngang"},
		{"value": 2, "label": "Dốc xuống"},
	],
	"thal": [
		{"value": 0, "label": "Bình thường"},
		{"value": 1, "label": "Khiếm khuyết cố định"},
		{"value": 2, "label": "Khiếm khuyết hồi phục"},
	],
}


class HeartMetricsInput(BaseModel):
	age: int = Field(..., ge=0, le=120)
	sex: int = Field(..., ge=0, le=1)
	weight: float = Field(..., ge=0, le=500)
	height: float = Field(..., ge=0, le=250)
	smoking: int = Field(..., ge=0, le=1)
	family_history: int = Field(..., ge=0, le=1)
	congenital_heart_disease: int = Field(..., ge=0, le=1)
	diagnosed_diabetes: int = Field(..., ge=0, le=1)
	physical_activity_level: int = Field(..., ge=0, le=2)
	symptom_severity: int = Field(..., ge=0, le=3)
	chest_pain_type: int = Field(..., ge=0, le=3)
	resting_bp: int = Field(..., ge=0, le=300)
	cholesterol: int = Field(..., ge=0, le=1000)
	fasting_blood_sugar: int = Field(..., ge=0, le=1)
	resting_ecg: int = Field(..., ge=0, le=2)
	max_heart_rate: int = Field(..., ge=0, le=300)
	shortness_of_breath: int = Field(..., ge=0, le=1)
	leg_swelling: int = Field(..., ge=0, le=1)
	palpitations: int = Field(..., ge=0, le=1)
	exercise_angina: int = Field(..., ge=0, le=1)
	oldpeak: float = Field(..., ge=-10, le=10)
	slope: int = Field(..., ge=0, le=2)
	num_major_vessels: int = Field(..., ge=0, le=4)
	thal: int = Field(..., ge=0, le=2)

	model_config = {
		"json_schema_extra": {
			"example": {
				"age": 58,
				"sex": 1,
				"weight": 78,
				"height": 170,
				"smoking": 1,
				"family_history": 1,
				"congenital_heart_disease": 0,
				"diagnosed_diabetes": 0,
				"physical_activity_level": 0,
				"symptom_severity": 2,
				"chest_pain_type": 0,
				"resting_bp": 148,
				"cholesterol": 262,
				"fasting_blood_sugar": 0,
				"resting_ecg": 1,
				"max_heart_rate": 132,
				"shortness_of_breath": 1,
				"leg_swelling": 0,
				"palpitations": 1,
				"exercise_angina": 1,
				"oldpeak": 2.2,
				"slope": 1,
				"num_major_vessels": 1,
				"thal": 2,
			}
		}
	}


class PredictionResponse(BaseModel):
	input_data: dict[str, Any]
	expert_system: dict[str, Any]
	machine_learning: dict[str, Any]
	final_assessment: dict[str, Any]


DEMO_CASES = [
	{
		"id": "case-low-01",
		"title": "Ca nguy cơ thấp",
		"description": "Bệnh nhân trẻ, ít yếu tố nguy cơ, ECG lúc nghỉ bình thường.",
		"payload": {
			"age": 41,
			"sex": 0,
			"weight": 56,
			"height": 164,
			"smoking": 0,
			"family_history": 0,
			"congenital_heart_disease": 0,
			"diagnosed_diabetes": 0,
			"physical_activity_level": 2,
			"symptom_severity": 0,
			"chest_pain_type": 2,
			"resting_bp": 118,
			"cholesterol": 192,
			"fasting_blood_sugar": 0,
			"resting_ecg": 0,
			"max_heart_rate": 168,
			"shortness_of_breath": 0,
			"leg_swelling": 0,
			"palpitations": 0,
			"exercise_angina": 0,
			"oldpeak": 0.2,
			"slope": 0,
			"num_major_vessels": 0,
			"thal": 0,
		},
	},
	{
		"id": "case-moderate-01",
		"title": "Ca nguy cơ trung bình",
		"description": "Tăng huyết áp, mỡ máu cao và có bất thường nhẹ trên ECG.",
		"payload": {
			"age": 54,
			"sex": 1,
			"weight": 82,
			"height": 170,
			"smoking": 1,
			"family_history": 1,
			"congenital_heart_disease": 0,
			"diagnosed_diabetes": 0,
			"physical_activity_level": 1,
			"symptom_severity": 2,
			"chest_pain_type": 1,
			"resting_bp": 142,
			"cholesterol": 246,
			"fasting_blood_sugar": 0,
			"resting_ecg": 1,
			"max_heart_rate": 145,
			"shortness_of_breath": 1,
			"leg_swelling": 0,
			"palpitations": 1,
			"exercise_angina": 0,
			"oldpeak": 1.3,
			"slope": 1,
			"num_major_vessels": 0,
			"thal": 1,
		},
	},
	{
		"id": "case-high-01",
		"title": "Ca nguy cơ cao",
		"description": "Đau ngực điển hình khi gắng sức, oldpeak cao và có mạch lớn bị ảnh hưởng.",
		"payload": {
			"age": 63,
			"sex": 1,
			"weight": 91,
			"height": 168,
			"smoking": 1,
			"family_history": 1,
			"congenital_heart_disease": 1,
			"diagnosed_diabetes": 1,
			"physical_activity_level": 0,
			"symptom_severity": 3,
			"chest_pain_type": 0,
			"resting_bp": 156,
			"cholesterol": 288,
			"fasting_blood_sugar": 1,
			"resting_ecg": 2,
			"max_heart_rate": 124,
			"shortness_of_breath": 1,
			"leg_swelling": 1,
			"palpitations": 1,
			"exercise_angina": 1,
			"oldpeak": 2.8,
			"slope": 2,
			"num_major_vessels": 2,
			"thal": 2,
		},
	},
]


app = FastAPI(
	title="API Dự đoán bệnh tim",
	version="1.0.0",
	description="API kết hợp hệ chuyên gia và học máy để đánh giá nguy cơ bệnh tim mạch.",
)

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

kbs_engine = HeartDiseaseKBS()
ml_model = None
ml_model_error = None
ml_reference_metrics = {
	"available": False,
	"accuracy_percent": None,
	"sample_count": 0,
	"dataset_name": "Cleveland",
	"message": "Chưa có số liệu đánh giá độ chính xác.",
}


def build_ml_feature_row(data: dict[str, Any]) -> dict[str, Any]:
	"""Trích xuất các trường đặc trưng cần thiết cho mô hình ML từ dict dữ liệu đầu vào."""
	return {
		"age": data["age"],
		"sex": data["sex"],
		"chest_pain_type": data["chest_pain_type"],
		"resting_bp": data["resting_bp"],
		"cholesterol": data["cholesterol"],
		"fasting_blood_sugar": data["fasting_blood_sugar"],
		"resting_ecg": data["resting_ecg"],
		"max_heart_rate": data["max_heart_rate"],
		"congenital_heart_disease": data["congenital_heart_disease"],
		"exercise_angina": data["exercise_angina"],
		"oldpeak": data["oldpeak"],
		"slope": data["slope"],
		"num_major_vessels": data["num_major_vessels"],
		"thal": data["thal"],
	}


def get_model_feature_columns() -> list[str]:
	"""Trả về danh sách tên cột đặc trưng phù hợp với phiên bản mô hình ML đang được tải."""
	legacy_feature_columns = [
		"age",
		"sex",
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

	if ml_model is None:
		return FEATURE_COLUMNS

	if hasattr(ml_model, "feature_names_in_"):
		return [str(column) for column in ml_model.feature_names_in_]

	if hasattr(ml_model, "n_features_in_"):
		feature_count = int(ml_model.n_features_in_)
		if feature_count == len(FEATURE_COLUMNS):
			return FEATURE_COLUMNS
		if feature_count == len(legacy_feature_columns):
			return legacy_feature_columns

	return FEATURE_COLUMNS


def assess_training_coverage(data: dict[str, Any]) -> dict[str, Any]:
	"""Kiểm tra xem ca đầu vào có mẫu trùng khớp trong tập dữ liệu huấn luyện Cleveland không."""
	try:
		dataframe = load_dataset()
		features = pd.DataFrame([build_ml_feature_row(data)], columns=FEATURE_COLUMNS)
		matches = int(dataframe[FEATURE_COLUMNS].eq(features.iloc[0]).all(axis=1).sum())
		return {
			"available": True,
			"exact_match": matches > 0,
			"exact_match_count": matches,
			"dataset_name": "Cleveland",
		}
	except Exception as exc:
		return {
			"available": False,
			"exact_match": False,
			"exact_match_count": 0,
			"dataset_name": "Cleveland",
			"message": f"Không thể kiểm tra độ phủ dữ liệu huấn luyện: {exc}",
		}


def load_ml_model() -> None:
	"""Tải mô hình Random Forest từ file .pkl vào biến toàn cục ml_model; cập nhật ml_model_error nếu thất bại."""
	global ml_model, ml_model_error

	if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
		ml_model_error = "Tệp mô hình chưa được huấn luyện. Hệ thống sẽ dùng hệ chuyên gia là chính."
		ml_model = None
		update_reference_metrics()
		return

	try:
		with MODEL_PATH.open("rb") as model_file:
			ml_model = pickle.load(model_file)
			ml_model_error = None
	except Exception as exc:
		ml_model = None
		ml_model_error = f"Không thể tải mô hình ML: {exc}"

	update_reference_metrics()


def load_extended_labeled_samples(available_columns: list[str]) -> tuple[Any, Any, int]:
	"""Load extended user-submitted samples for reference accuracy.

	Trả về (features, labels, total_saved).
	- Ưu tiên dùng confirmed_target nếu có
	- Nếu không có, dùng final_risk_percent >= 70% để gán nhãn: 1 = mắc bệnh, 0 = không mắc
	- features và labels là None nếu không có hàng nào phù hợp.
	- total_saved là tổng số hàng đã lưu.
	"""
	if not USER_TRAINING_DATA_PATH.exists() or USER_TRAINING_DATA_PATH.stat().st_size == 0:
		return None, None, 0

	try:
		df = pd.read_csv(USER_TRAINING_DATA_PATH)
		total_saved = int(len(df))

		# Ưu tiên confirmed_target nếu có, nếu không dùng final_risk_percent >= 70
		df["effective_target"] = df["confirmed_target"].copy()
		needs_fallback = df["effective_target"].isna() | (df["effective_target"].astype(str).str.strip() == "")
		if needs_fallback.any() and "final_risk_percent" in df.columns:
			risk_percent = pd.to_numeric(df.loc[needs_fallback, "final_risk_percent"], errors="coerce")
			df.loc[needs_fallback, "effective_target"] = (risk_percent >= 70).astype(int)

		usable = df[df["effective_target"].notna()]
		if usable.empty:
			return None, None, total_saved

		ext_available = [col for col in available_columns if col in usable.columns]
		if not ext_available:
			return None, None, total_saved

		features = usable[ext_available].apply(pd.to_numeric, errors="coerce").dropna()
		if features.empty:
			return None, None, total_saved

		target = usable.loc[features.index, "effective_target"].astype(int)
		return features[ext_available], target, total_saved
	except Exception:
		return None, None, 0


def update_reference_metrics() -> None:
	"""Tính và cập nhật độ chính xác tham chiếu của mô hình ML dựa trên dữ liệu Cleveland và mẫu người dùng đã xác nhận."""
	global ml_reference_metrics

	if ml_model is None:
		ml_reference_metrics = {
			"available": False,
			"accuracy_percent": None,
			"sample_count": 0,
			"extended_sample_total": 0,
			"dataset_name": "Cleveland",
			"message": ml_model_error or "Chưa có số liệu đánh giá độ chính xác.",
		}
		return

	try:
		dataframe = load_dataset()
		model_feature_columns = get_model_feature_columns()
		available_columns = [column for column in model_feature_columns if column in dataframe.columns]
		if not available_columns:
			raise ValueError("Không có trường đặc trưng phù hợp để tính độ tin cậy.")

		all_features = [dataframe[available_columns]]
		all_targets = [dataframe["target"]]
		cleveland_count = int(len(dataframe))

		ext_features, ext_target, ext_total = load_extended_labeled_samples(available_columns)
		if ext_features is not None:
			all_features.append(ext_features)
			all_targets.append(ext_target)

		combined_features = pd.concat(all_features, ignore_index=True)
		combined_target = pd.concat(all_targets, ignore_index=True)

		predictions = ml_model.predict(combined_features)
		accuracy = round(float(accuracy_score(combined_target, predictions)) * 100, 1)

		labeled_extended = int(len(combined_features)) - cleveland_count
		dataset_label = "Cleveland + mẫu người dùng" if labeled_extended > 0 else "Cleveland"

		parts = [f"{cleveland_count} mẫu Cleveland"]
		if labeled_extended > 0:
			parts.append(f"{labeled_extended} mẫu người dùng đã xác nhận")
		if ext_total > labeled_extended:
			unlabeled = ext_total - labeled_extended
			parts.append(f"{unlabeled} mẫu người dùng chưa có nhãn xác nhận (không tính vào độ chính xác)")

		ml_reference_metrics = {
			"available": True,
			"accuracy_percent": accuracy,
			"sample_count": int(len(combined_features)),
			"extended_sample_total": ext_total,
			"dataset_name": dataset_label,
			"message": f"Độ tin cậy tính trên: {'; '.join(parts)}.",
		}
	except Exception as exc:
		ml_reference_metrics = {
			"available": False,
			"accuracy_percent": None,
			"sample_count": 0,
			"extended_sample_total": 0,
			"dataset_name": "Cleveland",
			"message": f"Không thể tính độ tin cậy: {exc}",
		}


def compute_ml_prediction(data: dict[str, Any]) -> dict[str, Any]:
	"""Thực hiện dự đoán bằng mô hình ML và trả về xác suất, thông điệp trạng thái và thông tin phủ dữ liệu huấn luyện."""
	training_coverage = assess_training_coverage(data)

	if data["congenital_heart_disease"] == 1:
		return {
			"available": False,
			"prediction": 1,
			"probability": 100.0,
			"message": "Đã bỏ qua mô hình ML vì bệnh tim bẩm sinh được quy ước là bệnh tim xác định 100%.",
			"training_coverage": training_coverage,
			"reference_metrics": ml_reference_metrics,
		}

	if ml_model is None:
		if training_coverage["available"] and not training_coverage["exact_match"]:
			message = (
				"Ca này chưa có mẫu trùng khớp trong dữ liệu huấn luyện hiện tại và mô hình ML chưa sẵn sàng để suy luận bổ sung. "
				"Hệ thống vẫn lưu ca này để dùng cho lần huấn luyện tiếp theo."
			)
		else:
			message = ml_model_error or "Mô hình ML hiện chưa sẵn sàng."

		return {
			"available": False,
			"prediction": None,
			"probability": None,
			"message": message,
			"training_coverage": training_coverage,
			"reference_metrics": ml_reference_metrics,
		}

	model_feature_columns = get_model_feature_columns()
	base_feature_row = build_ml_feature_row(data)
	aligned_feature_row = {
		column: base_feature_row.get(column, 0)
		for column in model_feature_columns
	}
	features = pd.DataFrame([aligned_feature_row], columns=model_feature_columns)

	prediction = int(ml_model.predict(features)[0])
	probability = None
	if hasattr(ml_model, "predict_proba"):
		probability = round(float(ml_model.predict_proba(features)[0][1]) * 100, 1)

	if training_coverage["available"] and training_coverage["exact_match"]:
		message = (
			"Đã sử dụng mô hình Random Forest trên bộ 14 chỉ số Cleveland mở rộng để dự đoán bổ sung. "
			f"Ca này đã có {training_coverage['exact_match_count']} mẫu trùng khớp trong dữ liệu huấn luyện."
		)
	else:
		message = (
			"Ca này chưa có mẫu trùng khớp trong dữ liệu huấn luyện hiện tại. "
			"Hệ thống vẫn đưa ra dự đoán theo các trường đã được huấn luyện và sẽ tự lưu ca này để bổ sung huấn luyện sau."
		)

	return {
		"available": True,
		"prediction": prediction,
		"probability": probability,
		"message": message,
		"training_coverage": training_coverage,
		"reference_metrics": ml_reference_metrics,
	}


def risk_level_from_score(score: float) -> str:
	"""Chuyển điểm nguy cơ (%) thành nhãn mức nguy cơ: 'high', 'moderate' hoặc 'low'."""
	if score >= 70:
		return "high"
	if score >= 40:
		return "moderate"
	return "low"


def compute_expert_case_confidence(expert_result: dict[str, Any]) -> float:
	"""Ước tính độ tin cậy ca (%) dựa trên điểm chuẩn hóa và số luật kích hoạt từ hệ chuyên gia."""
	score = float(expert_result["normalized_score"])
	triggered_count = len(expert_result.get("rule_hits", []))

	if score < 40:
		confidence = 50 + ((40 - score) / 40) * 30
	elif score < 70:
		confidence = 52 + (min(score - 40, 70 - score) / 15) * 18
	else:
		confidence = 55 + ((score - 70) / 30) * 30

	if triggered_count >= 3:
		confidence += 5
	elif triggered_count == 0:
		confidence -= 5

	return round(max(40.0, min(95.0, confidence)), 1)


def compute_ml_case_confidence(probability: float | None) -> float | None:
	"""Ước tính độ tin cậy ca (%) từ xác suất ML; trả về None nếu mô hình không có xác suất."""
	if probability is None:
		return None
	return round(min(99.0, 50 + abs(float(probability) - 50)), 1)


def build_case_confidence(
	input_data: dict[str, Any],
	expert_result: dict[str, Any],
	ml_result: dict[str, Any],
	final_assessment: dict[str, Any],
) -> dict[str, Any]:
	"""Tổng hợp độ tin cậy toàn ca từ hệ chuyên gia, xác suất ML, độ chính xác tham chiếu và mức độ phủ dữ liệu."""
	if input_data["congenital_heart_disease"] == 1:
		return {
			"percent": 99.0,
			"label": "Rất cao",
			"message": "Độ tin cậy rất cao vì ca này được quy ước là bệnh tim xác định do có bệnh tim bẩm sinh.",
			"components": {
				"expert_percent": 100.0,
				"ml_percent": 100.0,
				"reference_accuracy_percent": ml_result.get("reference_metrics", {}).get("accuracy_percent"),
				"agreement": True,
				"exact_training_match": bool(ml_result.get("training_coverage", {}).get("exact_match")),
			},
		}

	expert_confidence = compute_expert_case_confidence(expert_result)
	ml_probability = ml_result.get("probability")
	ml_confidence = compute_ml_case_confidence(ml_probability)
	reference_accuracy = ml_result.get("reference_metrics", {}).get("accuracy_percent")
	training_coverage = ml_result.get("training_coverage", {})
	exact_match = bool(training_coverage.get("exact_match"))
	model_risk_level = risk_level_from_score(ml_probability) if ml_probability is not None else expert_result["risk_level"]
	agreement = expert_result["risk_level"] == model_risk_level

	weighted_parts: list[tuple[float, float]] = [(expert_confidence, 0.55 if ml_confidence is None else 0.4)]
	if ml_confidence is not None:
		weighted_parts.append((ml_confidence, 0.45))
	if reference_accuracy is not None:
		weighted_parts.append((float(reference_accuracy), 0.15 if ml_confidence is not None else 0.25))

	total_weight = sum(weight for _, weight in weighted_parts)
	confidence = sum(value * weight for value, weight in weighted_parts) / total_weight

	if ml_confidence is not None:
		confidence += 6 if agreement else -8
	if training_coverage.get("available"):
		confidence += 4 if exact_match else -2

	max_confidence = 88.0 if final_assessment["risk_level"] == "moderate" else 97.0
	confidence = round(max(35.0, min(max_confidence, confidence)), 1)

	reasons = []
	if ml_confidence is None:
		reasons.append("đang dựa chủ yếu vào hệ chuyên gia vì mô hình ML chưa sẵn sàng")
	else:
		reasons.append(
			"hệ chuyên gia và ML cho kết quả cùng xu hướng"
			if agreement
			else "hệ chuyên gia và ML còn có khác biệt nên độ chắc chắn bị giảm"
		)
		if abs(float(ml_probability) - 50) >= 20:
			reasons.append("xác suất ML cách khá xa ngưỡng 50%")
		else:
			reasons.append("xác suất ML còn gần ngưỡng 50%")

	if training_coverage.get("available"):
		if exact_match:
			match_count = int(training_coverage.get("exact_match_count", 0))
			reasons.append(f"đã có {match_count} mẫu trùng khớp trong dữ liệu huấn luyện")
		else:
			reasons.append("chưa có mẫu trùng khớp hoàn toàn trong dữ liệu huấn luyện")

	if reference_accuracy is not None:
		reasons.append(f"độ chính xác tham chiếu hiện tại của mô hình là {reference_accuracy}%")

	if confidence >= 85:
		label = "Cao"
	elif confidence >= 65:
		label = "Khá tốt"
	else:
		label = "Thăm dò"

	return {
		"percent": confidence,
		"label": label,
		"message": "Độ tin cậy ca này được ước tính từ mức nhất quán giữa hệ chuyên gia, xác suất ML và độ phủ dữ liệu huấn luyện; " + "; ".join(reasons) + ".",
		"components": {
			"expert_percent": expert_confidence,
			"ml_percent": ml_confidence,
			"reference_accuracy_percent": reference_accuracy,
			"agreement": agreement,
			"exact_training_match": exact_match,
		},
	}


def build_final_assessment(input_data: dict[str, Any], expert_result: dict[str, Any], ml_result: dict[str, Any]) -> dict[str, Any]:
	"""Kết hợp điểm hệ chuyên gia và xác suất ML thành đánh giá nguy cơ cuối cùng (mức, nhãn, % và bước tiếp theo)."""
	if input_data["congenital_heart_disease"] == 1:
		return {
			"risk_level": "high",
			"risk_label": "Bệnh tim xác định",
			"risk_percent": 100.0,
			"next_step": "Cần khám chuyên khoa tim mạch ngay và theo dõi điều trị theo hồ sơ tim bẩm sinh hiện có.",
		}

	expert_score = expert_result["normalized_score"]
	ml_score = ml_result["probability"] if ml_result["probability"] is not None else expert_score
	blended_score = round((expert_score * 0.6) + (ml_score * 0.4), 1)

	risk_level = risk_level_from_score(blended_score)
	risk_label = {
		"low": "Nguy cơ thấp",
		"moderate": "Nguy cơ trung bình",
		"high": "Nguy cơ cao",
	}[risk_level]

	return {
		"risk_level": risk_level,
		"risk_label": risk_label,
		"risk_percent": blended_score,
		"next_step": next_step_for_risk(risk_level),
	}


def build_prediction_response(input_data: dict[str, Any]) -> dict[str, Any]:
	"""Điều phối toàn bộ quy trình: chạy hệ chuyên gia, ML, tổng hợp kết quả, lưu bản ghi và trả về response đầy đủ."""
	expert_result = kbs_engine.analyze(input_data)
	ml_result = compute_ml_prediction(input_data)
	final_assessment = build_final_assessment(input_data, expert_result, ml_result)
	final_assessment["confidence"] = build_case_confidence(input_data, expert_result, ml_result, final_assessment)
	save_training_record(input_data, expert_result, ml_result, final_assessment)
	return {
		"input_data": input_data,
		"expert_system": expert_result,
		"machine_learning": ml_result,
		"final_assessment": final_assessment,
	}


def save_training_record(
	input_data: dict[str, Any],
	expert_result: dict[str, Any],
	ml_result: dict[str, Any],
	final_assessment: dict[str, Any],
) -> None:
	"""Lưu bản ghi dự đoán (dữ liệu đầu vào + kết quả) vào file CSV để phục vụ cho lần huấn luyện mô hình sau."""
	USER_TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
	file_exists = USER_TRAINING_DATA_PATH.exists() and USER_TRAINING_DATA_PATH.stat().st_size > 0

	final_risk_percent = final_assessment["risk_percent"]
	confirmed_target = 1 if final_risk_percent >= 65 else 0

	record = {
		"submitted_at": datetime.now().isoformat(timespec="seconds"),
		**input_data,
		"expert_risk_level": expert_result["risk_level"],
		"expert_score": expert_result["normalized_score"],
		"ml_probability": ml_result["probability"],
		"final_risk_level": final_assessment["risk_level"],
		"final_risk_percent": final_risk_percent,
		"confirmed_target": confirmed_target,
	}

	with USER_TRAINING_DATA_PATH.open("a", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=USER_TRAINING_FIELDS)
		if not file_exists:
			writer.writeheader()
		writer.writerow(record)


def next_step_for_risk(risk_level: str) -> str:
	"""Trả về chuỗi hướng dẫn bước tiếp theo tương ứng với mức nguy cơ (low/moderate/high)."""
	mapping = {
		"low": "Duy trì theo dõi định kỳ, ăn uống lành mạnh và tập luyện đều.",
		"moderate": "Nên khám chuyên khoa tim mạch để được chỉ định xét nghiệm bổ sung.",
		"high": "Cần gặp bác sĩ tim mạch sớm để được đánh giá chuyên sâu và xử trí kịp thời.",
	}
	return mapping[risk_level]


@app.on_event("startup")
def startup_event() -> None:
	"""Sự kiện khởi động ứng dụng FastAPI: tải mô hình ML vào bộ nhớ."""
	load_ml_model()


@app.get("/")
def root() -> dict[str, Any]:
	"""Endpoint gốc: trả về thông tin tổng quan về API và các đường dẫn hữu ích."""
	return {
		"message": "API Dự đoán bệnh tim đang hoạt động.",
		"docs": "/docs",
		"sample_prediction": "/api/predict/sample",
		"demo_cases": "/api/demo-cases",
		"demo_case_count": len(DEMO_CASES),
	}


@app.get("/health")
def health() -> dict[str, Any]:
	"""Endpoint kiểm tra trạng thái hệ thống: trả về trạng thái mô hình ML và số liệu độ chính xác tham chiếu."""
	return {
		"status": "ok",
		"ml_model_loaded": ml_model is not None,
		"ml_message": ml_model_error or "Mô hình ML đã sẵn sàng.",
		"ml_reference_metrics": ml_reference_metrics,
	}


@app.get("/api/options")
def get_options() -> dict[str, Any]:
	"""Endpoint trả về danh sách các tùy chọn giá trị (label/value) cho từng trường nhập liệu của form."""
	return OPTIONS


@app.get("/api/demo-cases")
def get_demo_cases() -> dict[str, Any]:
	"""Endpoint trả về danh sách các ca demo kèm kết quả dự đoán tương ứng (thấp / trung bình / cao)."""
	cases = []
	for case in DEMO_CASES:
		prediction = build_prediction_response(case["payload"])
		cases.append(
			{
				"id": case["id"],
				"title": case["title"],
				"description": case["description"],
				"risk_level": prediction["final_assessment"]["risk_level"],
				"risk_label": prediction["final_assessment"]["risk_label"],
				"risk_percent": prediction["final_assessment"]["risk_percent"],
				"payload": case["payload"],
			}
		)

	return {
		"total": len(cases),
		"items": cases,
	}


@app.get("/api/predict/sample")
def sample_prediction() -> dict[str, Any]:
	"""Endpoint trả về kết quả dự đoán mẫu sử dụng ca nguy cơ cao từ DEMO_CASES để kiểm tra nhanh API."""
	sample = HeartMetricsInput(**DEMO_CASES[2]["payload"])
	return predict(sample)


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: HeartMetricsInput) -> dict[str, Any]:
	"""Endpoint chính: nhận dữ liệu chỉ số sức khỏe của bệnh nhân và trả về đánh giá nguy cơ tim mạch đầy đủ."""
	input_data = payload.model_dump()
	return build_prediction_response(input_data)

