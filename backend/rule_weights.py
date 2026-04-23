"""Cấu hình trọng số cho các luật của hệ chuyên gia."""

from __future__ import annotations


RULE_WEIGHTS: dict[str, int] = {
	"obesity_bmi": 10,
	"smoking": 10,
	"family_history": 9,
	"congenital_heart_disease": 16,
	"diagnosed_diabetes": 11,
	"low_physical_activity": 8,
	"symptom_severity_moderate": 10,
	"symptom_severity_high": 16,
	"age_over_55": 8,
	"typical_angina": 18,
	"asymptomatic_pattern": 16,
	"high_resting_bp": 10,
	"high_cholesterol": 10,
	"fasting_blood_sugar": 7,
	"abnormal_ecg": 12,
	"low_max_heart_rate": 12,
	"shortness_of_breath": 12,
	"leg_swelling": 11,
	"palpitations": 8,
	"exercise_angina": 18,
	"st_depression": 14,
	"flat_or_downslope": 9,
	"major_vessels_visible": 15,
	"thal_defect": 16,
}


def get_rule_weight(rule_key: str) -> int:
	try:
		return RULE_WEIGHTS[rule_key]
	except KeyError as exc:
		raise KeyError(f"Chưa cấu hình trọng số cho luật: {rule_key}") from exc