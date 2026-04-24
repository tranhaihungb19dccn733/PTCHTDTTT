"""Cấu hình trọng số cho các luật của hệ chuyên gia."""

from __future__ import annotations


RULE_WEIGHTS: dict[str, int] = {
	# Nhóm Ưu tiên
    "congenital_heart_disease": 100,

    # Nhóm NẶNG (Dấu hiệu đỏ)
    "typical_angina": 65,
    "exercise_angina": 65,
    "st_depression": 60,
    "major_vessels_visible": 60,
    "symptom_severity_high": 35,
    "shortness_of_breath": 35,
    "leg_swelling": 35,

    # Nhóm TRUNG BÌNH (Đã tăng cường - Vùng đệm)
    "high_resting_bp": 20,
    "diagnosed_diabetes": 20,
    "thal_defect": 20,
    "symptom_severity_moderate": 15,
    "abnormal_ecg": 15,
    "fasting_blood_sugar": 15,
    "flat_or_downslope": 15,
    "low_max_heart_rate": 10,
    "palpitations": 10,

    # Nhóm NHẸ (Yếu tố nền - Không gây nhiễu)
    "smoking": 2,
    "family_history": 2,
    "asymptomatic_pattern": 1,
    "obesity_bmi": 1,
    "age_over_55": 1,
    "low_physical_activity": 1,
    "sex_male": 1,
}


def get_rule_weight(rule_key: str) -> int:
	try:
		return RULE_WEIGHTS[rule_key]
	except KeyError as exc:
		raise KeyError(f"Chưa cấu hình trọng số cho luật: {rule_key}") from exc