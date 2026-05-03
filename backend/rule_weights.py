"""Cấu hình trọng số cho các luật của hệ chuyên gia."""

from __future__ import annotations


RULE_WEIGHTS: dict[str, float] = {
    # ── NHÓM ƯU TIÊN ──────────────────────────────────────────────────────────
    # Bệnh tim bẩm sinh được quy ước = nguy cơ tuyệt đối 100%
    "congenital_heart_disease": 100.0,

    # ── NHÓM NẶNG (Dấu hiệu đỏ) ───────────────────────────────────────────────
    # Các chỉ số vàng (golden indicators) được ưu tiên điểm rất cao
    "typical_angina":    70.0,   # Đau thắt ngực điển hình (chỉ số vàng, severe)
    "major_vessels_3":   70.0,   # 3 mạch lớn bị ảnh hưởng (chỉ số vàng, severe)
    "st_depression_severe": 70.0, # ST chênh nặng (chỉ số vàng, severe)
    "thal_defect":       70.0,   # Thal bất thường (chỉ số vàng, severe)

    # Các mức khác của chỉ số vàng và các chỉ số nặng khác được tăng điểm để tổng phần trăm cao hơn
    "atypical_angina":    8.0,   # Đau thắt ngực không điển hình (moderate)
    "non_anginal_pain":   4.0,   # Đau không do thắt ngực (mild)
    "asymptomatic_pattern": 1.0, # Không triệu chứng

    "exercise_angina":   25.0,   # Đau ngực khi gắng sức

    "major_vessels_2":   8.0,    # 2 mạch lớn bị ảnh hưởng (moderate)
    "major_vessels_1":   3.0,    # 1 mạch lớn bị ảnh hưởng (mild)

    "st_depression_moderate": 8.0,   # ST chênh vừa (moderate)
    "st_depression_mild":     3.0,   # ST chênh nhẹ (mild)

    # Triệu chứng đi kèm (nhị phân Có/Không)
    "shortness_of_breath": 25.0,   # Khó thở
    "leg_swelling":        15.0,   # Phù chân

    # ── NHÓM TRUNG BÌNH ────────────────────────────────────────────────────────

    # Huyết áp và chuyển hóa
    "high_resting_bp":    5.0,
    "diagnosed_diabetes": 5.0,
    "fasting_blood_sugar": 4.0,

    # ECG lúc nghỉ (resting_ecg) — 2 mức bất thường, điểm khác nhau
    "ecg_lvh":         5.0,
    "ecg_st_abnormal": 4.0,

    # Độ dốc ST (slope) — 2 mức bất lợi, điểm khác nhau
    "slope_downslope": 4.0,
    "slope_flat":      3.0,

    "low_max_heart_rate": 3.0,
    "palpitations":      3.0,

    # ── NHÓM NHẸ (Yếu tố nền) ─────────────────────────────────────────────────
    # Mức độ vận động thể lực (physical_activity_level) — 2 mức nguy cơ
    "very_low_physical_activity": 2.0,
    "low_physical_activity":      1.0,
    #                                 # Mức 2: Cao      → không kích hoạt luật

    # Yếu tố lối sống và nhân khẩu
    "smoking":        1.0,
    "family_history": 1.0,
    "obesity_bmi":    0.5,
    "age_over_55":    0.5,
    "sex_male":       0.5,
}


def get_rule_weight(rule_key: str) -> int:
	try:
		return RULE_WEIGHTS[rule_key]
	except KeyError as exc:
		raise KeyError(f"Chưa cấu hình trọng số cho luật: {rule_key}") from exc
