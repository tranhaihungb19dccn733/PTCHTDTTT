"""Cấu hình trọng số cho các luật của hệ chuyên gia."""

from __future__ import annotations


RULE_WEIGHTS: dict[str, int] = {
    # ── NHÓM ƯU TIÊN ──────────────────────────────────────────────────────────
    # Bệnh tim bẩm sinh được quy ước = nguy cơ tuyệt đối 100%
    "congenital_heart_disease": 100,

    # ── NHÓM NẶNG (Dấu hiệu đỏ) ───────────────────────────────────────────────
    # Kiểu đau ngực (chest_pain_type) — 4 mức, điểm giảm dần
    "typical_angina":    65,   # Mức 0: Đau thắt ngực điển hình       → nguy cơ cao nhất
    "atypical_angina":   30,   # Mức 1: Đau thắt ngực không điển hình → nguy cơ cao
    "non_anginal_pain":  15,   # Mức 2: Đau không do thắt ngực        → nguy cơ trung bình
    "asymptomatic_pattern": 1, # Mức 3: Không triệu chứng             → ẩn nhưng không loại trừ

    # Đau thắt ngực khi gắng sức (exercise_angina)
    "exercise_angina": 65,     # Có đau ngực khi gắng sức             → dấu hiệu thiếu máu cơ tim mạnh

    # Số mạch lớn ảnh hưởng (num_major_vessels) — 3 mức theo số lượng
    "major_vessels_3": 60,     # 3 mạch bị ảnh hưởng                  → tổn thương lan rộng, nguy cơ rất cao
    "major_vessels_2": 45,     # 2 mạch bị ảnh hưởng                  → tổn thương đa mạch, nguy cơ cao
    "major_vessels_1": 25,     # 1 mạch bị ảnh hưởng                  → tổn thương khu trú, cần theo dõi

    # Độ chênh ST sau gắng sức (oldpeak) — 3 mức theo biên độ
    "st_depression_severe":   60,  # oldpeak ≥ 3.0                    → nặng, gợi ý thiếu máu cơ tim rõ
    "st_depression_moderate": 40,  # oldpeak 2.0 – 2.9                → trung bình-nặng
    "st_depression_mild":     20,  # oldpeak 1.0 – 1.9                → nhẹ, cần theo dõi

    # Triệu chứng đi kèm (nhị phân Có/Không)
    "shortness_of_breath": 35,     # Khó thở                          → gợi ý suy tim hoặc thiếu máu cơ tim
    "leg_swelling":        35,     # Phù chân                         → gợi ý ứ dịch ngoại biên

    # ── NHÓM TRUNG BÌNH ────────────────────────────────────────────────────────
    # Mức độ biểu hiện triệu chứng (symptom_severity) — 4 mức, điểm tăng dần
    "symptom_severity_high":     25,  # Mức 3: Triệu chứng rõ rệt, nghi ngờ đã mắc bệnh
    "symptom_severity_moderate": 15,  # Mức 2: Nhiều triệu chứng nghi ngờ bệnh tim
    "symptom_severity_mild":      5,  # Mức 1: Dấu hiệu nhẹ hoặc thoáng qua
    #                                 # Mức 0: Bình thường              → không kích hoạt luật

    # Huyết áp và chuyển hóa
    "high_resting_bp":    20,  # Huyết áp lúc nghỉ ≥ 140 mmHg
    "diagnosed_diabetes": 20,  # Đã chẩn đoán tiểu đường
    "fasting_blood_sugar": 15, # Đường huyết lúc đói > 120 mg/dl
    "thal_defect":        20,  # Kết quả thal bất thường (khiếm khuyết cố định/hồi phục)

    # ECG lúc nghỉ (resting_ecg) — 2 mức bất thường, điểm khác nhau
    "ecg_lvh":         20,  # Mức 2: Phì đại thất trái              → tổn thương cơ tim đáng kể
    "ecg_st_abnormal": 15,  # Mức 1: Sóng ST-T bất thường           → rối loạn tái cực

    # Độ dốc ST (slope) — 2 mức bất lợi, điểm khác nhau
    "slope_downslope": 15,  # Mức 2: Độ dốc ST giảm (downsloping)  → nguy cơ bệnh mạch vành cao hơn
    "slope_flat":      10,  # Mức 1: Độ dốc ST ngang (flat)        → ít thuận lợi hơn dốc lên

    "low_max_heart_rate": 10,  # Nhịp tim tối đa thấp hơn kỳ vọng theo tuổi
    "palpitations":      10,  # Hồi hộp — gợi ý rối loạn nhịp tim

    # ── NHÓM NHẸ (Yếu tố nền) ─────────────────────────────────────────────────
    # Mức độ vận động thể lực (physical_activity_level) — 2 mức nguy cơ
    "very_low_physical_activity": 5,  # Mức 0: Thấp    → lười vận động, nguy cơ rõ hơn
    "low_physical_activity":      3,  # Mức 1: Trung bình → chưa đạt khuyến nghị
    #                                 # Mức 2: Cao      → không kích hoạt luật

    # Yếu tố lối sống và nhân khẩu
    "smoking":        2,  # Đang hút thuốc lá
    "family_history": 2,  # Tiền sử gia đình có bệnh tim mạch
    "obesity_bmi":    1,  # BMI ≥ 30 (béo phì)
    "age_over_55":    1,  # Tuổi trên 55
    "sex_male":       1,  # Giới tính nam
}


def get_rule_weight(rule_key: str) -> int:
	try:
		return RULE_WEIGHTS[rule_key]
	except KeyError as exc:
		raise KeyError(f"Chưa cấu hình trọng số cho luật: {rule_key}") from exc