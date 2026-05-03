"""Hệ chuyên gia dựa trên luật để đánh giá sơ bộ nguy cơ bệnh tim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from rule_weights import RULE_WEIGHTS, get_rule_weight


RuleCheck = Callable[[dict], bool]


@dataclass(frozen=True)
class ExpertRule:
	key: str
	score: int
	title: str
	message: str
	recommendation: str
	check: RuleCheck


class HeartDiseaseKBS:
	"""Hệ chuyên gia sử dụng các luật IF-THEN có trọng số."""
	def __init__(self) -> None:
		self.rules = [
			# ── NHÓM YẾU TỐ NỀN ────────────────────────────────────────────────

			# Luật 1: Béo phì theo BMI
			ExpertRule(
				key="obesity_bmi",
				score=get_rule_weight("obesity_bmi"),
				title="BMI ở mức béo phì",
				message="BMI từ 30 trở lên làm tăng nguy cơ tim mạch và hội chứng chuyển hóa.",
				recommendation="Nên kiểm soát cân nặng, giảm mỡ nội tạng và duy trì chế độ ăn ít năng lượng dư thừa.",
				check=lambda data: self._bmi(data) >= 30,
			),
			# Luật 2: Hút thuốc lá
			ExpertRule(
				key="smoking",
				score=get_rule_weight("smoking"),
				title="Hút thuốc lá",
				message="Hút thuốc là yếu tố nguy cơ mạnh của bệnh mạch vành và rối loạn mạch máu.",
				recommendation="Cần ngừng hút thuốc càng sớm càng tốt và hạn chế tiếp xúc khói thuốc thụ động.",
				check=lambda data: data["smoking"] == 1,
			),
			# Luật 3: Tiền sử gia đình bệnh tim
			ExpertRule(
				key="family_history",
				score=get_rule_weight("family_history"),
				title="Tiền sử gia đình bệnh tim",
				message="Tiền sử gia đình làm tăng khả năng xuất hiện bệnh tim mạch sớm.",
				recommendation="Nên tầm soát tim mạch sớm hơn và theo dõi định kỳ nếu gia đình có người mắc bệnh tim.",
				check=lambda data: data["family_history"] == 1,
			),
			# Luật 4: Bệnh tim bẩm sinh — ưu tiên tuyệt đối
			ExpertRule(
				key="congenital_heart_disease",
				score=get_rule_weight("congenital_heart_disease"),
				title="Có bệnh tim bẩm sinh",
				message="Tiền sử tim bẩm sinh là yếu tố nền quan trọng khi đánh giá nguy cơ tim mạch và biến chứng về sau.",
				recommendation="Cần theo dõi với bác sĩ tim mạch, mang theo hồ sơ chẩn đoán cũ và ưu tiên siêu âm tim hoặc kiểm tra chuyên sâu khi có triệu chứng.",
				check=lambda data: data["congenital_heart_disease"] == 1,
			),
			# Luật 5: Đã được chẩn đoán tiểu đường
			ExpertRule(
				key="diagnosed_diabetes",
				score=get_rule_weight("diagnosed_diabetes"),
				title="Đã được chẩn đoán tiểu đường",
				message="Tiểu đường là bệnh nền làm gia tăng nguy cơ biến cố tim mạch.",
				recommendation="Cần kiểm soát đường huyết chặt chẽ và đánh giá định kỳ các yếu tố nguy cơ tim mạch đi kèm.",
				check=lambda data: data["diagnosed_diabetes"] == 1,
			),
			# Luật 6: Nguy cơ tim mạch theo tuổi
			ExpertRule(
				key="age_over_55",
				score=get_rule_weight("age_over_55"),
				title="Nguy cơ tim mạch theo tuổi",
				message="Tuổi trên 55 làm tăng nguy cơ mắc bệnh tim mạch.",
				recommendation="Theo dõi định kỳ huyết áp, mỡ máu và đánh giá nguy cơ tim mạch hằng năm.",
				check=lambda data: data["age"] >= 55,
			),
			# Luật 7: Giới tính nam
			ExpertRule(
				key="sex_male",
				score=get_rule_weight("sex_male"),
				title="Giới tính nam",
				message="Nam giới có nguy cơ mắc bệnh tim mạch cao hơn so với nữ giới ở cùng độ tuổi.",
				recommendation="Tăng cường tầm soát tim mạch định kỳ và duy trì lối sống lành mạnh.",
				check=lambda data: data["sex"] == 1,
			),

			# ── MỨC ĐỘ VẬN ĐỘNG THỂ LỰC (physical_activity_level) ─────────────
			# 3 mức: 0=Thấp, 1=Trung bình, 2=Cao → chỉ mức 0 và 1 kích hoạt luật
			# Luật 8a: Vận động mức 0 — Thấp (nguy cơ cao hơn)
			ExpertRule(
				key="very_low_physical_activity",
				score=get_rule_weight("very_low_physical_activity"),
				title="Mức vận động thể lực: Thấp",
				message="Hầu như không vận động thể lực làm tăng nguy cơ tăng cân, rối loạn chuyển hóa và bệnh tim mạch.",
				recommendation="Nên bắt đầu tăng dần hoạt động thể lực, tối thiểu 150 phút/tuần nếu không có chống chỉ định.",
				check=lambda data: data["physical_activity_level"] == 0,
			),
			# Luật 8b: Vận động mức 1 — Trung bình (chưa đạt khuyến nghị)
			ExpertRule(
				key="low_physical_activity",
				score=get_rule_weight("low_physical_activity"),
				title="Mức vận động thể lực: Trung bình",
				message="Mức vận động trung bình chưa đủ để đạt lợi ích tim mạch tối ưu theo khuyến nghị.",
				recommendation="Nên tăng cường thêm cường độ hoặc thời gian tập luyện để đạt mục tiêu 150–300 phút/tuần.",
				check=lambda data: data["physical_activity_level"] == 1,
			),
			# [Mức 2 — Cao: không kích hoạt luật, không cộng điểm nguy cơ]
			ExpertRule(
				key="typical_angina",
				score=get_rule_weight("typical_angina"),
				title="Kiểu đau ngực: Điển hình",
				message="Đau thắt ngực điển hình là dấu hiệu cảnh báo lớn nhất đối với bệnh mạch vành.",
				recommendation="Cần ưu tiên khám chuyên khoa tim mạch và cân nhắc nghiệm pháp gắng sức.",
				check=lambda data: data["chest_pain_type"] == 0,
			),
			# Luật 10b: Kiểu 1 — Đau thắt ngực không điển hình
			ExpertRule(
				key="atypical_angina",
				score=get_rule_weight("atypical_angina"),
				title="Kiểu đau ngực: Không điển hình",
				message="Đau thắt ngực không điển hình vẫn có liên quan đến bệnh mạch vành, đặc biệt ở phụ nữ và người cao tuổi.",
				recommendation="Nên đánh giá nguy cơ tổng thể và cân nhắc xét nghiệm bổ sung nếu đi kèm yếu tố nguy cơ khác.",
				check=lambda data: data["chest_pain_type"] == 1,
			),
			# Luật 10c: Kiểu 2 — Đau không do thắt ngực
			ExpertRule(
				key="non_anginal_pain",
				score=get_rule_weight("non_anginal_pain"),
				title="Kiểu đau ngực: Không do thắt ngực",
				message="Đau ngực không phải thắt ngực ít liên quan đến thiếu máu cơ tim nhưng không thể loại trừ hoàn toàn.",
				recommendation="Nên xem xét các nguyên nhân khác (tiêu hóa, cơ xương) đồng thời không bỏ qua đánh giá tim mạch.",
				check=lambda data: data["chest_pain_type"] == 2,
			),
			# Luật 10d: Kiểu 3 — Không triệu chứng (ẩn nhưng không loại trừ)
			ExpertRule(
				key="asymptomatic_pattern",
				score=get_rule_weight("asymptomatic_pattern"),
				title="Kiểu đau ngực: Không triệu chứng",
				message="Dạng không triệu chứng nhưng đi kèm chỉ số bất thường vẫn có thể nguy hiểm.",
				recommendation="Không nên chủ quan; cần kiểm tra ECG, men tim và đối chiếu lịch sử nguy cơ.",
				check=lambda data: data["chest_pain_type"] == 3,
			),

			# ── ECG LÚC NGHỈ (resting_ecg) ───────────────────────────────────────
			# 3 mức: 0=Bình thường, 1=ST-T bất thường, 2=Phì đại thất trái
			# Luật 11a: ECG mức 1 — ST-T bất thường
			ExpertRule(
				key="ecg_st_abnormal",
				score=get_rule_weight("ecg_st_abnormal"),
				title="ECG lúc nghỉ: ST-T bất thường",
				message="Sóng ST-T bất thường trên ECG gợi ý rối loạn tái cực, có thể liên quan thiếu máu cơ tim.",
				recommendation="Cần được đọc ECG bởi bác sĩ tim mạch và đối chiếu với triệu chứng lâm sàng.",
				check=lambda data: data["resting_ecg"] == 1,
			),
			# Luật 11b: ECG mức 2 — Phì đại thất trái (nặng hơn)
			ExpertRule(
				key="ecg_lvh",
				score=get_rule_weight("ecg_lvh"),
				title="ECG lúc nghỉ: Phì đại thất trái",
				message="Phì đại thất trái trên ECG phản ánh tổn thương cơ tim mạn tính, thường liên quan tăng huyết áp kéo dài.",
				recommendation="Nên đánh giá chức năng tim bằng siêu âm tim và kiểm soát huyết áp chặt chẽ.",
				check=lambda data: data["resting_ecg"] == 2,
			),
			# [Mức 0 — Bình thường: không kích hoạt luật]

			# ── ĐỘ CHÊNH ST SAU GẮNG SỨC — OLDPEAK (st_depression) ──────────────
			# 3 ngưỡng: nhẹ (1.0–1.9), trung bình (2.0–2.9), nặng (≥3.0)
			# Luật 12a: Oldpeak 1.0–1.9 — Nhẹ
			ExpertRule(
				key="st_depression_mild",
				score=get_rule_weight("st_depression_mild"),
				title="Độ chênh ST: Nhẹ (oldpeak 1.0 – 1.9)",
				message="Độ chênh ST nhẹ sau gắng sức cần được theo dõi cùng các dấu hiệu lâm sàng khác.",
				recommendation="Cần theo dõi bằng ECG gắng sức hoặc siêu âm tim gắng sức để đánh giá thêm.",
				check=lambda data: 1.0 <= data["oldpeak"] < 2.0,
			),
			# Luật 12b: Oldpeak 2.0–2.9 — Trung bình
			ExpertRule(
				key="st_depression_moderate",
				score=get_rule_weight("st_depression_moderate"),
				title="Độ chênh ST: Trung bình (oldpeak 2.0 – 2.9)",
				message="Độ chênh ST trung bình là dấu hiệu đáng lo ngại, liên quan rõ đến thiếu máu cơ tim khi gắng sức.",
				recommendation="Nên được khám chuyên khoa tim mạch và cân nhắc xét nghiệm hình ảnh chức năng tim.",
				check=lambda data: 2.0 <= data["oldpeak"] < 3.0,
			),
			# Luật 12c: Oldpeak ≥ 3.0 — Nặng
			ExpertRule(
				key="st_depression_severe",
				score=get_rule_weight("st_depression_severe"),
				title="Độ chênh ST: Nặng (oldpeak ≥ 3.0)",
				message="Độ chênh ST nặng là dấu hiệu mạnh của thiếu máu cơ tim diện rộng hoặc bệnh mạch vành nặng.",
				recommendation="Cần ưu tiên thăm khám tim mạch khẩn và tránh gắng sức mạnh trước khi có đánh giá chuyên khoa.",
				check=lambda data: data["oldpeak"] >= 3.0,
			),

			# ── ĐỘ DỐC ST (slope) ────────────────────────────────────────────────
			# 3 mức: 0=Dốc lên (thuận lợi), 1=Ngang, 2=Dốc xuống (bất lợi nhất)
			# Luật 13a: Slope 1 — Ngang (flat)
			ExpertRule(
				key="slope_flat",
				score=get_rule_weight("slope_flat"),
				title="Độ dốc ST: Ngang",
				message="Đoạn ST ngang sau gắng sức ít thuận lợi hơn dốc lên, có thể liên quan nguy cơ bệnh mạch vành.",
				recommendation="Cần tổng hợp với ECG và triệu chứng để quyết định hướng can thiệp tiếp theo.",
				check=lambda data: data["slope"] == 1,
			),
			# Luật 13b: Slope 2 — Dốc xuống (bất lợi nhất)
			ExpertRule(
				key="slope_downslope",
				score=get_rule_weight("slope_downslope"),
				title="Độ dốc ST: Dốc xuống",
				message="Đoạn ST dốc xuống sau gắng sức là dấu hiệu bất lợi rõ nhất, thường liên quan bệnh mạch vành.",
				recommendation="Cần đánh giá sâu hơn bằng chụp mạch hoặc xét nghiệm chức năng tim nếu kết hợp với triệu chứng.",
				check=lambda data: data["slope"] == 2,
			),
			# [Slope 0 — Dốc lên: thuận lợi, không kích hoạt luật]

			# ── SỐ MẠCH LỚN BỊ ẢNH HƯỞNG (num_major_vessels) ───────────────────
			# 4 mức: 0=Không, 1=1 mạch, 2=2 mạch, 3=3 mạch → điểm tăng theo số lượng
			# Luật 14a: 1 mạch ảnh hưởng
			ExpertRule(
				key="major_vessels_1",
				score=get_rule_weight("major_vessels_1"),
				title="Số mạch ảnh hưởng: 1 mạch",
				message="Một mạch lớn bị ảnh hưởng cho thấy tổn thương khu trú, cần theo dõi và đánh giá thêm.",
				recommendation="Nên đánh giá sâu hơn bởi bác sĩ tim mạch, có thể cần siêu âm tim hoặc chụp mạch.",
				check=lambda data: data["num_major_vessels"] == 1,
			),
			# Luật 14b: 2 mạch ảnh hưởng
			ExpertRule(
				key="major_vessels_2",
				score=get_rule_weight("major_vessels_2"),
				title="Số mạch ảnh hưởng: 2 mạch",
				message="Hai mạch lớn bị ảnh hưởng là tổn thương đa mạch, nguy cơ biến cố tim mạch tăng đáng kể.",
				recommendation="Cần tư vấn can thiệp mạch vành; bác sĩ tim mạch can thiệp nên được tham gia đánh giá.",
				check=lambda data: data["num_major_vessels"] == 2,
			),
			# Luật 14c: 3 mạch trở lên ảnh hưởng (nặng nhất)
			ExpertRule(
				key="major_vessels_3",
				score=get_rule_weight("major_vessels_3"),
				title="Số mạch ảnh hưởng: 3 mạch",
				message="Ba mạch lớn trở lên bị ảnh hưởng là tổn thương lan rộng, nguy cơ biến cố rất cao.",
				recommendation="Cần can thiệp chuyên sâu khẩn; đánh giá tái thông mạch máu (can thiệp hoặc phẫu thuật) sớm.",
				check=lambda data: data["num_major_vessels"] >= 3,
			),
			# [0 mạch ảnh hưởng: không kích hoạt luật]

			# ── CÁC DẤU HIỆU LÂM SÀNG KHÁC ─────────────────────────────────────

			# Luật 15: Huyết áp lúc nghỉ tăng cao
			ExpertRule(
				key="high_resting_bp",
				score=get_rule_weight("high_resting_bp"),
				title="Huyết áp lúc nghỉ tăng cao",
				message="Huyết áp lúc nghỉ ≥ 140 mmHg liên quan đến nguy cơ tăng huyết áp và tổn thương mạch máu.",
				recommendation="Điều chỉnh lối sống, theo dõi huyết áp tại nhà và cân nhắc điều trị theo chỉ định bác sĩ.",
				check=lambda data: data["resting_bp"] >= 140,
			),
			# Luật 16: Đường huyết lúc đói tăng
			ExpertRule(
				key="fasting_blood_sugar",
				score=get_rule_weight("fasting_blood_sugar"),
				title="Đường huyết lúc đói tăng",
				message="Đường huyết lúc đói > 120 mg/dl có liên quan đến đái tháo đường và biến chứng tim mạch.",
				recommendation="Theo dõi đường huyết và kiểm tra HbA1c nếu chưa được đánh giá.",
				check=lambda data: data["fasting_blood_sugar"] == 1,
			),
			# Luật 17: Nhịp tim tối đa thấp hơn kỳ vọng
			ExpertRule(
				key="low_max_heart_rate",
				score=get_rule_weight("low_max_heart_rate"),
				title="Nhịp tim tối đa thấp hơn kỳ vọng",
				message="Nhịp tim tối đa đạt được thấp hơn mong đợi có thể gợi ý hạn chế gắng sức do vấn đề tim mạch.",
				recommendation="Cân nhắc nghiệm pháp gắng sức nếu có triệu chứng khi vận động.",
				check=lambda data: data["max_heart_rate"] < max(120, 220 - data["age"] - 25),
			),
			# Luật 18: Khó thở
			ExpertRule(
				key="shortness_of_breath",
				score=get_rule_weight("shortness_of_breath"),
				title="Triệu chứng khó thở",
				message="Khó thở có thể liên quan đến suy tim, thiếu máu cơ tim hoặc giảm dung nạp gắng sức.",
				recommendation="Nếu khó thở xuất hiện thường xuyên hoặc tăng dần, nên khám tim mạch để làm rõ nguyên nhân.",
				check=lambda data: data["shortness_of_breath"] == 1,
			),
			# Luật 19: Phù chân
			ExpertRule(
				key="leg_swelling",
				score=get_rule_weight("leg_swelling"),
				title="Phù chân",
				message="Phù chân có thể gợi ý ứ dịch ngoại biên liên quan đến chức năng tim hoặc tuần hoàn.",
				recommendation="Cần đánh giá thêm chức năng tim, thận và tình trạng giữ nước nếu phù chân kéo dài.",
				check=lambda data: data["leg_swelling"] == 1,
			),
			# Luật 20: Hồi hộp
			ExpertRule(
				key="palpitations",
				score=get_rule_weight("palpitations"),
				title="Triệu chứng hồi hộp",
				message="Hồi hộp có thể liên quan đến rối loạn nhịp tim hoặc đáp ứng tim mạch bất thường.",
				recommendation="Nên theo dõi điện tim hoặc Holter nếu hồi hộp xuất hiện lặp lại hoặc kèm khó chịu ngực.",
				check=lambda data: data["palpitations"] == 1,
			),
			# Luật 21: Đau thắt ngực khi gắng sức
			ExpertRule(
				key="exercise_angina",
				score=get_rule_weight("exercise_angina"),
				title="Đau thắt ngực khi gắng sức",
				message="Đau thắt ngực khi gắng sức là dấu hiệu mạnh của tình trạng thiếu máu cơ tim.",
				recommendation="Nên được thăm khám sớm và hạn chế gắng sức mạnh trước khi có đánh giá chuyên khoa.",
				check=lambda data: data["exercise_angina"] == 1,
			),
			# Luật 22: Kết quả thal bất thường
			ExpertRule(
				key="thal_defect",
				score=get_rule_weight("thal_defect"),
				title="Kết quả thal bất thường",
				message="Kết quả thal bất thường (khiếm khuyết cố định hoặc hồi phục) thường đi kèm nguy cơ tổn thương mạch vành.",
				recommendation="Cần xem lại kết quả xét nghiệm và đối chiếu với chẩn đoán hình ảnh nếu có.",
				check=lambda data: data["thal"] in {1, 2},
			),
		]
		self._validate_rule_weights()
		self.max_score = sum(rule.score for rule in self.rules)

	def _validate_rule_weights(self) -> None:
		rule_keys = {rule.key for rule in self.rules}
		missing_keys = sorted(rule_keys - set(RULE_WEIGHTS))
		extra_keys = sorted(set(RULE_WEIGHTS) - rule_keys)
		if missing_keys or extra_keys:
			messages = []
			if missing_keys:
				messages.append(f"thiếu cấu hình cho: {', '.join(missing_keys)}")
			if extra_keys:
				messages.append(f"dư cấu hình không dùng tới: {', '.join(extra_keys)}")
			raise ValueError("Rule weight config không đồng bộ: " + "; ".join(messages))

	def analyze(self, data: dict) -> dict:
		triggered = []
		recommendations = []
		total_score = 0

		# Danh sách các chỉ số vàng (golden indicators)
		golden_keys = [
			"typical_angina",         # Đau thắt ngực điển hình
			"major_vessels_3",        # 3 mạch lớn bị ảnh hưởng
			"st_depression_severe",   # ST chênh nặng
			"thal_defect",            # Thal bất thường
		]
		golden_activated = []

		for rule in self.rules:
			if rule.check(data):
				total_score += rule.score
				triggered.append(
					{
						"key": rule.key,
						"title": rule.title,
						"score": rule.score,
						"message": rule.message,
					}
				)
				if rule.recommendation not in recommendations:
					recommendations.append(rule.recommendation)
				# Kiểm tra nếu là chỉ số vàng và được kích hoạt
				if rule.key in golden_keys:
					golden_activated.append(rule.key)

		# Hàm tính điểm phi tuyến cho các chỉ số vàng
		def nonlinear_golden_score(num_golden):
			"""
			Tính điểm phi tuyến cho các chỉ số vàng:
			- 1 chỉ số vàng: ~70 điểm
			- 2 chỉ số vàng: ~85 điểm
			- 3 chỉ số vàng: ~95 điểm
			- 4 chỉ số vàng: 100 điểm
			"""
			if num_golden <= 0:
				return 0
			nonlinear_map = {1: 70, 2: 85, 3: 95, 4: 100}
			return nonlinear_map.get(num_golden, 100)

		if data["congenital_heart_disease"] == 1:
			total_score = self.max_score
			normalized_score = 100.0
			risk_level = "high"
		elif golden_activated:
			# Nếu có ít nhất 1 chỉ số vàng, dùng điểm phi tuyến
			normalized_score = nonlinear_golden_score(len(golden_activated))
			risk_level = self._risk_level(normalized_score)
		else:
			# Nếu không có chỉ số vàng, dùng tổng điểm các luật khác (chuẩn hóa về 100)
			normalized_score = min(100, round((total_score / self.max_score) * 100, 1))
			risk_level = self._risk_level(normalized_score)

		summary = self._build_summary(data, risk_level, triggered)

		return {
			"raw_score": total_score,
			"normalized_score": normalized_score,
			"risk_level": risk_level,
			"rule_hits": triggered,
			"recommendations": recommendations or [
				"Duy trì lối sống lành mạnh và tiếp tục theo dõi sức khỏe định kỳ."
			],
			"summary": summary,
		}

	def _risk_level(self, normalized_score: float) -> str:
		if normalized_score >= 70:
			return "high"
		if normalized_score >= 40:
			return "moderate"
		return "low"

	def _bmi(self, data: dict) -> float:
		height_m = data["height"] / 100
		if height_m <= 0:
			return 0
		return data["weight"] / (height_m * height_m)

	def _build_summary(self, data: dict, risk_level: str, triggered: list[dict]) -> str:
		sex_label = "nam" if data["sex"] == 1 else "nữ"
		if data["congenital_heart_disease"] == 1:
			return (
				f"Bệnh nhân {sex_label}, {data['age']} tuổi có bệnh tim bẩm sinh. "
				"Hệ thống quy ước đây là ca bệnh tim xác định và gán mức nguy cơ 100%."
			)

		if not triggered:
			return (
				f"Hồ sơ của bệnh nhân {sex_label}, {data['age']} tuổi hiện không kích hoạt các luật nguy cơ lớn. "
				"Vẫn nên duy trì kiểm tra sức khỏe tim mạch định kỳ."
			)

		top_findings = ", ".join(item["title"].lower() for item in triggered[:3])
		risk_text = {
			"low": "nguy cơ thấp",
			"moderate": "nguy cơ trung bình",
			"high": "nguy cơ cao",
		}[risk_level]
		return (
			f"Hệ thống suy luận đánh giá bệnh nhân {sex_label}, {data['age']} tuổi ở mức {risk_text}. "
			f"Các yếu tố nổi bật: {top_findings}."
		)

