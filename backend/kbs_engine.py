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
			# Luật 4: Bệnh tim bẩm sinh
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
			# Luật 6: Mức vận động thể lực thấp
			ExpertRule(
				key="low_physical_activity",
				score=get_rule_weight("low_physical_activity"),
				title="Mức vận động thể lực thấp",
				message="Ít vận động làm tăng nguy cơ tăng cân, rối loạn chuyển hóa và bệnh tim mạch.",
				recommendation="Nên tăng dần hoạt động thể lực phù hợp, tối thiểu 150 phút mỗi tuần nếu không có chống chỉ định.",
				check=lambda data: data["physical_activity_level"] == 0,
			),
			# Luật 7: Nhiều triệu chứng nghi ngờ bệnh tim (mức trung bình)
			ExpertRule(
				key="symptom_severity_moderate",
				score=get_rule_weight("symptom_severity_moderate"),
				title="Nhiều triệu chứng nghi ngờ bệnh tim",
				message="Người bệnh đang có nhiều triệu chứng gợi ý bất thường tim mạch cần được theo dõi kỹ.",
				recommendation="Nên tổng hợp triệu chứng với các chỉ số tim mạch để đánh giá đầy đủ hơn.",
				check=lambda data: data["symptom_severity"] == 2,
			),
			# Luật 8: Triệu chứng bệnh rõ rệt (mức cao)
			ExpertRule(
				key="symptom_severity_high",
				score=get_rule_weight("symptom_severity_high"),
				title="Triệu chứng bệnh rõ rệt",
				message="Mức biểu hiện triệu chứng rõ rệt làm tăng khả năng người bệnh đã ở giai đoạn nguy cơ cao hoặc đã mắc bệnh.",
				recommendation="Cần ưu tiên khám chuyên khoa tim mạch sớm nếu triệu chứng kéo dài hoặc nặng dần.",
				check=lambda data: data["symptom_severity"] == 3,
			),
			# Luật 9: Nguy cơ tim mạch theo tuổi (trên 55)
			ExpertRule(
				key="age_over_55",
				score=get_rule_weight("age_over_55"),
				title="Nguy cơ tim mạch theo tuổi",
				message="Tuổi trên 55 làm tăng nguy cơ mắc bệnh tim mạch.",
				recommendation="Theo dõi định kỳ huyết áp, mỡ máu và đánh giá nguy cơ tim mạch hằng năm.",
				check=lambda data: data["age"] >= 55,
			),
			# Luật 10: Đau thắt ngực điển hình
			ExpertRule(
				key="typical_angina",
				score=get_rule_weight("typical_angina"),
				title="Kiểu đau thắt ngực điển hình",
				message="Kiểu đau ngực điển hình là dấu hiệu cảnh báo lớn đối với bệnh mạch vành.",
				recommendation="Cần ưu tiên khám chuyên khoa tim mạch và cân nhắc nghiệm pháp gắng sức.",
				check=lambda data: data["chest_pain_type"] == 0,
			),
			# Luật 11: Hồ sơ nguy cơ cao ít triệu chứng (không điển hình)
			ExpertRule(
				key="asymptomatic_pattern",
				score=get_rule_weight("asymptomatic_pattern"),
				title="Hồ sơ nguy cơ cao ít triệu chứng",
				message="Dạng đau ngực không triệu chứng nhưng đi kèm chỉ số bất thường vẫn có thể nguy hiểm.",
				recommendation="Không nên chủ quan; cần kiểm tra ECG, men tim và lịch sử nguy cơ.",
				check=lambda data: data["chest_pain_type"] == 3,
			),
			# Luật 12: Huyết áp lúc nghỉ tăng cao
			ExpertRule(
				key="high_resting_bp",
				score=get_rule_weight("high_resting_bp"),
				title="Huyết áp lúc nghỉ tăng cao",
				message="Huyết áp lúc nghỉ cao liên quan đến nguy cơ tăng huyết áp và tổn thương mạch máu.",
				recommendation="Điều chỉnh lối sống, theo dõi huyết áp tại nhà và cân nhắc điều trị theo chỉ định bác sĩ.",
				check=lambda data: data["resting_bp"] >= 140,
			),
			# Luật 13: Giới tính nam
			ExpertRule(
				key="sex_male",
				score=get_rule_weight("sex_male"),
				title="Giới tính nam",
				message="Nam giới có nguy cơ mắc bệnh tim mạch cao hơn so với nữ giới ở cùng độ tuổi.",
				recommendation="Tăng cường tầm soát tim mạch định kỳ và duy trì lối sống lành mạnh.",
				check=lambda data: data["sex"] == 1,
			),
			# Luật 14: Đường huyết lúc đói tăng
			ExpertRule(
				key="fasting_blood_sugar",
				score=get_rule_weight("fasting_blood_sugar"),
				title="Đường huyết lúc đói tăng",
				message="Đường huyết lúc đói cao có liên quan đến đái tháo đường và biến chứng tim mạch.",
				recommendation="Theo dõi đường huyết và kiểm tra HbA1c nếu chưa được đánh giá.",
				check=lambda data: data["fasting_blood_sugar"] == 1,
			),
			# Luật 15: ECG lúc nghỉ bất thường
			ExpertRule(
				key="abnormal_ecg",
				score=get_rule_weight("abnormal_ecg"),
				title="ECG lúc nghỉ bất thường",
				message="Điện tâm đồ lúc nghỉ bất thường có thể gợi ý rối loạn dẫn truyền hoặc thiếu máu cơ tim.",
				recommendation="Cần được đọc ECG bởi bác sĩ tim mạch và đối chiếu với triệu chứng lâm sàng.",
				check=lambda data: data["resting_ecg"] in {1, 2},
			),
			# Luật 16: Nhịp tim tối đa thấp hơn kỳ vọng
			ExpertRule(
				key="low_max_heart_rate",
				score=get_rule_weight("low_max_heart_rate"),
				title="Nhịp tim tối đa thấp hơn kỳ vọng",
				message="Nhịp tim tối đa đạt được thấp hơn mong đợi có thể gợi ý hạn chế gắng sức do vấn đề tim mạch.",
				recommendation="Cân nhắc nghiệm pháp gắng sức nếu có triệu chứng khi vận động.",
				check=lambda data: data["max_heart_rate"] < max(120, 220 - data["age"] - 25),
			),
			# Luật 17: Khó thở
			ExpertRule(
				key="shortness_of_breath",
				score=get_rule_weight("shortness_of_breath"),
				title="Triệu chứng khó thở",
				message="Khó thở có thể liên quan đến suy tim, thiếu máu cơ tim hoặc giảm dung nạp gắng sức.",
				recommendation="Nếu khó thở xuất hiện thường xuyên hoặc tăng dần, nên khám tim mạch để làm rõ nguyên nhân.",
				check=lambda data: data["shortness_of_breath"] == 1,
			),
			# Luật 18: Phù chân
			ExpertRule(
				key="leg_swelling",
				score=get_rule_weight("leg_swelling"),
				title="Phù chân",
				message="Phù chân có thể gợi ý ứ dịch ngoại biên liên quan đến chức năng tim hoặc tuần hoàn.",
				recommendation="Cần đánh giá thêm chức năng tim, thận và tình trạng giữ nước nếu phù chân kéo dài.",
				check=lambda data: data["leg_swelling"] == 1,
			),
			# Luật 19: Hồi hộp
			ExpertRule(
				key="palpitations",
				score=get_rule_weight("palpitations"),
				title="Triệu chứng hồi hộp",
				message="Hồi hộp có thể liên quan đến rối loạn nhịp tim hoặc đáp ứng tim mạch bất thường.",
				recommendation="Nên theo dõi điện tim hoặc Holter nếu hồi hộp xuất hiện lặp lại hoặc kèm khó chịu ngực.",
				check=lambda data: data["palpitations"] == 1,
			),
			# Luật 20: Đau thắt ngực khi gắng sức
			ExpertRule(
				key="exercise_angina",
				score=get_rule_weight("exercise_angina"),
				title="Đau thắt ngực khi gắng sức",
				message="Đau thắt ngực khi gắng sức là dấu hiệu mạnh của tình trạng thiếu máu cơ tim.",
				recommendation="Nên được thăm khám sớm và hạn chế gắng sức mạnh trước khi có đánh giá chuyên khoa.",
				check=lambda data: data["exercise_angina"] == 1,
			),
			# Luật 21: ST chênh xuống sau gắng sức
			ExpertRule(
				key="st_depression",
				score=get_rule_weight("st_depression"),
				title="ST chênh xuống sau gắng sức",
				message="Độ lệch ST tăng là dấu hiệu liên quan đến nguy cơ thiếu máu cơ tim.",
				recommendation="Cần theo dõi bằng ECG gắng sức hoặc các xét nghiệm hình ảnh chức năng tim.",
				check=lambda data: data["oldpeak"] >= 1.5,
			),
			# Luật 22: Độ dốc ST không thuận lợi
			ExpertRule(
				key="flat_or_downslope",
				score=get_rule_weight("flat_or_downslope"),
				title="Độ dốc ST không thuận lợi",
				message="Độ dốc ST bằng hoặc giảm sau gắng sức thường liên quan đến nguy cơ bệnh mạch vành cao hơn.",
				recommendation="Cần tổng hợp với ECG và triệu chứng để quyết định hướng can thiệp tiếp theo.",
				check=lambda data: data["slope"] in {1, 2},
			),
			# Luật 23: Tổn thương mạch lớn
			ExpertRule(
				key="major_vessels_visible",
				score=get_rule_weight("major_vessels_visible"),
				title="Tổn thương mạch lớn",
				message="Số mạch lớn bị ảnh hưởng trên phim chụp càng cao thì nguy cơ càng tăng.",
				recommendation="Cần đánh giá sâu hơn bởi bác sĩ tim mạch, có thể gồm siêu âm tim hoặc chụp mạch.",
				check=lambda data: data["num_major_vessels"] >= 1,
			),
			# Luật 24: Kết quả thal bất thường
			ExpertRule(
				key="thal_defect",
				score=get_rule_weight("thal_defect"),
				title="Kết quả thal bất thường",
				message="Kết quả thal bất thường thường đi kèm nguy cơ tổn thương mạch vành.",
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

		if data["congenital_heart_disease"] == 1:
			total_score = self.max_score
			normalized_score = 100.0
			risk_level = "high"
		else:
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

