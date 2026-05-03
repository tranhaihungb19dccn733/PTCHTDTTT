const API_BASE_URL = "http://127.0.0.1:8000";

const fallbackOptions = {
	sex: [
		{ value: 1, label: "Nam" },
		{ value: 0, label: "Nữ" },
	],
	smoking: [
		{ value: 0, label: "Không hút thuốc" },
		{ value: 1, label: "Đang hút thuốc" },
	],
	family_history: [
		{ value: 0, label: "Không có" },
		{ value: 1, label: "Có" },
	],
	congenital_heart_disease: [
		{ value: 0, label: "Không" },
		{ value: 1, label: "Có" },
	],
	diagnosed_diabetes: [
		{ value: 0, label: "Không" },
		{ value: 1, label: "Có" },
	],
	physical_activity_level: [
		{ value: 0, label: "Thấp" },
		{ value: 1, label: "Trung bình" },
		{ value: 2, label: "Cao" },
	],
	chest_pain_type: [
		{ value: 0, label: "Đau thắt ngực điển hình" },
		{ value: 1, label: "Đau thắt ngực không điển hình" },
		{ value: 2, label: "Đau không do thắt ngực" },
		{ value: 3, label: "Không có triệu chứng" },
	],
	fasting_blood_sugar: [
		{ value: 0, label: "< 120 mg/dl" },
		{ value: 1, label: "> 120 mg/dl" },
	],
	resting_ecg: [
		{ value: 0, label: "Bình thường" },
		{ value: 1, label: "ST-T bất thường" },
		{ value: 2, label: "Phì đại thất trái" },
	],
	exercise_angina: [
		{ value: 0, label: "Không" },
		{ value: 1, label: "Có" },
	],
	shortness_of_breath: [
		{ value: 0, label: "Không" },
		{ value: 1, label: "Có" },
	],
	leg_swelling: [
		{ value: 0, label: "Không" },
		{ value: 1, label: "Có" },
	],
	palpitations: [
		{ value: 0, label: "Không" },
		{ value: 1, label: "Có" },
	],
	slope: [
		{ value: 0, label: "Dốc lên" },
		{ value: 1, label: "Ngang" },
		{ value: 2, label: "Dốc xuống" },
	],
	thal: [
		{ value: 0, label: "Bình thường" },
		{ value: 1, label: "Khiếm khuyết cố định" },
		{ value: 2, label: "Khiếm khuyết hồi phục" },
	],
};

const form = document.querySelector("#prediction-form");
const checkApiButton = document.querySelector("#check-api");
const resetFormButton = document.querySelector("#reset-form");
const statusElement = document.querySelector("#api-status");
const emptyState = document.querySelector("#empty-state");
const resultPanel = document.querySelector("#result-panel");
const riskLabel = document.querySelector("#risk-label");
const resultSummary = document.querySelector("#result-summary");
const riskPercent = document.querySelector("#risk-percent");
const expertScore = document.querySelector("#expert-score");
const mlScore = document.querySelector("#ml-score");
const mlMessage = document.querySelector("#ml-message");
const mlAccuracy = document.querySelector("#ml-accuracy");
const mlAccuracyNote = document.querySelector("#ml-accuracy-note");
const progressBar = document.querySelector("#progress-bar");
const ruleList = document.querySelector("#rule-list");
const recommendationList = document.querySelector("#recommendation-list");
const scoreRing = document.querySelector(".score-ring");
const LAST_PREDICTION_KEY = "heart-disease:last-prediction";
const LAST_FORM_DATA_KEY = "heart-disease:last-form-data";


document.addEventListener("DOMContentLoaded", async () => {
	console.log("DOM loaded, form element:", form);
	console.log("Sex select element:", form.querySelector('select[name="sex"]'));
	const options = await loadOptions();
	console.log("Options loaded:", options);
	populateSelects(options);

	// Verify population worked
	const sexSelect = form.querySelector('select[name="sex"]');
	console.log("Sex select after populate:", sexSelect);
	console.log("Sex select children count:", sexSelect?.children?.length);

	restorePreviousState();
});

form.addEventListener("submit", async (event) => {
	event.preventDefault();

	const payload = readFormData();
	setStatus("Đang phân tích nguy cơ...", "pending");

	try {
		const response = await fetch(`${API_BASE_URL}/api/predict`, {
			method: "POST",
			headers: {
				"Content-Type": "application/json",
			},
			body: JSON.stringify(payload),
		});

		if (!response.ok) {
			let details = "";
			try {
				const errorBody = await response.json();
				details = errorBody.detail ? `: ${errorBody.detail}` : "";
			} catch {
				details = "";
			}

			throw new Error(`API dự đoán trả lỗi ${response.status}${details}`);
		}

		const data = await response.json();
		persistPrediction(payload, data);
		renderPrediction(data);
		setStatus("API sẵn sàng và đã trả kết quả.", "ok");
	} catch (error) {
		if (error instanceof TypeError) {
			setStatus("Không thể kết nối đến API dự đoán. Vui lòng kiểm tra backend đang chạy.", "error");
			return;
		}

		setStatus(error.message, "error");
	}
});

checkApiButton.addEventListener("click", async () => {
	setStatus("Đang kiểm tra kết nối API...", "pending");
	await updateApiStatus();
});

resetFormButton.addEventListener("click", () => {
	const shouldReset = window.confirm("Bạn có chắc muốn xóa dữ liệu đang nhập và kết quả phân tích hiện tại không?");
	if (!shouldReset) {
		return;
	}

	form.reset();
	clearStoredPrediction();
	resetPredictionView();
	setStatus("Biểu mẫu đã được làm mới để nhập ca mới.", "ok");
	form.querySelector("input, select")?.focus();
});


async function loadOptions() {
	try {
		const response = await fetch(`${API_BASE_URL}/api/options`);
		if (!response.ok) {
			throw new Error();
		}

		return await response.json();
	} catch {
		return fallbackOptions;
	}
}


function populateSelects(options) {
	console.log("populateSelects called with options:", options);
	Object.entries(options).forEach(([name, values]) => {
		const select = form.querySelector(`select[name="${name}"]`);
		if (!select) {
			console.warn(`Select for ${name} not found`);
			return;
		}

		const html = values
			.map((item) => `<option value="${item.value}">${item.label}</option>`)
			.join("");
		console.log(`Populating ${name} with ${values.length} options`);
		select.innerHTML = html;

		if (select.children.length === 0) {
			console.error(`Select ${name} is still empty after populate! Adding fallback.`);
			select.innerHTML = '<option value="">-- Không có dữ liệu --</option>';
		}
	});
}


async function updateApiStatus() {
	try {
		const response = await fetch(`${API_BASE_URL}/health`);
		if (!response.ok) {
			throw new Error();
		}

		const data = await response.json();
		setStatus(
			data.ml_model_loaded
				? "API đang hoạt động, mô hình ML đã sẵn sàng."
				: `API đang hoạt động, ${data.ml_message}`,
			data.ml_model_loaded ? "ok" : "pending"
		);
	} catch {
		setStatus("Không kết nối được backend. Hãy chạy FastAPI tại cổng 8000.", "error");
	}
}


function readFormData() {
	const formData = new FormData(form);
	const floatFields = new Set(["oldpeak", "weight", "height"]);

	return Object.fromEntries(
		Array.from(formData.entries()).map(([key, value]) => [
			key,
			floatFields.has(key) ? Number.parseFloat(value) : Number.parseInt(value, 10),
		])
	);
}


function fillForm(values) {
	Object.entries(values).forEach(([key, value]) => {
		const field = form.elements.namedItem(key);
		if (field) {
			field.value = String(value);
		}
	});
}


function persistPrediction(payload, prediction) {
	localStorage.setItem(LAST_FORM_DATA_KEY, JSON.stringify(payload));
	localStorage.setItem(LAST_PREDICTION_KEY, JSON.stringify(prediction));
}


function clearStoredPrediction() {
	localStorage.removeItem(LAST_FORM_DATA_KEY);
	localStorage.removeItem(LAST_PREDICTION_KEY);
}


function restorePreviousState() {
	try {
		const savedFormData = localStorage.getItem(LAST_FORM_DATA_KEY);
		const savedPrediction = localStorage.getItem(LAST_PREDICTION_KEY);

		if (!savedPrediction) {
			return;
		}

		if (savedFormData) {
			fillForm(JSON.parse(savedFormData));
		}

		renderPrediction(JSON.parse(savedPrediction));
		setStatus("Đã khôi phục kết quả phân tích gần nhất.", "ok");
	} catch {
		clearStoredPrediction();
	}
}


function resetPredictionView() {
	emptyState.classList.remove("hidden");
	resultPanel.classList.add("hidden");
	riskLabel.textContent = "-";
	resultSummary.innerHTML = "";
	riskPercent.textContent = "-";
	expertScore.textContent = "0";
	mlScore.textContent = "Không có";
	mlMessage.textContent = "Chưa sử dụng mô hình";
	mlAccuracy.textContent = "Chưa có";
	mlAccuracyNote.textContent = "So với dữ liệu hiện có";
	progressBar.style.width = "0%";
	scoreRing.style.background = "conic-gradient(#d95d39 0deg, rgba(217, 93, 57, 0.14) 0deg)";
	ruleList.innerHTML = "";
	recommendationList.innerHTML = "";
}

function getRiskDisplayLabel(percent) {
	if (percent < 30) {
		return "Nguy cơ thấp";
	}

	if (percent < 70) {
		return "Nguy cơ trung bình";
	}

	return "Nguy cơ cao";
}

function renderPrediction(data) {
	emptyState.classList.add("hidden");
	resultPanel.classList.remove("hidden");

	const { final_assessment: finalAssessment, expert_system: expertSystem, machine_learning: machineLearning } = data;
	const percent = finalAssessment.risk_percent;
	const summaryLines = buildSummaryLines(expertSystem.summary, finalAssessment.next_step);
	const referenceMetrics = machineLearning.reference_metrics;
	const caseConfidence = finalAssessment.confidence;

	riskLabel.textContent = finalAssessment.risk_label;
	resultSummary.innerHTML = summaryLines
		.map((line) => `<li>${line}</li>`)
		.join("");
	riskPercent.textContent = getRiskDisplayLabel(percent);
	expertScore.textContent = `${expertSystem.normalized_score}%`;
	mlScore.textContent = machineLearning.probability !== null ? `${machineLearning.probability}%` : "Không có";
	mlMessage.textContent = machineLearning.message || "Không có thông tin từ mô hình.";
	mlAccuracy.textContent = caseConfidence?.percent !== null && caseConfidence?.percent !== undefined
		? `${caseConfidence.percent}%`
		: "Chưa có";
	mlAccuracyNote.textContent = buildConfidenceNote(caseConfidence, referenceMetrics);
	progressBar.style.width = `${percent}%`;
	scoreRing.style.background = `conic-gradient(#d95d39 ${percent * 3.6}deg, rgba(217, 93, 57, 0.14) 0deg)`;

	ruleList.innerHTML = expertSystem.rule_hits.length
		? expertSystem.rule_hits
			.map((rule) => `<li><strong>${rule.title}:</strong> ${rule.message} (+${rule.score} điểm)</li>`)
			.join("")
		: "<li>Chưa phát hiện luật nguy cơ lớn nào được kích hoạt.</li>";

	recommendationList.innerHTML = expertSystem.recommendations
		.map((item) => `<li>${item}</li>`)
		.join("");
}



function buildConfidenceNote(caseConfidence, referenceMetrics) {
	if (caseConfidence?.message) {
		if (referenceMetrics?.accuracy_percent !== null && referenceMetrics?.accuracy_percent !== undefined) {
			return `${caseConfidence.message} Tham chiếu mô hình hiện tại: ${referenceMetrics.accuracy_percent}% trên dữ liệu đã gắn nhãn.`;
		}

		return caseConfidence.message;
	}

	if (referenceMetrics?.available) {
		return referenceMetrics.message;
	}

	return referenceMetrics?.message || "Độ tin cậy sẽ được cập nhật khi có đủ dữ liệu đối chiếu.";
}


function buildSummaryLines(summary, nextStep) {
	const normalizedSummary = summary
		.replace(/\s+/g, " ")
		.split(".")
		.map((item) => item.trim())
		.filter(Boolean);

	return [...normalizedSummary, nextStep].filter(Boolean);
}


function setStatus(message, state) {
	statusElement.classList.remove("hidden");
	statusElement.textContent = message;
	if (state === "ok") {
		statusElement.style.background = "rgba(28, 124, 125, 0.12)";
		statusElement.style.color = "#1c7c7d";
		return;
	}

	if (state === "error") {
		statusElement.style.background = "rgba(150, 47, 21, 0.12)";
		statusElement.style.color = "#962f15";
		return;
	}

	statusElement.style.background = "rgba(217, 93, 57, 0.12)";
	statusElement.style.color = "#962f15";
}
