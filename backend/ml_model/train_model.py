from __future__ import annotations
import pickle
from pathlib import Path

# pandas dùng để đọc và xử lý dữ liệu CSV
import pandas as pd

# RandomForestClassifier là mô hình Machine Learning dùng để phân loại
from sklearn.ensemble import RandomForestClassifier

# accuracy_score dùng để tính độ chính xác
# classification_report dùng để in precision, recall, f1-score
from sklearn.metrics import accuracy_score, classification_report

# train_test_split dùng để chia dữ liệu thành tập train và test
from sklearn.model_selection import train_test_split


# Lấy đường dẫn thư mục hiện tại của file Python này
BASE_DIR = Path(__file__).resolve().parent

# Đường dẫn đến file dữ liệu huấn luyện
DATA_PATH = BASE_DIR / "data" / "heart_disease_cleveland.csv"

# Đường dẫn lưu model sau khi train xong
MODEL_PATH = BASE_DIR / "random_forest.pkl"


# Danh sách các cột đặc trưng dùng để huấn luyện model
# Đây là input X của mô hình
FEATURE_COLUMNS = [
    "age",                       # Tuổi
    "sex",                       # Giới tính
    "congenital_heart_disease",  # Bệnh tim bẩm sinh
    "chest_pain_type",           # Loại đau ngực
    "resting_bp",                # Huyết áp lúc nghỉ
    "cholesterol",               # Cholesterol
    "fasting_blood_sugar",       # Đường huyết khi đói
    "resting_ecg",               # Điện tâm đồ lúc nghỉ
    "max_heart_rate",            # Nhịp tim tối đa
    "exercise_angina",           # Đau thắt ngực khi vận động
    "oldpeak",                   # Độ chênh xuống đoạn ST
    "slope",                     # Độ dốc đoạn ST
    "num_major_vessels",         # Số mạch máu chính bị tổn thương
    "thal",                      # Kết quả thalassemia
]


# Nếu dataset không có cột congenital_heart_disease
# thì tự thêm cột này với giá trị mặc định là 0
DEFAULT_FEATURE_VALUES = {
    "congenital_heart_disease": 0,
}


# Các tên cột có thể là target trong dataset
# Một số dataset dùng target, một số dùng num hoặc heart_disease
TARGET_CANDIDATES = ["target", "num", "heart_disease"]


# Đổi tên cột gốc trong Cleveland dataset sang tên dễ hiểu hơn
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


# Chuẩn hóa giá trị category
# Dataset gốc Cleveland thường dùng:
# cp: 1, 2, 3, 4
# slope: 1, 2, 3
# thal: 3, 6, 7
#
# Ở đây ta đổi về dạng bắt đầu từ 0 để model dễ xử lý hơn
CATEGORY_MAPPINGS = {
    "chest_pain_type": {1: 0, 2: 1, 3: 2, 4: 3},
    "slope": {1: 0, 2: 1, 3: 2},
    "thal": {3: 0, 6: 1, 7: 2},
}


def load_dataset() -> pd.DataFrame:
    """
    Hàm này dùng để:
    1. Đọc file CSV
    2. Đổi tên cột
    3. Chuẩn hóa dữ liệu
    4. Thêm cột mặc định nếu thiếu
    5. Tìm cột target
    6. Loại bỏ dữ liệu thiếu
    7. Trả về dataframe đã sẵn sàng để train model
    """

    # Kiểm tra file dữ liệu có tồn tại không
    # Nếu file không tồn tại hoặc file rỗng thì báo lỗi
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Khong tim thay du lieu huan luyen hop le tai {DATA_PATH}."
        )

    # Đọc file CSV
    dataframe = pd.read_csv(DATA_PATH)

    # Đổi tên cột từ tên gốc sang tên dễ hiểu hơn
    dataframe = dataframe.rename(columns=RENAMES)

    # Chuẩn hóa kiểu dữ liệu và category
    dataframe = normalize_dataset(dataframe)

    # Nếu thiếu các cột mặc định, ví dụ congenital_heart_disease,
    # thì thêm vào với giá trị mặc định là 0
    for column, default_value in DEFAULT_FEATURE_VALUES.items():
        if column not in dataframe.columns:
            dataframe[column] = default_value

    # Tìm cột target trong dataset
    # Ưu tiên theo thứ tự: target, num, heart_disease
    target_column = next(
        (column for column in TARGET_CANDIDATES if column in dataframe.columns),
        None
    )

    # Nếu không tìm thấy cột target thì báo lỗi
    if target_column is None:
        raise ValueError(
            "Khong tim thay cot target. Hay dung mot trong cac cot: target, num, heart_disease."
        )

    # Kiểm tra xem dataset có thiếu cột đặc trưng nào không
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in dataframe.columns
    ]

    # Nếu thiếu cột đặc trưng thì báo lỗi
    if missing_columns:
        raise ValueError(f"Thieu cot dac trung: {missing_columns}")

    # Chỉ giữ lại các cột đặc trưng và cột target
    # dropna() dùng để xóa các dòng có dữ liệu thiếu
    dataframe = dataframe[FEATURE_COLUMNS + [target_column]].dropna().copy()

    # Chuyển target về dạng nhị phân:
    # 0 = không bệnh tim
    # 1 = có bệnh tim
    #
    # Trong Cleveland dataset, target có thể là:
    # 0 = không bệnh
    # 1, 2, 3, 4 = có bệnh
    dataframe[target_column] = (dataframe[target_column] > 0).astype(int)

    # Đổi tên cột target về thống nhất là "target"
    dataframe = dataframe.rename(columns={target_column: "target"})

    return dataframe


def normalize_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm này dùng để chuẩn hóa dữ liệu:
    1. Chuyển các cột đặc trưng về dạng số
    2. Chuyển các giá trị category về dạng chuẩn
    """

    # Tạo bản copy để không làm thay đổi dataframe gốc
    normalized = dataframe.copy()

    # Chuyển tất cả cột feature sang dạng số
    # Nếu có giá trị lỗi như "?" thì sẽ chuyển thành NaN
    for column in FEATURE_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(
                normalized[column],
                errors="coerce"
            )

    # Áp dụng mapping cho các biến category
    # Ví dụ cp 1,2,3,4 -> 0,1,2,3
    for column, mapping in CATEGORY_MAPPINGS.items():
        if column in normalized.columns:
            normalized[column] = normalized[column].replace(mapping)

    return normalized


def train_model() -> None:
    """
    Hàm chính dùng để huấn luyện mô hình Random Forest.
    Quy trình:
    1. Load dữ liệu
    2. Tách X và y
    3. Chia train/test
    4. Train model
    5. Đánh giá model
    6. Lưu model ra file .pkl
    """

    # Đọc và xử lý dữ liệu
    dataframe = load_dataset()

    # X là các đặc trưng đầu vào
    features = dataframe[FEATURE_COLUMNS]

    # y là nhãn cần dự đoán
    target = dataframe["target"]

    # Chia dữ liệu thành train và test
    # 80% train, 20% test
    #
    # stratify=target giúp giữ tỷ lệ có bệnh / không bệnh
    # ở train và test tương đối giống nhau
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    # Tạo mô hình Random Forest
    model = RandomForestClassifier(
        n_estimators=300,       # Số lượng cây quyết định trong rừng
        max_depth=None,           # Độ sâu tối đa của mỗi cây
        min_samples_split=4,    # Số mẫu tối thiểu để chia tiếp một node
        random_state=42,        # Giúp kết quả ổn định mỗi lần chạy
    )

    # Huấn luyện model bằng dữ liệu train
    model.fit(X_train, y_train)

    # Dự đoán trên tập test
    predictions = model.predict(X_test)

    # Tính độ chính xác
    accuracy = accuracy_score(y_test, predictions)

    # Lưu model đã train vào file random_forest.pkl
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    # In kết quả
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Accuracy: {accuracy:.4f}")

    # In báo cáo chi tiết:
    # precision, recall, f1-score, support
    print(classification_report(y_test, predictions, digits=4))


# Nếu chạy trực tiếp file này bằng lệnh:
# python train_model.py
# thì chương trình sẽ gọi hàm train_model()
if __name__ == "__main__":
    train_model()