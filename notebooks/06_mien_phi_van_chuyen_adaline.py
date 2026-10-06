import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

# In tiếng Việt không lỗi trên terminal Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ============================================================
# CẤU HÌNH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / ".." / "data" / "ecommerce_sales_34500.csv"
OUTPUT_DIR = BASE_DIR / "adaline_outputs"

FEATURE_NAMES = ["total_amount", "distance_km", "membership_code"]

MEMBERSHIP_MAP = {
    "Bronze": 0,
    "Silver": 1,
    "Gold": 2,
    "VIP": 3
}

LEARNING_RATE = 0.003
EPOCHS = 50
THRESHOLD = 0.5
TRAIN_RATIO = 0.8
SEED = 42


def print_title(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. THU THẬP / TẠO DỮ LIỆU
# ============================================================

print_title("1. THU THẬP / TẠO DỮ LIỆU")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Không tìm thấy dữ liệu: {DATA_PATH.resolve()}")

df = pd.read_csv(DATA_PATH)

df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

# Sắp xếp theo khách hàng + ngày để tính lịch sử mua hàng TRƯỚC mỗi đơn
df = df.sort_values(
    ["customer_id", "order_date", "order_id"]
).reset_index(drop=True)

df["previous_spending"] = (
    df.groupby("customer_id")["total_amount"].cumsum()
    - df["total_amount"]
)

df["previous_orders"] = df.groupby("customer_id").cumcount()


def create_membership(previous_spending, previous_orders):
    if previous_spending >= 1500 or previous_orders >= 8:
        return "VIP"
    elif previous_spending >= 800 or previous_orders >= 5:
        return "Gold"
    elif previous_spending >= 300 or previous_orders >= 2:
        return "Silver"
    else:
        return "Bronze"


df["membership_level"] = [
    create_membership(spending, orders)
    for spending, orders in zip(
        df["previous_spending"],
        df["previous_orders"]
    )
]

# Dataset gốc không có khoảng cách -> mô phỏng ngẫu nhiên 1-30 km
np.random.seed(SEED)
df["distance_km"] = np.random.uniform(1, 30, len(df))

df["membership_code"] = df["membership_level"].map(MEMBERSHIP_MAP)


def create_target(row):
    high_value = row["total_amount"] >= 100
    loyal_customer = row["membership_level"] in ["Gold", "VIP"]
    short_distance = row["distance_km"] <= 10
    reasonable_order = row["total_amount"] >= 70

    if high_value and short_distance:
        return 1

    if loyal_customer and short_distance and reasonable_order:
        return 1

    return 0


df["y"] = df.apply(create_target, axis=1)

X = df[FEATURE_NAMES].to_numpy(dtype=float)
t = df["y"].to_numpy(dtype=float)

print("Shape dữ liệu gốc:", df.shape)
print("X shape:", X.shape)
print("t shape:", t.shape)

print("\n5 dòng đầu của X:")
print(X[:5])

print("\n5 giá trị đầu của target:")
print(t[:5])

print("\nPhân bố nhãn:")
print(pd.Series(t).value_counts().sort_index())


# ============================================================
# 2. CHUẨN HÓA FEATURE
# ============================================================

print_title("2. CHUẨN HÓA FEATURE")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Trước khi chuẩn hóa:")
print(X[:5])

print("\nSau khi chuẩn hóa:")
print(X_scaled[:5])

print("\nMean:", np.round(X_scaled.mean(axis=0), 4))
print("Std :", np.round(X_scaled.std(axis=0), 4))


# ============================================================
# 3. TÁCH TRAIN / TEST
# ============================================================

print_title("3. TÁCH TRAIN / TEST")

split_index = int(len(X_scaled) * TRAIN_RATIO)

X_train = X_scaled[:split_index]
X_test = X_scaled[split_index:]

t_train = t[:split_index]
t_test = t[split_index:]

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("\nPhân bố nhãn train:")
print(pd.Series(t_train).value_counts().sort_index())

print("\nPhân bố nhãn test:")
print(pd.Series(t_test).value_counts().sort_index())


# ============================================================
# 4. HUẤN LUYỆN ADALINE (Widrow-Hoff / LMS)
# ============================================================

print_title("4. HUẤN LUYỆN ADALINE")


def train_adaline(X_tr, t_tr, X_te, t_te, learning_rate, epochs, seed=SEED):
    rng = np.random.default_rng(seed)

    w = np.zeros(X_tr.shape[1], dtype=float)
    b = 0.0

    # Ghi MSE ở epoch 0 (trước khi học, w = 0 và b = 0)
    train_hist = [np.mean((t_tr - (np.dot(X_tr, w) + b)) ** 2)]
    test_hist = [np.mean((t_te - (np.dot(X_te, w) + b)) ** 2)]

    for epoch in range(epochs):

        # Xáo trộn thứ tự đơn hàng ở mỗi epoch
        order = rng.permutation(len(X_tr))

        for i in order:
            x_i = X_tr[i]
            t_i = t_tr[i]

            net = np.dot(w, x_i) + b      # điểm số liên tục
            e = t_i - net                 # sai số

            w = w + learning_rate * e * x_i   # luật Widrow-Hoff
            b = b + learning_rate * e

        train_hist.append(np.mean((t_tr - (np.dot(X_tr, w) + b)) ** 2))
        test_hist.append(np.mean((t_te - (np.dot(X_te, w) + b)) ** 2))

    return w, b, train_hist, test_hist


w, b, train_mse_history, test_mse_history = train_adaline(
    X_train, t_train, X_test, t_test, LEARNING_RATE, EPOCHS
)

lr_limit = 2 / np.max(np.sum(X_train ** 2, axis=1) + 1)

print("Learning rate:", LEARNING_RATE)
print("Epochs:", EPOCHS)
print("Giới hạn ổn định lý thuyết của learning rate:", round(lr_limit, 5))

print("\nWeights:", w)
print("Bias   :", b)

print("\nTrain MSE epoch 0 (chưa học):", round(train_mse_history[0], 6))
print("Train MSE epoch 1            :", round(train_mse_history[1], 6))
print("Final train MSE              :", train_mse_history[-1])
print("Final test MSE               :", test_mse_history[-1])


# ============================================================
# 5. ĐƯỜNG HỌC (MSE THEO EPOCH)
# ============================================================

print_title("5. ĐƯỜNG HỌC (MSE THEO EPOCH)")

baseline_mse = np.mean((t_test - t_train.mean()) ** 2)

# ---- 5a. Đường học chính (hai khung) ----
fig_a, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(np.arange(0, EPOCHS + 1), train_mse_history, label="Train MSE")
axes[0].plot(np.arange(0, EPOCHS + 1), test_mse_history, label="Test MSE")
axes[0].axhline(
    baseline_mse,
    color="gray",
    linestyle="--",
    label="Baseline (đoán hằng số)"
)
axes[0].set_title("Toàn bộ quá trình học (từ epoch 0)")

axes[1].plot(np.arange(1, EPOCHS + 1), train_mse_history[1:], label="Train MSE")
axes[1].plot(np.arange(1, EPOCHS + 1), test_mse_history[1:], label="Test MSE")
axes[1].set_title("Phóng to (từ epoch 1)")

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)

fig_a.suptitle("Adaline Learning Curve")
fig_a.tight_layout()
fig_a.savefig(OUTPUT_DIR / "learning_curve.png", dpi=120)

print("Baseline MSE (đoán hằng số):", round(baseline_mse, 6))
print("Initial Train MSE (epoch 0):", round(train_mse_history[0], 6))
print("Final Train MSE             :", round(train_mse_history[-1], 6))
print("Final Test MSE              :", round(test_mse_history[-1], 6))

# ---- 5b. So sánh nhiều learning rate ----
learning_rates = [0.0001, 0.001, 0.003, 0.01, 0.02]

results = {LEARNING_RATE: (train_mse_history, test_mse_history)}

for lr in learning_rates:
    if lr not in results:
        _, _, tr_h, te_h = train_adaline(
            X_train, t_train, X_test, t_test, lr, EPOCHS
        )
        results[lr] = (tr_h, te_h)

fig_b, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr, (tr_h, te_h) in sorted(results.items()):
    axes[0].plot(np.arange(EPOCHS + 1), tr_h, label=f"lr = {lr}")
    axes[1].plot(np.arange(EPOCHS + 1), te_h, label=f"lr = {lr}")

axes[0].set_title("Train MSE theo learning rate")
axes[1].set_title("Test MSE theo learning rate")

for ax in axes:
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True)

fig_b.tight_layout()
fig_b.savefig(OUTPUT_DIR / "learning_rate_comparison.png", dpi=120)

final_level = {lr: np.mean(h[0][-5:]) for lr, h in results.items()}
threshold_mse = min(final_level.values()) * 1.05

rows = []

for lr, (tr_h, te_h) in sorted(results.items()):
    tr_arr = np.array(tr_h)

    converged = [
        e for e in range(len(tr_arr))
        if np.all(tr_arr[e:] <= threshold_mse)
    ]

    rows.append({
        "learning_rate": lr,
        "MSE epoch 0": round(tr_arr[0], 5),
        "MSE epoch 1": round(tr_arr[1], 5),
        "MSE cuối": round(tr_arr[-1], 5),
        "dao động 10 epoch cuối": round(tr_arr[-10:].max() - tr_arr[-10:].min(), 5),
        "epoch hội tụ": converged[0] if converged else "chưa ổn định"
    })

print("\nSo sánh learning rate:")
print(pd.DataFrame(rows).to_string(index=False))


# ---- 5c. Vì sao lr = 0.001 cho đường phẳng: nhìn vào bên trong epoch 1 ----
def first_epoch_curve(learning_rate, n_steps=6000, every=100, seed=SEED):
    rng = np.random.default_rng(seed)

    w_ = np.zeros(X_train.shape[1], dtype=float)
    b_ = 0.0

    order = rng.permutation(len(X_train))[:n_steps]

    steps = [0]
    mses = [np.mean((t_train - (np.dot(X_train, w_) + b_)) ** 2)]

    for k, i in enumerate(order, start=1):
        x_i = X_train[i]
        t_i = t_train[i]

        e = t_i - (np.dot(w_, x_i) + b_)

        w_ = w_ + learning_rate * e * x_i
        b_ = b_ + learning_rate * e

        if k % every == 0:
            steps.append(k)
            mses.append(np.mean((t_train - (np.dot(X_train, w_) + b_)) ** 2))

    return steps, mses


steps, mses = first_epoch_curve(0.001)

fig_c = plt.figure(figsize=(10, 5))
plt.plot(steps, mses)
plt.xlabel("Số mẫu đã học (trong epoch 1)")
plt.ylabel("Train MSE")
plt.title("lr = 0.001: mô hình học xong chỉ sau vài nghìn mẫu đầu")
plt.grid(True)
fig_c.savefig(OUTPUT_DIR / "first_epoch_curve.png", dpi=120)

print("\nlr = 0.001 - MSE trước khi học:", round(mses[0], 5))
print("lr = 0.001 - MSE sau 6000 mẫu :", round(mses[-1], 5))


# ============================================================
# 6. ĐÁNH GIÁ VÀ PHÂN TÍCH
# ============================================================

print_title("6. ĐÁNH GIÁ VÀ PHÂN TÍCH")

test_output = np.dot(X_test, w) + b
test_error = t_test - test_output

analysis_df = df.iloc[split_index:].copy()
analysis_df["actual"] = t_test
analysis_df["prediction_continuous"] = test_output
analysis_df["error"] = test_error
analysis_df["absolute_error"] = np.abs(test_error)

largest_errors = (
    analysis_df
    .sort_values("absolute_error", ascending=False)
    .head(5)
)

print("Trọng số (đã chuẩn hóa nên so sánh được độ lớn):")
for feature, weight in zip(FEATURE_NAMES, w):
    print(f"  {feature}: {weight:.6f}")

print(f"\nBias: {b:.6f}")

print("\n5 mẫu có sai số lớn nhất:")
print(
    largest_errors[
        [
            "order_id",
            "total_amount",
            "distance_km",
            "membership_level",
            "actual",
            "prediction_continuous",
            "error"
        ]
    ].to_string(index=False)
)


# ============================================================
# 7. CHỌN NGƯỠNG VÀ THỬ NGHIỆM
# ============================================================

print_title("7. CHỌN NGƯỠNG VÀ THỬ NGHIỆM")

t_test_int = t_test.astype(int)

y_pred = (test_output >= THRESHOLD).astype(int)

accuracy = np.mean(y_pred == t_test_int)

cm = np.zeros((2, 2), dtype=int)

for actual, predicted in zip(t_test_int, y_pred):
    cm[actual, predicted] += 1

tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

majority_class = int(t_train.mean() >= 0.5)
baseline_accuracy = np.mean(t_test_int == majority_class)

print("Threshold:", THRESHOLD)
print("Accuracy:", round(accuracy, 4))
print("Baseline accuracy (luôn đoán Voucher):", round(baseline_accuracy, 4))
print("Recall (Free Shipping):", round(recall, 4))

print("\nConfusion matrix (hàng = thực tế, cột = dự đoán):")
print(cm)
print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")

# Thử các ngưỡng khác nhau
sweep_rows = []

for th in [0.5, 0.4, 0.3, 0.25, 0.2]:
    yp = (test_output >= th).astype(int)

    tp_ = int(np.sum((yp == 1) & (t_test_int == 1)))
    fp_ = int(np.sum((yp == 1) & (t_test_int == 0)))
    fn_ = int(np.sum((yp == 0) & (t_test_int == 1)))

    sweep_rows.append({
        "threshold": th,
        "accuracy": np.mean(yp == t_test_int),
        "precision": tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0,
        "recall": tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0,
        "TP": tp_,
        "FP": fp_,
        "FN": fn_
    })

print("\nẢnh hưởng của ngưỡng:")
print(pd.DataFrame(sweep_rows).round(4).to_string(index=False))


def predict_one(total_amount, distance_km, membership_level):

    if membership_level not in MEMBERSHIP_MAP:
        raise ValueError("Invalid membership level")

    if total_amount < 0:
        raise ValueError("total_amount must be >= 0")

    if distance_km <= 0 or distance_km > 30:
        raise ValueError("distance_km must be in the range (0, 30]")

    sample = np.array(
        [[total_amount, distance_km, MEMBERSHIP_MAP[membership_level]]],
        dtype=float
    )

    sample_scaled = scaler.transform(sample)

    net = float(np.dot(w, sample_scaled[0]) + b)

    prediction = int(net >= THRESHOLD)

    return {
        "net": net,
        "prediction": prediction,
        "decision": "Free Shipping" if prediction == 1 else "Voucher"
    }


samples = [
    (200, 6, "Gold"),
    (60, 25, "Bronze"),
    (80, 8, "VIP")
]

print("\nDự đoán đơn hàng mới:")

for total_amount, distance_km, membership_level in samples:
    result = predict_one(total_amount, distance_km, membership_level)

    print(
        f"  {total_amount} $, {distance_km} km, {membership_level:<6} "
        f"-> net = {result['net']:.4f} -> {result['decision']}"
    )

# Tính tay net cho một đơn hàng (nhiệm vụ 3 của đề)
print("\nTính tay net cho đơn (200 $, 6 km, Gold):")

raw = np.array([200.0, 6.0, MEMBERSHIP_MAP["Gold"]], dtype=float)
z = (raw - scaler.mean_) / scaler.scale_

for name, raw_i, mean_i, std_i, z_i, w_i in zip(
    FEATURE_NAMES, raw, scaler.mean_, scaler.scale_, z, w
):
    print(
        f"  {name}: z = ({raw_i:g} - {mean_i:.3f}) / {std_i:.3f} = {z_i:.4f}"
        f"  ->  w * z = {w_i:.5f} * {z_i:.4f} = {w_i * z_i:.5f}"
    )

net_by_hand = float(np.dot(w, z) + b)
print(f"  net = tổng(w * z) + b = {net_by_hand:.5f}")
print(f"  (predict_one cho kết quả: {predict_one(200, 6, 'Gold')['net']:.5f})")


# ============================================================
# 8. KẾT LUẬN
# ============================================================

print_title("8. KẾT LUẬN")

print(
    "1. Adaline cho điểm số liên tục (học bằng MSE, luật Widrow-Hoff), nên có thể\n"
    "   hiểu điểm là mức giảm phí vận chuyển; Perceptron chỉ cho 0/1."
)

print(
    f"2. Train MSE = {train_mse_history[-1]:.4f}, Test MSE = {test_mse_history[-1]:.4f}, "
    f"baseline (đoán hằng số) = {baseline_mse:.4f}."
)

if test_mse_history[-1] < baseline_mse:
    print("   Mô hình có học được so với việc không học gì.")

print("3. Dấu của trọng số:")
for feature, weight in zip(FEATURE_NAMES, w):
    direction = "tăng" if weight > 0 else "giảm"
    print(f"   - {feature}: {weight:+.4f} (càng lớn thì khả năng miễn phí {direction})")

print(
    f"4. Ở ngưỡng {THRESHOLD}: accuracy = {accuracy:.4f} "
    f"(baseline {baseline_accuracy:.4f}), recall = {recall:.4f}."
)

if recall < 0.5:
    print(
        "   Recall thấp: mô hình bỏ sót phần lớn đơn đáng được Free Shipping, "
        "accuracy cao chủ yếu do dữ liệu lệch về lớp Voucher."
    )

print(
    "5. Giới hạn: mô hình tuyến tính khó bắt luật 'đơn đủ lớn VÀ giao đủ gần' (phi tuyến);\n"
    "   total_amount có ngoại lai lớn nên learning rate cao làm MSE dao động;\n"
    "   khoảng cách và nhãn là dữ liệu mô phỏng nên kết quả chỉ mang tính minh họa học thuật.\n"
    "   Hướng tiếp theo: dùng target liên tục (mức giảm phí), hạ ngưỡng, hoặc thử MLP."
)

print(f"\nBiểu đồ đã lưu tại: {OUTPUT_DIR.resolve()}")

# Hiện tất cả biểu đồ cùng lúc (đóng cửa sổ biểu đồ để kết thúc chương trình)
plt.show()