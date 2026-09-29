import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ============================================================
# 1. DATA COLLECTION / CREATION
# ============================================================

DATA_PATH = Path("../../data/ecommerce_sales_34500.csv")
MODEL_PATH = Path("../../data/adaline_free_shipping.pkl")

FEATURE_NAMES = [
    "total_amount",
    "distance_km",
    "membership_code"
]

MEMBERSHIP_MAP = {
    "Bronze": 0,
    "Silver": 1,
    "Gold": 2,
    "VIP": 3
}

LABEL_MAP = {
    0: "Voucher",
    1: "Free Shipping"
}

LEARNING_RATE = 0.001
EPOCHS = 50


if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("DATASET")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

df = df.sort_values(
    ["customer_id", "order_date", "order_id"]
).reset_index(drop=True)


df["previous_spending"] = (
    df.groupby("customer_id")["total_amount"]
    .cumsum()
    - df["total_amount"]
)

df["previous_orders"] = (
    df.groupby("customer_id")
    .cumcount()
)


def create_membership(
    previous_spending,
    previous_orders
):
    if previous_spending >= 1500 or previous_orders >= 8:
        return "VIP"

    elif previous_spending >= 800 or previous_orders >= 5:
        return "Gold"

    elif previous_spending >= 300 or previous_orders >= 2:
        return "Silver"

    return "Bronze"


df["membership_level"] = [
    create_membership(
        spending,
        orders
    )
    for spending, orders in zip(
        df["previous_spending"],
        df["previous_orders"]
    )
]


np.random.seed(42)

df["distance_km"] = np.random.uniform(
    1,
    30,
    size=len(df)
)


def create_target(row):

    high_value = (
        row["total_amount"] >= 100
    )

    loyal_customer = (
        row["membership_level"]
        in ["Gold", "VIP"]
    )

    short_distance = (
        row["distance_km"] <= 10
    )

    reasonable_order = (
        row["total_amount"] >= 70
    )

    if high_value and short_distance:
        return 1

    elif (
        loyal_customer
        and short_distance
        and reasonable_order
    ):
        return 1

    return 0


df["y"] = df.apply(
    create_target,
    axis=1
)

df["membership_code"] = (
    df["membership_level"]
    .map(MEMBERSHIP_MAP)
)


print("\nCreated data:")

print(
    df[
        [
            "total_amount",
            "distance_km",
            "membership_level",
            "membership_code",
            "y"
        ]
    ].head(10)
)

print("\nTarget distribution:")

print(
    df["y"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 2. FEATURE NORMALIZATION
# ============================================================

df_model = (
    df
    .sort_values(
        ["order_date", "order_id"]
    )
    .reset_index(drop=True)
)

X = df_model[
    FEATURE_NAMES
].copy()

y = df_model[
    "y"
].astype(int).copy()


print("=" * 70)
print("FEATURE CHECK")
print("=" * 70)

print(
    X.describe()
)

print("\nMissing values:")

print(
    X.isna().sum()
)

print(
    "\nInfinite values:",
    np.isinf(
        X.to_numpy()
    ).sum()
)


# ============================================================
# 3. TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(df_model) * 0.8
)

X_train = X.iloc[
    :split_index
].copy()

X_test = X.iloc[
    split_index:
].copy()

y_train = y.iloc[
    :split_index
].copy()

y_test = y.iloc[
    split_index:
].copy()


train_medians = X_train.median()

X_train = X_train.fillna(
    train_medians
)

X_test = X_test.fillna(
    train_medians
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


X_train_scaled = np.nan_to_num(
    X_train_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

X_test_scaled = np.nan_to_num(
    X_test_scaled,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)


print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    "Train samples:",
    len(X_train)
)

print(
    "Test samples:",
    len(X_test)
)

print(
    "\nTrain date:",
    df_model["order_date"].iloc[0],
    "to",
    df_model["order_date"].iloc[
        split_index - 1
    ]
)

print(
    "Test date:",
    df_model["order_date"].iloc[
        split_index
    ],
    "to",
    df_model["order_date"].iloc[-1]
)

print("\nTrain target distribution:")

print(
    y_train.value_counts()
    .sort_index()
)

print("\nTest target distribution:")

print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 4. ADALINE TRAINING
# ============================================================

X_train_np = X_train_scaled

X_test_np = X_test_scaled

y_train_np = y_train.to_numpy(
    dtype=float
)

y_test_np = y_test.to_numpy(
    dtype=int
)


w = np.zeros(
    X_train_np.shape[1],
    dtype=float
)

b = 0.0

train_mse_history = []

test_mse_history = []


for epoch in range(EPOCHS):

    squared_errors = []

    for i in range(
        len(X_train_np)
    ):

        x_i = X_train_np[i]

        t_i = y_train_np[i]

        net = np.dot(
            w,
            x_i
        ) + b

        error = t_i - net

        w = (
            w
            + LEARNING_RATE
            * error
            * x_i
        )

        b = (
            b
            + LEARNING_RATE
            * error
        )

        squared_errors.append(
            error ** 2
        )


    train_output = (
        np.dot(
            X_train_np,
            w
        ) + b
    )

    test_output = (
        np.dot(
            X_test_np,
            w
        ) + b
    )


    train_mse = np.mean(
        (
            y_train_np
            - train_output
        ) ** 2
    )

    test_mse = np.mean(
        (
            y_test_np
            - test_output
        ) ** 2
    )


    train_mse_history.append(
        train_mse
    )

    test_mse_history.append(
        test_mse
    )


print("=" * 70)
print("ADALINE TRAINING")
print("=" * 70)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Weights:",
    w
)

print(
    "Bias:",
    b
)

print(
    "Final train MSE:",
    train_mse_history[-1]
)

print(
    "Final test MSE:",
    test_mse_history[-1]
)


# ============================================================
# 5. LEARNING CURVE
# ============================================================

epochs_range = np.arange(
    1,
    EPOCHS + 1
)


plt.figure(
    figsize=(10, 5)
)

plt.plot(
    epochs_range,
    train_mse_history,
    label="Train MSE"
)

plt.plot(
    epochs_range,
    test_mse_history,
    label="Test MSE"
)

plt.xlabel("Epoch")

plt.ylabel("MSE")

plt.title(
    "Adaline Learning Curve"
)

plt.legend()

plt.grid(True)

plt.show()


print(
    "Initial train MSE:",
    round(
        train_mse_history[0],
        6
    )
)

print(
    "Final train MSE:",
    round(
        train_mse_history[-1],
        6
    )
)

print(
    "Initial test MSE:",
    round(
        test_mse_history[0],
        6
    )
)

print(
    "Final test MSE:",
    round(
        test_mse_history[-1],
        6
    )
)


# ============================================================
# 6. EVALUATION AND ANALYSIS
# ============================================================

test_continuous = (
    np.dot(
        X_test_np,
        w
    ) + b
)

test_error = (
    y_test_np
    - test_continuous
)

absolute_error = np.abs(
    test_error
)


analysis_df = (
    df_model
    .iloc[split_index:]
    .copy()
)

analysis_df["actual"] = y_test_np

analysis_df["continuous_output"] = (
    test_continuous
)

analysis_df["error"] = test_error

analysis_df["absolute_error"] = (
    absolute_error
)


largest_errors = (
    analysis_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(5)
)


print("=" * 70)
print("ADALINE TEST ANALYSIS")
print("=" * 70)

print("\nWeights:")

for feature, weight in zip(
    FEATURE_NAMES,
    w
):
    print(
        f"{feature}: {weight:.6f}"
    )

print(
    "\nBias:",
    round(
        b,
        6
    )
)

print(
    "\nLargest prediction errors:"
)

print(
    largest_errors[
        [
            "order_id",
            "total_amount",
            "distance_km",
            "membership_level",
            "actual",
            "continuous_output",
            "error",
            "absolute_error"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 7. THRESHOLD AND EXPERIMENT
# ============================================================

THRESHOLD = 0.5


y_pred = (
    test_continuous >= THRESHOLD
).astype(int)


accuracy = accuracy_score(
    y_test_np,
    y_pred
)


cm = confusion_matrix(
    y_test_np,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


recall_class_1 = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0.0
)


majority_class = (
    y_train
    .value_counts()
    .idxmax()
)


baseline_pred = np.full(
    len(y_test_np),
    majority_class
)


baseline_accuracy = (
    accuracy_score(
        y_test_np,
        baseline_pred
    )
)


print("=" * 70)
print("THRESHOLD EVALUATION")
print("=" * 70)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Recall class 1:",
    round(
        recall_class_1,
        4
    )
)

print(
    "Baseline accuracy:",
    round(
        baseline_accuracy,
        4
    )
)

print("\nConfusion matrix:")

print(cm)

print("\nClassification report:")

print(
    classification_report(
        y_test_np,
        y_pred,
        target_names=[
            "Voucher",
            "Free Shipping"
        ],
        zero_division=0
    )
)


def predict_one(
    total_amount,
    distance_km,
    membership_level
):

    if not isinstance(
        total_amount,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):
        raise TypeError(
            "total_amount must be numeric"
        )


    if not isinstance(
        distance_km,
        (
            int,
            float,
            np.integer,
            np.floating
        )
    ):
        raise TypeError(
            "distance_km must be numeric"
        )


    if membership_level not in MEMBERSHIP_MAP:
        raise ValueError(
            "Invalid membership level"
        )


    if total_amount < 0:
        raise ValueError(
            "total_amount must be >= 0"
        )


    if (
        distance_km <= 0
        or distance_km > 30
    ):
        raise ValueError(
            "distance_km must be in the range (0, 30]"
        )


    record = pd.DataFrame(
        [[
            float(total_amount),
            float(distance_km),
            MEMBERSHIP_MAP[
                membership_level
            ]
        ]],
        columns=FEATURE_NAMES
    )


    record_scaled = scaler.transform(
        record
    )


    net = float(
        np.dot(
            w,
            record_scaled[0]
        ) + b
    )


    predicted_label = int(
        net >= THRESHOLD
    )


    return {
        "net": net,
        "threshold": THRESHOLD,
        "predicted_label": predicted_label,
        "decision": LABEL_MAP[
            predicted_label
        ]
    }


test_cases = [
    {
        "name": "High value and short distance",
        "total_amount": 200,
        "distance_km": 6,
        "membership_level": "Gold"
    },
    {
        "name": "Medium value and long distance",
        "total_amount": 60,
        "distance_km": 25,
        "membership_level": "Bronze"
    },
    {
        "name": "Loyal customer",
        "total_amount": 80,
        "distance_km": 8,
        "membership_level": "VIP"
    }
]


results = []


for case in test_cases:

    result = predict_one(
        total_amount=case[
            "total_amount"
        ],
        distance_km=case[
            "distance_km"
        ],
        membership_level=case[
            "membership_level"
        ]
    )


    results.append({
        "test_case": case["name"],
        "total_amount": case[
            "total_amount"
        ],
        "distance_km": case[
            "distance_km"
        ],
        "membership_level": case[
            "membership_level"
        ],
        "net": result["net"],
        "prediction": result[
            "decision"
        ]
    })


results_df = pd.DataFrame(
    results
)


print(
    "\nNew sample predictions:"
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 8. SAVE MODEL
# ============================================================

artifact = {
    "feature_names": FEATURE_NAMES,
    "membership_map": MEMBERSHIP_MAP,
    "label_map": LABEL_MAP,
    "scaler": scaler,
    "train_medians": train_medians.to_dict(),
    "weights": w,
    "bias": b,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "threshold": THRESHOLD,
    "accuracy": accuracy,
    "recall_class_1": recall_class_1
}


MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)


joblib.dump(
    artifact,
    MODEL_PATH
)


print(
    "\nModel saved to:",
    MODEL_PATH
)


# ============================================================
# 9. RELOAD MODEL TEST
# ============================================================

loaded_artifact = joblib.load(
    MODEL_PATH
)


print("\nReload test:")


loaded_feature_names = (
    loaded_artifact[
        "feature_names"
    ]
)

loaded_membership_map = (
    loaded_artifact[
        "membership_map"
    ]
)

loaded_label_map = (
    loaded_artifact[
        "label_map"
    ]
)

loaded_scaler = (
    loaded_artifact[
        "scaler"
    ]
)

loaded_w = (
    loaded_artifact[
        "weights"
    ]
)

loaded_b = (
    loaded_artifact[
        "bias"
    ]
)

loaded_threshold = (
    loaded_artifact[
        "threshold"
    ]
)


sample = pd.DataFrame(
    [[
        200.0,
        6.0,
        loaded_membership_map[
            "Gold"
        ]
    ]],
    columns=loaded_feature_names
)


sample_scaled = (
    loaded_scaler.transform(
        sample
    )
)


loaded_net = float(
    np.dot(
        loaded_w,
        sample_scaled[0]
    ) + loaded_b
)


loaded_prediction = int(
    loaded_net >= loaded_threshold
)


print(
    "Net:",
    loaded_net
)

print(
    "Prediction:",
    loaded_label_map[
        loaded_prediction
    ]
)


# ============================================================
# 10. MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)

print("MODEL SUMMARY")

print("=" * 70)

print(
    "Features:",
    FEATURE_NAMES
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Epochs:",
    EPOCHS
)

print(
    "Threshold:",
    THRESHOLD
)

print(
    "Accuracy:",
    round(
        accuracy,
        4
    )
)

print(
    "Recall:",
    round(
        recall_class_1,
        4
    )
)

print(
    "Baseline accuracy:",
    round(
        baseline_accuracy,
        4
    )
)

print(
    "TN:",
    tn
)

print(
    "FP:",
    fp
)

print(
    "FN:",
    fn
)

print(
    "TP:",
    tp
)


print("\n" + "=" * 70)

print("CONCLUSION")

print("=" * 70)

print(
    "The Adaline model classifies orders into "
    "Free Shipping or Voucher."
)

print(
    "The model uses order value, delivery distance "
    "and membership level."
)

print(
    "The distance feature is simulated because "
    "the original dataset does not provide it."
)

print(
    "The membership level is derived from previous "
    "customer purchasing history."
)

print(
    "The target label is generated using a predefined "
    "rule for this academic experiment."
)

print(
    "The model uses the Widrow-Hoff learning rule "
    "with continuous output and MSE."
)

print(
    "A threshold is applied to convert the continuous "
    "Adaline output into a binary decision."
)

print(
    "The model should be used only within the "
    "range and data conditions tested above."
)

print(
    "Human review is required for cases near "
    "the decision boundary or abnormal inputs."
)

print(
    "A next step is to collect real free-shipping "
    "decision data and compare Adaline with "
    "more flexible classification models."
)


print("\n" + "=" * 70)

print("TEST SUMMARY")

print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)