import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from IPython.display import display

warnings.filterwarnings("ignore")

# ============================================================
# 0. CONFIGURATION
# ============================================================

CSV_PATH = "../data/ecommerce_sales_34500.csv"
MODEL_PATH = "../data/perceptron_free_shipping.pkl"

TARGET_NAME = "y"

LABEL_MAP = {
    0: "Voucher",
    1: "Free Shipping"
}

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

LEARNING_RATE = 0.01
EPOCHS = 50


# ============================================================
# 1. LOAD DATA
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}"
    )

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("DATASET")
print("=" * 70)

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
display(df.head())


# ============================================================
# 2. DATA PREPARATION
# ============================================================

df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

df = df.sort_values(
    by=["customer_id", "order_date"]
).reset_index(drop=True)


# Previous customer spending
df["previous_spending"] = (
    df.groupby("customer_id")["total_amount"].cumsum()
    - df["total_amount"]
)


# Previous number of orders
df["previous_orders"] = (
    df.groupby("customer_id").cumcount()
)


# Membership level
def get_membership(row):

    if (
        row["previous_spending"] >= 1500
        or row["previous_orders"] >= 8
    ):
        return "VIP"

    elif (
        row["previous_spending"] >= 800
        or row["previous_orders"] >= 5
    ):
        return "Gold"

    elif (
        row["previous_spending"] >= 300
        or row["previous_orders"] >= 2
    ):
        return "Silver"

    else:
        return "Bronze"


df["membership_level"] = df.apply(
    get_membership,
    axis=1
)

df["membership_code"] = df[
    "membership_level"
].map(MEMBERSHIP_MAP)


# ============================================================
# 3. SIMULATE DISTANCE
# ============================================================

np.random.seed(42)

df["distance_km"] = np.random.uniform(
    1,
    30,
    size=len(df)
)


# ============================================================
# 4. CREATE LABEL
# ============================================================

def create_label(row):

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

    else:
        return 0


df["y"] = df.apply(
    create_label,
    axis=1
)


# ============================================================
# 5. DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("DATA CHECK")
print("=" * 70)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nLabel distribution:")
print(df["y"].value_counts().sort_index())

print("\nLabel percentage:")
print(
    df["y"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
)

print("\nMembership distribution:")
print(
    df["membership_level"].value_counts()
)


# ============================================================
# 6. SELECT FEATURES
# ============================================================

X = df[FEATURE_NAMES].copy()

y = df[TARGET_NAME].copy()

assert len(X) == len(y)

assert set(y.unique()).issubset({0, 1})

assert list(X.columns) == FEATURE_NAMES

print("\n" + "=" * 70)
print("FEATURES")
print("=" * 70)

print("Features:", FEATURE_NAMES)
print("Target:", TARGET_NAME)
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

df = df.sort_values(
    by="order_date"
).reset_index(drop=True)

X = df[FEATURE_NAMES].copy()

y = df[TARGET_NAME].copy()

split_index = int(
    len(df) * 0.8
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

print("\n" + "=" * 70)
print("TRAIN / TEST")
print("=" * 70)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

print("\nTrain labels:")
print(
    y_train.value_counts()
    .sort_index()
)

print("\nTest labels:")
print(
    y_test.value_counts()
    .sort_index()
)


# ============================================================
# 8. PREPROCESSING
# ============================================================

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

assert not np.isnan(
    X_train_scaled
).any()

assert not np.isnan(
    X_test_scaled
).any()

assert not np.isinf(
    X_train_scaled
).any()

assert not np.isinf(
    X_test_scaled
).any()

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

print("Train mean:")
print(
    X_train_scaled.mean(axis=0)
)

print("\nTrain standard deviation:")
print(
    X_train_scaled.std(axis=0)
)


# ============================================================
# 9. TRAIN PERCEPTRON
# ============================================================

X_train_np = X_train_scaled

y_train_np = y_train.to_numpy()

w = np.zeros(
    X_train_np.shape[1]
)

b = 0.0

errors_history = []

print("\n" + "=" * 70)
print("PERCEPTRON TRAINING")
print("=" * 70)

for epoch in range(EPOCHS):

    errors = 0

    for i in range(
        len(X_train_np)
    ):

        x_i = X_train_np[i]

        y_i = y_train_np[i]

        net = (
            np.dot(w, x_i)
            + b
        )

        y_hat = (
            1
            if net >= 0
            else 0
        )

        error = (
            y_i - y_hat
        )

        if error != 0:

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

            errors += 1

    errors_history.append(
        errors
    )

    print(
        f"Epoch {epoch + 1:02d}: "
        f"errors = {errors}"
    )


print("\nFinal weights:")

for feature, weight in zip(
    FEATURE_NAMES,
    w
):
    print(
        f"{feature}: {weight:.6f}"
    )

print("\nFinal bias:")
print(
    f"{b:.6f}"
)


# ============================================================
# 10. TRAINING ERROR GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        EPOCHS + 1
    ),
    errors_history,
    marker="o"
)

plt.title(
    "Perceptron Training Errors"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Number of Errors"
)

plt.grid(True)

plt.show()


# ============================================================
# 11. PREDICTION FUNCTION
# ============================================================

def predict_perceptron(
    X,
    weights,
    bias
):

    predictions = []

    for x in X:

        net = (
            np.dot(
                weights,
                x
            )
            + bias
        )

        prediction = (
            1
            if net >= 0
            else 0
        )

        predictions.append(
            prediction
        )

    return np.array(
        predictions
    )


# ============================================================
# 12. TEST SET EVALUATION
# ============================================================

y_pred = predict_perceptron(
    X_test_scaled,
    w,
    b
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()

if (
    tp + fn
) > 0:

    recall_class_1 = (
        tp
        / (tp + fn)
    )

else:

    recall_class_1 = 0


print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Recall class 1: {recall_class_1:.4f}"
)

print("\nConfusion Matrix:")

print(cm)

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Voucher",
            "Free Shipping"
        ],
        zero_division=0
    )
)


# ============================================================
# 13. BASELINE
# ============================================================

majority_class = (
    y_train.mode()[0]
)

baseline_pred = np.full(
    len(y_test),
    majority_class
)

baseline_accuracy = (
    accuracy_score(
        y_test,
        baseline_pred
    )
)

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    "Majority class:",
    majority_class
)

print(
    f"Baseline accuracy: "
    f"{baseline_accuracy:.4f}"
)

print(
    f"Perceptron accuracy: "
    f"{accuracy:.4f}"
)


# ============================================================
# 14. WRONG PREDICTIONS
# ============================================================

wrong_indices = np.where(
    y_test.to_numpy()
    != y_pred
)[0]

print("\n" + "=" * 70)
print("WRONG PREDICTIONS")
print("=" * 70)

print(
    "Number of wrong predictions:",
    len(wrong_indices)
)

if len(wrong_indices) > 0:

    wrong_sample_indices = (
        wrong_indices[:5]
    )

    wrong_samples = df.iloc[
        split_index
        + wrong_sample_indices
    ].copy()

    wrong_samples[
        "actual"
    ] = y_test.iloc[
        wrong_sample_indices
    ].values

    wrong_samples[
        "predicted"
    ] = y_pred[
        wrong_sample_indices
    ]

    display(
        wrong_samples[
            [
                "order_id",
                "total_amount",
                "distance_km",
                "membership_level",
                "actual",
                "predicted"
            ]
        ]
    )

    for index in wrong_sample_indices:

        x = X_test_scaled[
            index
        ]

        net = (
            np.dot(
                w,
                x
            )
            + b
        )

        actual = (
            y_test.iloc[index]
        )

        predicted = (
            y_pred[index]
        )

        print("\n" + "-" * 50)

        print(
            "Test index:",
            index
        )

        print(
            "Actual:",
            actual
        )

        print(
            "Predicted:",
            predicted
        )

        print(
            "Net:",
            round(net, 6)
        )


# ============================================================
# 15. SINGLE PREDICTION
# ============================================================

def predict_one(
    total_amount,
    distance_km,
    membership_level
):

    if not isinstance(
        total_amount,
        (int, float)
    ):
        raise TypeError(
            "total_amount must be numeric"
        )

    if not isinstance(
        distance_km,
        (int, float)
    ):
        raise TypeError(
            "distance_km must be numeric"
        )

    if not isinstance(
        membership_level,
        str
    ):
        raise TypeError(
            "membership_level must be a string"
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
            "distance_km must be between 0 and 30"
        )

    if (
        membership_level
        not in MEMBERSHIP_MAP
    ):

        raise ValueError(
            "Invalid membership_level"
        )

    membership_code = (
        MEMBERSHIP_MAP[
            membership_level
        ]
    )

    record = pd.DataFrame(
        [[
            total_amount,
            distance_km,
            membership_code
        ]],
        columns=FEATURE_NAMES
    )

    record = record.fillna(
        train_medians
    )

    record_scaled = (
        scaler.transform(
            record
        )
    )

    net = (
        np.dot(
            w,
            record_scaled[0]
        )
        + b
    )

    prediction = (
        1
        if net >= 0
        else 0
    )

    decision = LABEL_MAP[
        prediction
    ]

    return {
        "total_amount": total_amount,
        "distance_km": distance_km,
        "membership_level": membership_level,
        "net": net,
        "predicted_label": prediction,
        "decision": decision
    }


# ============================================================
# 16. DEMO
# ============================================================

print("\n" + "=" * 70)
print("DEMO")
print("=" * 70)


demo_cases = [
    {
        "name": "Case 1",
        "total_amount": 250,
        "distance_km": 5,
        "membership_level": "Gold"
    },
    {
        "name": "Case 2",
        "total_amount": 30,
        "distance_km": 25,
        "membership_level": "Bronze"
    },
    {
        "name": "Case 3",
        "total_amount": 150,
        "distance_km": 8,
        "membership_level": "Silver"
    },
    {
        "name": "Case 4",
        "total_amount": 50,
        "distance_km": 20,
        "membership_level": "Gold"
    }
]


for case in demo_cases:

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

    print("\n" + "-" * 50)

    print(
        case["name"]
    )

    print(
        "Order value:",
        result["total_amount"]
    )

    print(
        "Distance:",
        result["distance_km"],
        "km"
    )

    print(
        "Membership:",
        result["membership_level"]
    )

    print(
        "Net:",
        round(
            result["net"],
            6
        )
    )

    print(
        "Predicted label:",
        result["predicted_label"]
    )

    print(
        "Decision:",
        result["decision"]
    )


# ============================================================
# 17. VALIDATION TEST CASES
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION TEST CASES")
print("=" * 70)

test_results = []


# Normal class 0
try:

    result = predict_one(
        total_amount=30,
        distance_km=25,
        membership_level="Bronze"
    )

    expected = 0

    actual = (
        result["predicted_label"]
    )

    status = (
        "PASS"
        if actual == expected
        else "FAIL"
    )

    test_results.append({
        "test_case": "Normal class 0",
        "expected": expected,
        "actual": actual,
        "status": status
    })

except Exception as e:

    test_results.append({
        "test_case": "Normal class 0",
        "expected": 0,
        "actual": str(e),
        "status": "FAIL"
    })


# Normal class 1
try:

    result = predict_one(
        total_amount=250,
        distance_km=5,
        membership_level="Gold"
    )

    expected = 1

    actual = (
        result["predicted_label"]
    )

    status = (
        "PASS"
        if actual == expected
        else "FAIL"
    )

    test_results.append({
        "test_case": "Normal class 1",
        "expected": expected,
        "actual": actual,
        "status": status
    })

except Exception as e:

    test_results.append({
        "test_case": "Normal class 1",
        "expected": 1,
        "actual": str(e),
        "status": "FAIL"
    })


# Near threshold
try:

    result = predict_one(
        total_amount=100,
        distance_km=10,
        membership_level="Bronze"
    )

    expected = 1

    actual = (
        result["predicted_label"]
    )

    status = (
        "PASS"
        if actual == expected
        else "FAIL"
    )

    test_results.append({
        "test_case": "Near threshold",
        "expected": expected,
        "actual": actual,
        "status": status
    })

except Exception as e:

    test_results.append({
        "test_case": "Near threshold",
        "expected": 1,
        "actual": str(e),
        "status": "FAIL"
    })


# Missing feature
try:

    predict_one(
        total_amount=100,
        distance_km=10,
        membership_level=None
    )

    test_results.append({
        "test_case": "Missing feature",
        "expected": "Error",
        "actual": "No error",
        "status": "FAIL"
    })

except (
    TypeError,
    ValueError
):

    test_results.append({
        "test_case": "Missing feature",
        "expected": "Error",
        "actual": "Error",
        "status": "PASS"
    })


# Wrong type
try:

    predict_one(
        total_amount="100",
        distance_km=10,
        membership_level="Gold"
    )

    test_results.append({
        "test_case": "Wrong type",
        "expected": "Error",
        "actual": "No error",
        "status": "FAIL"
    })

except (
    TypeError,
    ValueError
):

    test_results.append({
        "test_case": "Wrong type",
        "expected": "Error",
        "actual": "Error",
        "status": "PASS"
    })


# Invalid range
try:

    predict_one(
        total_amount=100,
        distance_km=50,
        membership_level="Gold"
    )

    test_results.append({
        "test_case": "Invalid range",
        "expected": "Error",
        "actual": "No error",
        "status": "FAIL"
    })

except (
    TypeError,
    ValueError
):

    test_results.append({
        "test_case": "Invalid range",
        "expected": "Error",
        "actual": "Error",
        "status": "PASS"
    })


# ============================================================
# 18. SAVE MODEL ARTIFACT
# ============================================================

artifact = {
    "feature_names": FEATURE_NAMES,
    "membership_map": MEMBERSHIP_MAP,
    "label_map": LABEL_MAP,
    "scaler": scaler,
    "train_medians": train_medians,
    "weights": w,
    "bias": b,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS
}

joblib.dump(
    artifact,
    MODEL_PATH
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    "Path:",
    MODEL_PATH
)


# ============================================================
# 19. RELOAD MODEL
# ============================================================

loaded_artifact = joblib.load(
    MODEL_PATH
)

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

loaded_train_medians = (
    loaded_artifact[
        "train_medians"
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


def predict_one_loaded(
    total_amount,
    distance_km,
    membership_level
):

    if not isinstance(
        total_amount,
        (int, float)
    ):
        raise TypeError(
            "total_amount must be numeric"
        )

    if not isinstance(
        distance_km,
        (int, float)
    ):
        raise TypeError(
            "distance_km must be numeric"
        )

    if not isinstance(
        membership_level,
        str
    ):
        raise TypeError(
            "membership_level must be a string"
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
            "distance_km must be between 0 and 30"
        )

    if (
        membership_level
        not in loaded_membership_map
    ):
        raise ValueError(
            "Invalid membership_level"
        )

    membership_code = (
        loaded_membership_map[
            membership_level
        ]
    )

    record = pd.DataFrame(
        [[
            total_amount,
            distance_km,
            membership_code
        ]],
        columns=loaded_feature_names
    )

    record = record.fillna(
        loaded_train_medians
    )

    record_scaled = (
        loaded_scaler.transform(
            record
        )
    )

    net = (
        np.dot(
            loaded_w,
            record_scaled[0]
        )
        + loaded_b
    )

    prediction = (
        1
        if net >= 0
        else 0
    )

    return {
        "net": net,
        "predicted_label": prediction,
        "decision": loaded_label_map[
            prediction
        ]
    }


# ============================================================
# 20. RELOAD TEST
# ============================================================

original_result = predict_one(
    total_amount=250,
    distance_km=5,
    membership_level="Gold"
)

reloaded_result = predict_one_loaded(
    total_amount=250,
    distance_km=5,
    membership_level="Gold"
)

reload_status = (
    original_result[
        "predicted_label"
    ]
    == reloaded_result[
        "predicted_label"
    ]
    and
    np.isclose(
        original_result["net"],
        reloaded_result["net"]
    )
)

test_results.append({
    "test_case": "Reload artifact",
    "expected": original_result[
        "predicted_label"
    ],
    "actual": reloaded_result[
        "predicted_label"
    ],
    "status": (
        "PASS"
        if reload_status
        else "FAIL"
    )
})


# ============================================================
# 21. FINAL TEST RESULT
# ============================================================

results_df = pd.DataFrame(
    test_results
)

print("\n" + "=" * 70)
print("FINAL TEST RESULTS")
print("=" * 70)

display(results_df)

passed = (
    results_df["status"] == "PASS"
).sum()

failed = (
    results_df["status"] == "FAIL"
).sum()

print(
    "\nPassed:",
    passed
)

print(
    "Failed:",
    failed
)


# ============================================================
# 22. FINAL DEMO
# ============================================================

print("\n" + "=" * 70)
print("FINAL DEMO")
print("=" * 70)

demo = predict_one_loaded(
    total_amount=200,
    distance_km=6,
    membership_level="Gold"
)

print(
    "Input:"
)

print(
    "total_amount = 200"
)

print(
    "distance_km = 6"
)

print(
    "membership_level = Gold"
)

print(
    "\nNet:",
    round(
        demo["net"],
        6
    )
)

print(
    "Predicted label:",
    demo["predicted_label"]
)

print(
    "Decision:",
    demo["decision"]
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)