import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib


# ==========================================
# 1. CREATE DATASET
# ==========================================

np.random.seed(42)

n = 1000

temperature = np.random.uniform(15, 45, n)
humidity = np.random.uniform(20, 95, n)
uv_index = np.random.uniform(1, 12, n)
wind_speed = np.random.uniform(0, 30, n)
exposure_hours = np.random.uniform(0.5, 8, n)

sunscreen_used = np.random.randint(0, 2, n)
moisturizer_used = np.random.randint(0, 2, n)

skin_types = np.random.choice(
    ["Oily", "Dry", "Normal"],
    n
)


# ==========================================
# 2. CALCULATE RISK SCORE
# ==========================================

risk_score = (
    temperature * 0.8
    + humidity * 0.15
    + uv_index * 4
    + exposure_hours * 5
)

# Sunscreen protection
risk_score -= sunscreen_used * 15

# Moisturizer protection
risk_score -= moisturizer_used * 8


# Dry skin + low humidity
risk_score += np.where(
    (skin_types == "Dry") & (humidity < 50),
    12,
    0
)


# Oily skin + high humidity
risk_score += np.where(
    (skin_types == "Oily") & (humidity > 75),
    10,
    0
)


# ==========================================
# 3. CREATE RISK LABEL
# ==========================================

risk = np.where(
    risk_score < 45,
    "Low",
    np.where(
        risk_score < 75,
        "Medium",
        "High"
    )
)


# ==========================================
# 4. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame({

    "temperature": temperature,

    "humidity": humidity,

    "uv_index": uv_index,

    "wind_speed": wind_speed,

    "exposure_hours": exposure_hours,

    "sunscreen_used": sunscreen_used,

    "moisturizer_used": moisturizer_used,

    "skin_type": skin_types,

    "skin_risk": risk

})


# ==========================================
# 5. ENCODE SKIN TYPE
# ==========================================

encoder = LabelEncoder()

df["skin_type_encoded"] = encoder.fit_transform(
    df["skin_type"]
)


# ==========================================
# 6. FEATURES
# ==========================================

X = df[
    [
        "temperature",
        "humidity",
        "uv_index",
        "wind_speed",
        "exposure_hours",
        "sunscreen_used",
        "moisturizer_used",
        "skin_type_encoded"
    ]
]


y = df["skin_risk"]


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)


# ==========================================
# 8. RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(

    n_estimators=200,

    random_state=42

)


model.fit(X_train, y_train)


# ==========================================
# 9. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 10. MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(

    y_test,

    y_pred

)


print("\n==============================")
print("AURELIA AI MODEL")
print("==============================")

print(
    "Model Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

joblib.dump(

    model,

    "model/skin_risk_model.pkl"

)


joblib.dump(

    encoder,

    "model/skin_type_encoder.pkl"

)


# ==========================================
# 12. SAVE DATASET
# ==========================================

df.to_csv(

    "dataset/skin_weather_data.csv",

    index=False

)


print("\nModel saved:")
print("model/skin_risk_model.pkl")

print("\nEncoder saved:")
print("model/skin_type_encoder.pkl")

print("\nDataset saved:")
print("dataset/skin_weather_data.csv")

print("\nAURELIA AI ML pipeline completed successfully!")