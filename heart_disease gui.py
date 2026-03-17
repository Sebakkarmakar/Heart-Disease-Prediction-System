import pandas as pd
import pickle
import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv(r"C:\Users\admin\Desktop\ML PROJECTS\Heart Deases prediction\1-heart.csv")
X = df.drop("target", axis=1)
y = df["target"]
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)
pickle.dump(model, open("heart_model.pkl", "wb"))
model = pickle.load(open("heart_model.pkl", "rb"))

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease")

st.subheader("🧾 Patient Medical Details")

age = st.number_input(
    "Age (years)",
    min_value=20,
    max_value=100,
    help="Patient age in years"
)

sex = st.selectbox(
    "Sex",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic"
    }[x]
)

trestbps = st.number_input(
    "Resting Blood Pressure (mm Hg)",
    min_value=80,
    max_value=200,
    help="Normal: 90–120 | High: >140"
)

chol = st.number_input(
    "Serum Cholesterol (mg/dl)",
    min_value=100,
    max_value=600,
    help="Normal: <200 | Borderline: 200–239 | High: ≥240"
)

fbs = st.selectbox(
    "Fasting Blood Sugar",
    options=[0, 1],
    format_func=lambda x: "≤ 120 mg/dl (Normal)" if x == 0 else "> 120 mg/dl (High)"
)

restecg = st.selectbox(
    "Resting ECG Result",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    }[x]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved",
    min_value=60,
    max_value=220,
    help="Normal depends on age (≈ 220 − age)"
)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=6.0,
    help="0 = Normal | Higher values indicate abnormality"
)

slope = st.selectbox(
    "Slope of Peak Exercise ST Segment",
    options=[0, 1, 2],
    format_func=lambda x: {
        0: "Upsloping (Better)",
        1: "Flat (Moderate Risk)",
        2: "Downsloping (High Risk)"
    }[x]
)

ca = st.selectbox(
    "Number of Major Vessels Colored by Fluoroscopy",
    options=[0, 1, 2, 3],
    format_func=lambda x: f"{x} vessel(s)"
)

thal = st.selectbox(
    "Thalassemia",
    options=[0, 1, 2, 3],
    format_func=lambda x: {
        0: "Normal",
        1: "Fixed Defect",
        2: "Reversible Defect",
        3: "Unknown"
    }[x]
)
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")