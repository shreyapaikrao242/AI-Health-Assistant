import streamlit as st
import pandas as pd
import joblib
from pdf_report import create_report

model = joblib.load("heart_disease_model.pkl")

st.set_page_config(
    page_title="Heart Disease",
    page_icon="❤️"
)

st.title("❤️ Heart Disease Prediction")

st.write(
    "Enter the patient's health information below and click Predict."
)

# User Inputs
age = st.number_input(
    "Enter Age",
    min_value=1,
    max_value=120,
    value=30
)

sex = st.selectbox(
    "Select Sex",
    ("Female", "Male")
)

sex = 1 if sex == "Male" else 0

cp_option = st.selectbox(
    "Chest Pain Type",
    (
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    )
)


cp = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }[cp_option]

trestbps = st.number_input(
        "Resting Blood Pressure",
        min_value=50,
        max_value=250,
        value=120
    )

chol = st.number_input(
        "Cholesterol Level",
        min_value=50,
        max_value=600,
        value=200
    )

fbs_option = st.selectbox(
        "Fasting Blood Sugar (>120 mg/dl)",
        (
            "No",
            "Yes"
        )
    )

fbs = {
        "No":0,
        "Yes":1
    }[fbs_option]

restecg_option = st.selectbox(
        "Resting ECG Result",
        (
            "Normal",
            "ST-T Wave Abnormality",
            "Left Ventricular Hypertrophy"
        )
    )

restecg = {
        "Normal":0,
        "ST-T Wave Abnormality":1,
        "Left Ventricular Hypertrophy":2
    }[restecg_option]

thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=50,
        max_value=250,
        value=150
    )

exang_option = st.selectbox(
        "Exercise Induced Angina",
        (
            "No",
            "Yes"
        )
    )

exang = {
        "No":0,
        "Yes":1
    }[exang_option]

oldpeak = st.number_input(
        "Oldpeak (ST Depression)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

slope_option = st.selectbox(
        "Slope of Peak Exercise",
        (
            "Upsloping",
            "Flat",
            "Downsloping"
        )
    )

slope = {
        "Upsloping":0,
        "Flat":1,
        "Downsloping":2
    }[slope_option]

ca = st.slider(
        "Number of Major Vessels (0–4)",
        0,
        4,
        0
    )

thal_option = st.selectbox(
        "Thalassemia Test",
        (
            "Unknown",
            "Normal",
            "Fixed Defect",
            "Reversible Defect"
        )
    )

thal = {
        "Unknown":0,
        "Normal":1,
        "Fixed Defect":2,
        "Reversible Defect":3
    }[thal_option]

    # Predict Button
if st.button("Predict"):

        # Create DataFrame
        patient_data = pd.DataFrame({
            "age": [age],
            "sex": [sex],
            "cp": [cp],
            "trestbps": [trestbps],
            "chol": [chol],
            "fbs": [fbs],
            "restecg": [restecg],
            "thalach": [thalach],
            "exang": [exang],
            "oldpeak": [oldpeak],
            "slope": [slope],
            "ca": [ca],
            "thal": [thal]
        })

        # Prediction
        with st.spinner("Analyzing patient data..."):
            prediction = model.predict(patient_data)
            probability = model.predict_proba(patient_data)

        confidence = max(probability[0]) * 100

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        # Disease probability
        risk = probability[0][0] * 100

        st.subheader("Prediction Result")

        if prediction[0] == 0:
    
            st.error("⚠️ Patient is at Risk of Heart Disease")

            st.metric(
                "Estimated Risk",
               f"{risk:.2f}%"
            )

            st.progress(int(risk))

            if risk < 40:
                st.info("🟢 Low Risk")
            elif risk < 70:
                st.warning("🟡 Moderate Risk")
            else:
                st.error("🔴 High Risk")

        else:

            no_risk = probability[0][1] * 100
            st.balloons()

            st.success("✅ Patient is NOT at Risk of Heart Disease")

            st.metric(
                label="Healthy Probability",
                value=f"{no_risk:.2f}%"
            )

            st.progress(int(no_risk))

            st.success("🟢 Healthy")
            st.divider()

        st.subheader("💡 Health Suggestions")

        if prediction[0] == 0:

            st.write("• 🥗 Eat a balanced, heart-healthy diet.")
            st.write("• 🚶 Exercise regularly (after consulting a doctor).")
            st.write("• 🚭 Avoid smoking and tobacco.")
            st.write("• 🩺 Schedule a medical check-up with a cardiologist.")
            st.write("• ❤️ Monitor blood pressure and cholesterol regularly.")

        else:

            st.write("• ✅ Continue a healthy lifestyle.")
            st.write("• 🥗 Eat nutritious foods.")
            st.write("• 🏃 Stay physically active.")
            st.write("• 😴 Get enough sleep.")
            st.write("• 🩺 Have routine health check-ups.")
            st.divider()

        st.caption(
            "⚠️ Disclaimer: This prediction is generated using a Machine Learning model trained on historical data. It is for educational purposes only and should not be considered a medical diagnosis. Always consult a qualified healthcare professional."
        )
        st.divider()

        st.caption(
            "This prediction is generated using a Machine Learning model trained on historical data. "
            "It is intended for educational purposes only and should not be considered a medical diagnosis. "
            "Always consult a qualified healthcare professional."
        )
        st.divider()