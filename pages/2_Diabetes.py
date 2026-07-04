import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("diabetes_model.pkl")

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩸"
)

st.title("🩸 Diabetes Prediction")

st.write(
    "Enter the patient's information below and click Predict."
)

st.markdown("---")
# User Inputs
pregnancies = st.number_input(
    "Number of Pregnancies",
    min_value=0,
    max_value=20,
    value=0
)

glucose = st.number_input(
    "Glucose Level",
    min_value=0,
    max_value=300,
    value=100
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin Level",
    min_value=0,
    max_value=900,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0,
    step=0.1
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5,
    step=0.001
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=30
)
# Predict Button
if st.button("Predict Diabetes Risk"):

    patient_data = pd.DataFrame({
        "Pregnancies": [pregnancies],
        "Glucose": [glucose],
        "BloodPressure": [blood_pressure],
        "SkinThickness": [skin_thickness],
        "Insulin": [insulin],
        "BMI": [bmi],
        "DiabetesPedigreeFunction": [dpf],
        "Age": [age]
    })

    prediction = model.predict(patient_data)
    probability = model.predict_proba(patient_data)

    confidence = max(probability[0]) * 100

    st.metric(
        "Model Confidence",
        f"{confidence:.2f}%"
    )

    st.subheader("Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ Patient is at Risk of Diabetes")

        risk = probability[0][1] * 100

        st.metric(
            "Estimated Risk",
            f"{risk:.2f}%"
        )

        st.progress(int(risk))
        st.divider()

        st.subheader("💡 Health Suggestions")

        st.write("• 🥗 Eat a balanced diet with less sugar.")
        st.write("• 🚶 Exercise for at least 30 minutes daily.")
        st.write("• 💧 Drink plenty of water.")     
        st.write("• 🩺 Consult a doctor for proper diagnosis.")
        st.write("• 📊 Monitor your blood sugar regularly.")

    else:

        healthy = probability[0][0] * 100

        st.balloons()

        st.success("✅ Patient is NOT at Risk of Diabetes")

        st.metric(
            "Healthy Probability",
            f"{healthy:.2f}%"
        )

        st.progress(int(healthy))
        st.divider()
        st.subheader("💡 Healthy Lifestyle Tips")

        st.write("• ✅ Continue eating a balanced diet.")
        st.write("• 🏃 Stay physically active.")
        st.write("• 😴 Get 7–8 hours of sleep.")
        st.write("• 💧 Drink enough water every day.")
        st.write("• 🩺 Get regular health check-ups.")
        st.divider()

st.caption(
    "⚠️ Disclaimer: This prediction is generated using a Machine Learning model trained on historical data. "
    "It is for educational purposes only and should not be considered a medical diagnosis. "
    "Always consult a qualified healthcare professional."
)
