import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊"
)

st.title("📊 Dataset Dashboard")

model = joblib.load("heart_disease_model.pkl")

df = pd.read_csv("heart-disease.csv")
st.header("Heart Disease Distribution")


    # Load dataset for dashboard (fallback if file missing)
try:
        df = pd.read_csv("heart-disease.csv")
except Exception:
        st.warning("Dataset 'heart-disease.csv' not found. Showing placeholder distribution.")
        # create a small placeholder dataframe with both classes
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0]})

fig, ax = plt.subplots(figsize=(5, 5))

df["target"].value_counts().sort_index().plot.pie(
        labels=["Heart Disease", "No Heart Disease"],
        autopct="%1.1f%%",
        startangle=90,
        ax=ax
    )

ax.set_ylabel("")

st.pyplot(fig)

st.subheader("Age Distribution")

fig2, ax2 = plt.subplots(figsize=(8,4))

ax2.hist(
        df["age"],
        bins=10,
        edgecolor="black"
    )

ax2.set_xlabel("Age")
ax2.set_ylabel("Number of Patients")
ax2.set_title("Distribution of Patient Ages")

st.pyplot(fig2)

st.subheader("Feature Correlation Heatmap")

fig3, ax3 = plt.subplots(figsize=(10, 8))

correlation = df.corr(numeric_only=True)

sns.heatmap(
        correlation,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax3
    )

st.pyplot(fig3)

st.subheader("Feature Importance")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
        "Feature": df.drop("target", axis=1).columns,
        "Importance": importance
    })

feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=True
    )

fig4, ax4 = plt.subplots(figsize=(8,6))

ax4.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

ax4.set_xlabel("Importance")
ax4.set_ylabel("Feature")
ax4.set_title("Random Forest Feature Importance")

st.pyplot(fig4)