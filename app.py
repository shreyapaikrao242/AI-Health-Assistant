import streamlit as st
 
 
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 AI Health Assistant")

st.write("## Welcome!")

st.write("""
This application helps predict different health conditions using
Machine Learning.

Choose a page from the left sidebar.
""")

st.info("👈 Click a page in the sidebar to begin.")

st.markdown("---")

st.subheader("Available Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("❤️ Heart Disease")

with col2:
    st.warning("🩸 Diabetes")
    st.caption("Coming Soon")

with col3:
    st.info("🫁 Lung Disease")
    st.caption("Coming Soon")