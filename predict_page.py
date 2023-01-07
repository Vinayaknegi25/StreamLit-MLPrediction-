import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### Enter information to predict the salary""")

    countries = (
        "India",
        "United States of America",
        "United Kingdom of Great Britain and Northern Ireland",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)",
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)",
        "Associate degree (A.A., A.S., etc.)",
        "Some college/university study without earning a degree",
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)",
        "Professional degree (JD, MD, etc.)",
        "Other doctoral degree (Ph.D., Ed.D., etc.)",
        "Primary/elementary school",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    expericence = st.slider("Years of Experience", 0, 50, 1)

    OnClick = st.button("Calculate Salary")

    if OnClick:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"Estimated salary in $ is {salary[0]:.2f}")


