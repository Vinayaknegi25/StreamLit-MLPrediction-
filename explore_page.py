import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
#functions from the notebook
def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map


def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)

def clean_education(x):
    if 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)' in x:
        return 'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)'

    if 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)' in x:
        return 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)'

    if 'Associate degree (A.A., A.S., etc.)' in x:
        return 'Associate degree (A.A., A.S., etc.)'

    if 'Some college/university study without earning a degree' in x:
        return 'Some college/university study without earning a degree'
    if 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)' in x:
        return 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)'
    if 'Professional degree (JD, MD, etc.)' in x:
        return 'Professional degree (JD, MD, etc.)'
    if 'Other doctoral degree (Ph.D., Ed.D., etc.)' in x:
        return 'Other doctoral degree (Ph.D., Ed.D., etc.)'
    return 'Primary/elementary school'


@st.cache #to avoid reloading 
def load_data():
    df = pd.read_csv("survey_results_public.csv")
    df = df[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedCompYearly"]]

    df = df[df["ConvertedCompYearly"].notnull()]

    df = df.dropna()
    df = df[df["Employment"] == "Employed full-time"]
    df = df.drop("Employment", axis=1)

    country_map = shorten_categories(df.Country.value_counts(), 400)
    df["Country"] = df["Country"].map(country_map)
    df = df[df["ConvertedCompYearly"] <= 250000]
    df = df[df["ConvertedCompYearly"] >= 10000]
    df = df[df["Country"] != "Other"]

    df["YearsCodePro"] = df["YearsCodePro"].apply(clean_experience)
    df["EdLevel"] = df["EdLevel"].apply(clean_education)
    df = df.rename({"ConvertedCompYearly": "Salary"}, axis=1)
    return df

df = load_data()

def show_explore_page():

    st.title("Explore Software Engineer Salaries")

    data = df["Country"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)
    
    st.write("""#### Mean Salary Based On Country""") 

    data = df.groupby(["Country"])["Salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write("""#### Mean Salary Based On Experience""")
    data = df.groupby(["YearsCodePro"])["Salary"].mean().sort_values(ascending=True)
    st.line_chart(data)

