import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page


var= st.sidebar.selectbox("Explore Or Predict", ("Predict", "Explore"))

if var == "Predict":
    show_predict_page()
else:
   show_explore_page()

 