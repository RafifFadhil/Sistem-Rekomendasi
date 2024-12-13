import streamlit as st 
import pandas as pd 

st.title("Sistem Rekomendasi")

df = pd.read_excel("article.xlsx")
st.dataframe(df)
