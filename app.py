import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
from dotenv import load_dotenv

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query_huggingface(prompt):
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.status_code}"

st.title("🧠 AI CSV Co-Pilot (Free Version)")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    question = st.text_input("Ask a question about your data:")

    if question:
        prompt = f"You are a data expert. Given the following dataframe:\n\n{df.head(5).to_string()}\n\nAnswer this question:\n{question}"
        answer = query_huggingface(prompt)
        st.write("🤖", answer)
