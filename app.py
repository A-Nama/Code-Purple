import streamlit as st
import pandas as pd
from inference import predict_bias, interpret_bias_score

EQUALITY_PURPLE = "#9999FF"

st.set_page_config(page_title="Code Purple - Policy Bias Analyzer", layout="wide")

st.markdown(
    f"""
    <style>
        html, body, [class*="css"]  {{
            font-family: 'Segoe UI', sans-serif;
            background-color: {EQUALITY_PURPLE};
            color: white;
        }}
        .stTextArea textarea, .stFileUploader, .stTextInput {{
            background-color: white !important;
            color: black !important;
        }}
        .stButton button {{
            background-color: white;
            color: black;
            font-weight: bold;
        }}
        .stMetric {{
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💜 Code Purple - Policy Bias Analyzer")

st.markdown("Upload a policy document or paste it below to get an overall bias score (0 = neutral, 1 = highly biased).")

uploaded_file = st.file_uploader("📄 Upload a `.txt` or `.csv` file", type=["txt", "csv"])
user_input = st.text_area("Or paste the full document here:", height=300)

def split_into_paragraphs(text):
    return [p.strip() for p in text.split("\n") if p.strip()]

if uploaded_file or user_input:
    with st.spinner("Analyzing..."):
        paragraphs = []

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                if 'paragraph_text' in df.columns:
                    paragraphs = df['paragraph_text'].tolist()
                else:
                    st.error("CSV must have a 'paragraph_text' column.")
                    st.stop()
            else:
                paragraphs = split_into_paragraphs(uploaded_file.read().decode("utf-8"))

        elif user_input:
            paragraphs = split_into_paragraphs(user_input)

        scores = predict_bias(paragraphs)
        avg_score = float(sum(scores)) / len(scores)
        report = interpret_bias_score(avg_score)

        st.subheader("📊 Bias Analysis Result")
        st.metric("Overall Bias Score", f"{avg_score:.3f}", help="Score ranges from 0 (no bias) to 1 (highly biased)")
        st.write(report)
