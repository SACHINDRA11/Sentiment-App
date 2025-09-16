import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Sentiment Analysis", page_icon="📝")
st.title("Sentiment Analysis")
st.write("Type any text and click **Analyze**.")

# --- load model (3-class: NEGATIVE / NEUTRAL / POSITIVE)
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )

clf = load_model()

# label mapping (covers both styles: LABEL_0/1/2 or negative/neutral/positive)
LABEL_MAP = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "NEUTRAL",
    "LABEL_2": "POSITIVE",
    "negative": "NEGATIVE",
    "neutral":  "NEUTRAL",
    "positive": "POSITIVE"
}

text = st.text_area("Your text:", height=160, placeholder="e.g. This product is amazing!")

if st.button("Analyze"):
    if text.strip():
        res = clf(text[:1000])[0]
        label = LABEL_MAP.get(res["label"], str(res["label"]).upper())
        score = res["score"] * 100
        st.success(f"Sentiment: {label} | Confidence: {score:.1f}%")
    else:
        st.warning("Please enter some text above.")