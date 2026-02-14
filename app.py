import streamlit as st
import os
import gdown
from tensorflow.keras.models import load_model
import pickle
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(page_title="Fake News Classifier", layout="centered")


MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/file/d/1gKuy9T0Vo6TpD6lGjox4u-jZU2pWoBa9/view"

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

trained_model = load_trained_model()



@st.cache_resource
def load_model_and_tokenizer():
    model=trained_model
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

with st.spinner("Loading AI model... 🤖"):
    model, t_N = load_model_and_tokenizer()


lemmatizer = WordNetLemmatizer()


@st.cache_data
def preprocessor(input_text):
    words = re.sub('[^a-zA-Z]', ' ', input_text.lower())
    words = words.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    words = " ".join(words)

    seq = t_N.texts_to_sequences([words])
    pad_seq = pad_sequences(seq, maxlen=800)
    return pad_seq


animated_title = """
<style>
@keyframes fadeSlide {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}

.animated-title {
    font-family: 'Poppins', sans-serif;
    color: black;
    font-size: 40px !important;
    font-weight: 800;
    text-align: center;
    animation: fadeSlide 1.5s ease-out;
    background: white;
    margin-bottom: 20px;
}
</style>

<div class="animated-title">Fake News Classifier (USA)</div>
"""
st.markdown(animated_title, unsafe_allow_html=True)


page_element = """
<style>
[data-testid="stAppViewContainer"] {
  background-image: url("https://as1.ftcdn.net/v2/jpg/00/45/81/10/1000_F_45811028_gYeeHYNlWubiaBfaOjD5BzbUBm6sNrfu.jpg");
  background-size: cover;
  background-position: center;
}
</style>
"""
st.markdown(page_element, unsafe_allow_html=True)


news = st.text_area(
    label="News",
    placeholder="Enter the news article here...",
    height=250
)


if st.button("Submit"):
    if news.strip() == "":
        st.warning("⚠️ Please enter some news text")
    else:
        with st.spinner("Analyzing news... 🧠"):
            prob = model.predict(preprocessor(news))[0][0]

        label = "🟢 Real News" if prob >= 0.5 else "🔴 Fake News"

        st.markdown(
            f"""
            <div style="
                font-size: 25px;
                font-weight: bold;
                color:red;
                text-align: center;
                background: white;
                padding: 10px;
                border-radius: 10px;">
                {label}
            </div>
            """,
            unsafe_allow_html=True
        )


