import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- NLTK FIX FOR STREAMLIT CLOUD --------------------
nltk.data.path.append("/home/appuser/nltk_data")

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# -------------------- SETUP --------------------
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------- TEXT PREPROCESSING --------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    return " ".join(
        ps.stem(word)
        for word in text
        if word.isalnum() and word not in stop_words
    )

# -------------------- LOAD MODEL & VECTORIZER --------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------- STREAMLIT UI --------------------
st.title("📩 SMS / Email Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display result
        if result == 1:
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")
