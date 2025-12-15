import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download nltk resources (runs once)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -------------------- Text Preprocessing --------------------
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    return " ".join(
        ps.stem(word)
        for word in text
        if word.isalnum() and word not in stop_words
    )

# -------------------- Load Model & Vectorizer --------------------
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# -------------------- Streamlit UI --------------------
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
