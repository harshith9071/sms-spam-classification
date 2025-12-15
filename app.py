# import streamlit as st
# import pickle
# import string
# import nltk

# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # Download nltk resources (runs once)
# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# # -------------------- Text Preprocessing --------------------
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     return " ".join(
#         ps.stem(word)
#         for word in text
#         if word.isalnum() and word not in stop_words
#     )

# # -------------------- Load Model & Vectorizer --------------------
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# # -------------------- Streamlit UI --------------------
# st.title("📩 SMS / Email Spam Classifier")

# input_sms = st.text_area("Enter the message")

# if st.button("Predict"):
#     if input_sms.strip() == "":
#         st.warning("Please enter a message")
#     else:
#         # Preprocess
#         transformed_sms = transform_text(input_sms)

#         # Vectorize
#         vector_input = tfidf.transform([transformed_sms])

#         # Predict
#         result = model.predict(vector_input)[0]

#         # Display result
#         if result == 1:
#             st.error("🚨 Spam Message")
#         else:
#             st.success("✅ Not Spam")


#############################################################################

# import streamlit as st
# import pickle
# import string
# import nltk

# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# # -------------------- PAGE CONFIG (UI) --------------------
# st.set_page_config(
#     page_title="Spam Classifier",
#     page_icon="📩",
#     layout="centered"
# )

# # -------------------- NLTK FIX FOR STREAMLIT CLOUD --------------------
# nltk.data.path.append("/home/appuser/nltk_data")

# try:
#     nltk.data.find('tokenizers/punkt')
# except LookupError:
#     nltk.download('punkt')

# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('stopwords')

# # -------------------- SETUP --------------------
# ps = PorterStemmer()
# stop_words = set(stopwords.words('english'))

# # -------------------- TEXT PREPROCESSING --------------------
# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     return " ".join(
#         ps.stem(word)
#         for word in text
#         if word.isalnum() and word not in stop_words
#     )

# # -------------------- LOAD MODEL & VECTORIZER --------------------
# tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
# model = pickle.load(open('model.pkl', 'rb'))

# # -------------------- SIDEBAR (UI ONLY) --------------------
# st.sidebar.title("ℹ️ About")
# st.sidebar.markdown(
#     """
#     **SMS / Email Spam Classifier**

#     - Built using **Machine Learning**
#     - Model: **Multinomial Naive Bayes**
#     - Vectorization: **TF-IDF**
#     - Hosted on **Streamlit Cloud**
    
#     🔍 Paste any message to check whether it is **Spam** or **Not Spam**.
#     """
# )

# st.sidebar.markdown("---")
# st.sidebar.markdown("👨‍💻 **Developed by:** *Harshith S*")

# # -------------------- MAIN UI --------------------
# st.markdown(
#     "<h1 style='text-align: center;'>📩 SMS / Email Spam Classifier</h1>",
#     unsafe_allow_html=True
# )

# st.markdown(
#     "<p style='text-align: center; color: gray;'>Detect spam messages instantly using Machine Learning</p>",
#     unsafe_allow_html=True
# )

# st.markdown("---")

# input_sms = st.text_area(
#     "✉️ Enter your message below:",
#     height=150,
#     placeholder="Type or paste the SMS / Email message here..."
# )

# col1, col2, col3 = st.columns([1, 2, 1])

# with col2:
#     predict_btn = st.button("🔍 Predict", use_container_width=True)

# # -------------------- PREDICTION OUTPUT --------------------
# if predict_btn:
#     if input_sms.strip() == "":
#         st.warning("⚠️ Please enter a message to classify.")
#     else:
#         transformed_sms = transform_text(input_sms)
#         vector_input = tfidf.transform([transformed_sms])
#         result = model.predict(vector_input)[0]

#         st.markdown("---")

#         if result == 1:
#             st.error("🚨 **This message is classified as SPAM**")
#         else:
#             st.success("✅ **This message is NOT SPAM**")

# # -------------------- FOOTER --------------------
# st.markdown("---")
# st.markdown(
#     "<p style='text-align: center; font-size: 12px; color: gray;'>Made with ❤️ using Streamlit & Machine Learning</p>",
#     unsafe_allow_html=True
# )


import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -------------------- PAGE CONFIG (UI) --------------------
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# -------------------- NLTK FIX FOR STREAMLIT CLOUD (FINAL FIX) --------------------
nltk.data.path.append("/home/appuser/nltk_data")
nltk.data.path.append("/home/adminuser/nltk_data")

def ensure_nltk_resource(resource_path, download_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(download_name)

# 🔑 REQUIRED resources (punkt_tab IS THE KEY)
ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")
ensure_nltk_resource("corpora/stopwords", "stopwords")

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

# -------------------- SIDEBAR (UI ONLY) --------------------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown(
    """
    **SMS / Email Spam Classifier**

    - Built using **Machine Learning**
    - Model: **Multinomial Naive Bayes**
    - Vectorization: **TF-IDF**
    - Hosted on **Streamlit Cloud**
    
    🔍 Paste any message to check whether it is **Spam** or **Not Spam**.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("👨‍💻 **Developed by:** *Harshith S*")

# -------------------- MAIN UI --------------------
st.markdown(
    "<h1 style='text-align: center;'>📩 SMS / Email Spam Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>Detect spam messages instantly using Machine Learning</p>",
    unsafe_allow_html=True
)

st.markdown("---")

input_sms = st.text_area(
    "✉️ Enter your message below:",
    height=150,
    placeholder="Type or paste the SMS / Email message here..."
)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

# -------------------- PREDICTION OUTPUT --------------------
if predict_btn:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.markdown("---")

        if result == 1:
            st.error("🚨 **This message is classified as SPAM**")
        else:
            st.success("✅ **This message is NOT SPAM**")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 12px; color: gray;'>Made with ❤️ using Streamlit & Machine Learning</p>",
    unsafe_allow_html=True
)
