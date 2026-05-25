import streamlit as st
import pickle
import re
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

st.title("Sentiment Analysis")
st.markdown("### The app checks whether a review is a positive or negative one")

review = st.text_area("Enter your review")

def cleaning(text):
    text = re.sub('[^a-zA-Z]', " ", text)
    text = text.lower()
    text = " ".join(word for word in text.split() if word not in stop_words)

    return text

cleaned_review = cleaning(review)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

if st.button("Predict"):
    vec_review = vectorizer.transform([cleaned_review])

    prediction = model.predict(vec_review)

    if prediction[0] == 1:
        st.success("Your review is a Positive one")
    else:
        st.warning("Your review is a Negative one")