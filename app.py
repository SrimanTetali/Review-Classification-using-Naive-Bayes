
import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

# Load the trained model and fitted TF-IDF vectorizer
@st.cache_resource
def load_artifacts():
    return joblib.load("sentiment_model.pkl")

artifacts = load_artifacts()
nb = artifacts["model"]
tfidf = artifacts["tfidf"]

# Use the same preprocessing settings as your notebook
stop_words = set(stopwords.words("english"))

important_words = {
    "no", "not", "nor", "neither", "never",
    "nothing", "nowhere", "hardly", "barely", "scarcely",
    "but", "however", "although", "though", "yet", "while",
    "very", "too", "so", "really", "quite",
    "extremely", "highly", "completely", "absolutely"
}

stop_words -= important_words
lemmatizer = WordNetLemmatizer()

# Paste the SAME contractions dictionary from your notebook here
contractions = {
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "can't": "cannot",
    "couldn't": "could not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "won't": "will not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "mustn't": "must not",
    "needn't": "need not",
    "i'm": "i am",
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i've": "i have",
    "you've": "you have",
    "we've": "we have",
    "they've": "they have",
    "i'll": "i will",
    "you'll": "you will",
    "we'll": "we will",
    "they'll": "they will",
    "i'd": "i would",
    "you'd": "you would",
    "we'd": "we would",
    "they'd": "they would",
    "what's": "what is",
    "who's": "who is",
    "let's": "let us"
}

def clean_text(text):
    text = str(text).lower()

    for contraction, expansion in contractions.items():
        text = re.sub(
            r"\b" + re.escape(contraction) + r"\b",
            expansion,
            text
        )

    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Streamlit user interface
st.set_page_config(
    page_title="Review Sentiment Classifier",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Review Sentiment Classifier")
st.write(
    "Enter a review below to predict whether its sentiment "
    "is positive or negative."
)

review = st.text_area(
    "Enter your review",
    placeholder="Example: The food was fantastic and the service was excellent!",
    height=150
)

if st.button("Analyze Sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        cleaned_review = clean_text(review)
        review_tfidf = tfidf.transform([cleaned_review])
        prediction = nb.predict(review_tfidf)[0]

        st.subheader("Prediction")

        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")

        with st.expander("View text preprocessing"):
            st.write("**Original review:**")
            st.write(review)
            st.write("**Cleaned review:**")
            st.write(cleaned_review)
