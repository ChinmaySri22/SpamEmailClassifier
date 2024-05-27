import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import string

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    y = []
    for i in tokens:
        if i.isalnum():
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(ps.stem(i))

    return " ".join(y)

with open('.venv/vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)
with open('.venv/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


st.set_page_config(page_title="Spam Mail Guard")

st.title('SPAM GUARD: Spam Mail Detector')


input_sms = st.text_input('Enter Your Message')

# Add a submit button
submit_button = st.button('Submit')

# Check if the submit button is clicked
if submit_button:
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header('Spam, It is a SPAM mail')
    else:
        st.header('Ham, It is NOT A SPAM mail')
