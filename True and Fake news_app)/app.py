import streamlit as st
import numpy as np
import joblib

# load trained model vectorizer
model  = joblib.load('model/model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

st.title('True and Fake News Prediction Model')

# Display your first image
st.image(r'C:\Users\Opsydee\Downloads\Fake and true.jpeg', caption='Fake vs True News', use_container_width=True)

# Display the second image
# st.image(r'C:\Users\Opsydee\Downloads\news.jpeg', caption='News Image', use_container_width=True)

# Input feature
text_input = st.text_area("Enter the news text:", key ="unique_text_input")

# prepare input data
if text_input:
    # Transform the input text using the vectorizer
    transformed_input =vectorizer.transform([text_input])

    # make Prediction using the model
    prediction = model.predict(transformed_input)

    # display the result
    if prediction ==1:
        st.write('Prediction: True News')
    else:
        st.write('Prediction: Fake News')