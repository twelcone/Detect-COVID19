import streamlit as st
import pandas as pd
from img_classification import *

st.title("Detecting COVID-19 / Pneumonia through chest x-ray image")
st.markdown("University of Information and Technology - Vietnam National University")
st.header("Upload Image")
st.text("Description: Upload a chest x-ray image for detecting as COVID19, Pneumonia or Normal")

uploaded_file = st.file_uploader("Choose an image ...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image.', use_column_width=True)
    
    buffer = io.BytesIO()
    image.save(buffer, "png")
    buffer.seek(0)
    bg_image = buffer
    
    with st.spinner("Progressing..."):
        prob = prediction(bg_image)

    st.header("Result")

    if np.argmax(prob) == 0:
        st.write("Class: COVID-19")
    elif np.argmax(prob) == 1:
        st.write("Class: Normal")
    else: 
        st.write("Class: Pneumonia")
    
    d = {'Class': ["COVID-19", "Normal", "Pneumonia"], 
         'Probability (%)': [ "%.2f" % (prob[0]*100), "%.2f" % (prob[1]*100), "%.2f" % (prob[2]*100)]}
    table = pd.DataFrame(data=d)
    st.write("Probability:")
    st.table(d)