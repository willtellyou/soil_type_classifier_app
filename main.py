import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib
import cv2


st.title("Soil Type Classifier")
file_uploader = st.file_uploader("Upload Files", type=['png', 'jpg'])


def getPrediction(filename):
    classes = ['clay', 'loam', 'loamy sand', 'sand', 'sandy loam']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    img = Image.open(filename)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    # Load model
    SIZE = (256,256)
    img = ImageOps.fit(img, SIZE, Image.ANTIALIAS)
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    input_img = np.expand_dims(img, axis=0)  # Expand dims so the input is (num images, x, y, c)
    input_img = input_img / 255.0
    # img_path = 'static/images/' + filename
    loaded_model1 = load_model('C:\\Users\\dell\\Downloads\\soil-type-classifier\\model1.h5')
    loaded_model2 = joblib.load('C:\\Users\\dell\\Downloads\\soil-type-classifier\\model2.joblib')
    VGG_model = loaded_model1
    classifier = loaded_model2


    input_img_feature = VGG_model.predict(input_img)
    input_img_features = input_img_feature.reshape(input_img_feature.shape[0], -1)
    prediction_RF = classifier.predict(input_img_features)[0]
    prediction_RF = le.inverse_transform([prediction_RF])
    # print("Diagnosis is:", prediction_RF)
    return prediction_RF


if file_uploader is  None:
    st.text("Please upload an image file")
else:
    label=getPrediction(file_uploader)
    st.write("Type of soil in the image is:")
    st.write(label)