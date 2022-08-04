# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RaCWoMZTXcWda7CC_qwJ9ekY3ciRdYMz
"""

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import pickle

"""**prediction class**"""

def getPrediction(filename):
    
    classes = ['clay','loam','loamy sand','sand','sandy loam']
    le = LabelEncoder()
    le.fit(classes)
    le.inverse_transform([2])
    
    
    #Load model
    img_path = 'static/images/' + filename
    pickle.load(open("model/resnet.pkl",'rb'))
    
    SIZE = 224 #Resize to same size as training images
    img = np.asarray(Image.open(img_path).resize((SIZE,SIZE)))
    
    img = img/255.      #Scale pixel values
    
    img = np.expand_dims(img, axis=0)  #Get it tready as input to the network       
    
    pred = my_model.predict(img) #Predict                    
    
    #Convert prediction to class name
    pred_class = le.inverse_transform([np.argmax(pred)])[0]
    print("Diagnosis is:", pred_class)
    return pred_class
