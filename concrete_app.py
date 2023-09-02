import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from tensorflow.keras.activations import softmax
import time
import os
import h5py
fig = plt.figure()


with open("customs.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title('Concrete Classifier')

st.markdown("Welcome to this simple web application that concrete into crack and non crack")


def main():
    file_uploaded = st.file_uploader("Choose File", type=["png","jpg","jpeg"])
    class_btn = st.button("Classify")
    if file_uploaded is not None:    
        image = Image.open(file_uploaded)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
    if class_btn:
        if file_uploaded is None:
            st.write("Invalid command, please upload an image")
        else:
            with st.spinner('Model working....'):
                plt.imshow(image)
                plt.axis("off")
                predictions = predict(image)
                time.sleep(1)
                st.success('Classified')
                st.write(predictions)
                st.pyplot(fig)
        


def predict(image):
    #classifier_model = tf.keras.models.load_model(r'/C:/Users/antwi/Downloads/my_mode.h5')
    #classifier_model = "C:/Users/antwi/Downloads/my_mode.h5"
    classifier_model = "modeltrans_dir.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    test_image = image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'Crack',
          'Negative'
          ]
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is {}".format(image_class)
    #result = f"{class_names[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2) } % confidence."
    return result

    
    

if __name__ == "__main__":
    main()
