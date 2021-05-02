import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from dogs import unique_breeds


IMG_SIZE = 224
BATCH_SIZE = 32


def process(x):
    try:
        x = x / 255.0
        image = tf.image.resize(x, size=[IMG_SIZE, IMG_SIZE])
        image = tf.expand_dims(
            image, axis=0, name=None
        )
        model = tf.keras.models.load_model('dogs_classifier.h5')
        pred = model.predict(image)
        if(max(pred[0]) * 100 > 59):
            dog = unique_breeds[np.argmax(pred[0])]
            return (dog, max(pred[0]) * 100)

    except Exception as e:
        print(e)


st.title('Dog classifier..')

st.write("upload the dog's  image our deep learning model will predict the breed")

dog = st.file_uploader("choose an image", type="jpg")


if(dog is not None):

    try:

        dog = Image.open(dog)
        dog = np.array(dog)
        st.image(dog, caption='Uploaded Image.', use_column_width="always")
        name, prob = process(dog)
        prob = round(prob)
        st.write(f"it is {name} with the probability of {prob} %")

    except Exception as e:
        print(e)


# streamlit run app.py
# streamlit hello
# https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0
