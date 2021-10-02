import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from dogs import unique_breeds
import tensorflow_hub as hub

IMG_SIZE = 224
BATCH_SIZE = 8


def create_model(model="mobilenet_v2_130_224", OUTPUT_SHAPE=120, dropout=True, dropout_rate=0.2):
    Model_urls = {
        "mobilenet_v2_130_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"}
    MODEL_URL = Model_urls[model]
    # Setup the model layers
    if dropout:
        model = tf.keras.Sequential([
            hub.KerasLayer(MODEL_URL),  # Layer 1 (input layer)
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                                  activation="softmax")  # Layer 2 (output layer)
        ])
    else:
        model = tf.keras.Sequential([
            hub.KerasLayer(MODEL_URL),  # Layer 1 (input layer)
            tf.keras.layers.Dense(units=OUTPUT_SHAPE,
                                  activation="softmax")  # Layer 2 (output layer)
        ])

    model = tf.keras.models.clone_model(model)

    return model


def create_model_best(model_name="mobilenet_v2_130_224",  learning_rate=0.001, dropout_rate=0.0):

    model = create_model(model_name, dropout_rate=dropout_rate)
    model.compile(
        # Our model wants to reduce this (how wrong its guesses are)
        loss=tf.keras.losses.CategoricalCrossentropy(),
        # A friend telling our model how to improve its guesses
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
        metrics=["accuracy"]  # We'd like this to go up
    )

    # Build the model
    INPUT_SHAPE = [None, 224, 224, 3]
    # Let the model know what kind of inputs it'll be getting
    model.build(INPUT_SHAPE)
    return model


def process(x):
    try:
        x = x / 255.0
        image = tf.image.resize(x, size=[IMG_SIZE, IMG_SIZE])
        image = tf.expand_dims(
            image, axis=0, name=None
        )
        # model = tf.keras.models.load_model("dogs_classifier.h5")
        model = create_model_best()
        model.load_weights("mobilenet_v2_130_224_weights_final_3.h5")
        pred = model.predict(image)
        if(max(pred[0]) * 100 > 59):
            dog = unique_breeds[np.argmax(pred[0])]
            # return (dog, max(pred[0]) * 100)
            return f"it is {dog} with the chances of {round( max(pred[0]) * 100)} %"
        else:
            return "not sure......."

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
        pred = process(dog)

        st.write(pred)

    except Exception as e:
        print(e)


# streamlit run app.py
# streamlit hello
# https://towardsdatascience.com/image-classification-of-uploaded-files-using-streamlits-killer-new-feature-7dd6aa35fe0
# https://towardsdatascience.com/a-quick-tutorial-on-how-to-deploy-your-streamlit-app-to-heroku-874e1250dadd
# https://share.streamlit.io/mb16biswas/dog_classifier_web_app/main/app.py

