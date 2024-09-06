import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import pickle

# Load the trained model
model = tf.keras.models.load_model('resnet_model_dupli')

# Load the training history
with open('resnet_training_history_dupli.pkl', 'rb') as file:
    history = pickle.load(file)

# Function to preprocess the image for prediction
def preprocess_image(image):
    try:
        # Resize and convert to RGB if image is not in RGB mode
        img = image.convert('RGB').resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Streamlit app
st.title('Orthodontic Treatment ')
st.write('Upload an image for prediction')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = preprocess_image(image)

    if img_array is not None:
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        class_prob = prediction[0][predicted_class]

        # Define classes
        classes = ['Dental Braces is not needed', 'Dental Braces is needed']

        # Display the prediction result
        st.subheader('Prediction Result:')
        st.write(f'Predicted class: {predicted_class} - {classes[predicted_class]}')
        st.write(f'Probability: {class_prob:.2f}')

        # Display training history
        #st.subheader('Training History:')
        #st.write('Training Accuracy:', history['accuracy'])
        #st.write('Validation Accuracy:', history['val_accuracy'])
