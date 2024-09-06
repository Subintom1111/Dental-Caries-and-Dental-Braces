import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle

# Load the trained model for Dental Braces prediction
model_braces = tf.keras.models.load_model('resnet_model_dupli')
with open('resnet_training_history_dupli.pkl', 'rb') as file:
    history_braces = pickle.load(file)

# Load the trained model for Dental Caries detection
model_caries = tf.keras.models.load_model('resnet_model_dupli_Caries')
with open('resnet_training_history_dupli_Caries.pkl', 'rb') as file:
    history_caries = pickle.load(file)

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

# Initialize session_state
if 'previous_option' not in st.session_state:
    st.session_state.previous_option = None
    st.session_state.uploaded_image = None

# Streamlit app
st.title('Dental Image Analysis')

# Sidebar selection for different functionalities
option = st.sidebar.selectbox('Select Analysis Type', ('Dental Braces Necessity', 'Dental Caries Detection'))

# Create empty space for image display
uploaded_image_placeholder = st.empty()

# Clear uploaded image if option changes
if st.session_state.previous_option != option:
    st.session_state.uploaded_image = None

st.session_state.previous_option = option

if option == 'Dental Braces Necessity':
    st.write('## Dental Braces Necessity Prediction')
    st.write('Upload an image to predict whether dental braces are needed or not.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        uploaded_image_placeholder.image(st.session_state.uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(st.session_state.uploaded_image)

        if img_array is not None:
            # Make prediction
            prediction = model_braces.predict(img_array)
            predicted_class = np.argmax(prediction)
            class_prob = prediction[0][predicted_class]

            # Define classes
            classes = ['Dental Braces is not needed', 'Dental Braces is needed']

            # Display the prediction result
            st.subheader('Prediction Result:')
            st.write(f'Predicted class: {predicted_class} - {classes[predicted_class]}')
            st.write(f'Probability: {class_prob:.2f}')

            # Display training history (commented out)
            #st.subheader('Training History:')
            #st.write('Training Accuracy:', history_braces['accuracy'])
            #st.write('Validation Accuracy:', history_braces['val_accuracy'])

elif option == 'Dental Caries Detection':
    st.write('## Dental Caries Detection')
    st.write('Upload an image to detect the presence of dental caries.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.session_state.uploaded_image = Image.open(uploaded_file)
        uploaded_image_placeholder.image(st.session_state.uploaded_image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        img_array = preprocess_image(st.session_state.uploaded_image)

        if img_array is not None:
            # Make prediction
            prediction = model_caries.predict(img_array)
            predicted_class = np.argmax(prediction)
            class_prob = prediction[0][predicted_class]

            # Define classes
            classes = ['Dental Caries has not been detected in the dental image.', 'Dental Caries has been detected in the dental image.']

            # Display the prediction result
            st.subheader('Prediction Result:')
            st.write(f'Predicted class: {predicted_class} - {classes[predicted_class]}')
            st.write(f'Probability: {class_prob:.2f}')

            # Display training history (commented out)
            #st.subheader('Training History:')
            #st.write('Training Accuracy:', history_caries['accuracy'])
            #st.write('Validation Accuracy:', history_caries['val_accuracy'])
