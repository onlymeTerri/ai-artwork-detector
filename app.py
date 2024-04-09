import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

# Constants
IMG_SIZE = (150, 150)

# Define a function to load models safely
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        #st.write(f"Successfully loaded model from {model_path}")
        #st.write("Model Summary:")
        #model.summary(print_fn=st.write)  # Print model summary
        return model
    except Exception as e:
        st.error(f"Error loading the model from {model_path}: {e}")
        return None

# Define model paths
MODEL_PATHS = {
    "Model 1": "model_mobilenetv2.keras",
    "Model 2": "model_final.keras"
    # Add or update model paths as needed
}

# Attempt to load models and create a radio button for model selection
models = {name: load_model(path) for name, path in MODEL_PATHS.items() if os.path.exists(path)}
if not models:
    st.error("No models loaded successfully. Please check the model paths and file extensions.")
    st.stop()

model_choice = st.sidebar.radio("Choose a model for identification:", list(models.keys()))

# Define a function for image preprocessing and prediction
def predict(model, uploaded_file):
    img = image.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Make sure to match the training preprocessing
    img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch of images

    predictions = model.predict(img_array)
    return predictions[0][0]  # Assuming binary classification: [AI-generated, Real]

# Streamlit UI
st.title("AI-Generated Image Detector")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    selected_model = models[model_choice]
    
    if st.button('Identify'):
        with st.spinner('Analyzing the image...'):
            prediction = predict(selected_model, uploaded_file)
            is_ai_generated = prediction > 0.5  # Adjust threshold as needed

            if is_ai_generated:
                st.error("The image is suspected to be AI-generated.")
            else:
                st.success("The image is suspected to be real.")
