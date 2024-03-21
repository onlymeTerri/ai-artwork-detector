import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Assuming helper_functions.py is in the same directory and contains the necessary functions
from helper_functions import pre_process_image

# Load the model (make sure the path is correct relative to where you're running the Streamlit app)
MODEL_PATH = 'model_best.keras'
model = tf.keras.models.load_model(MODEL_PATH)

def classify_image(image, model):
    # Process the uploaded image
    img = pre_process_image(image, IMG_SIZE)
    img_array = tf.expand_dims(img, 0)  # Create a batch containing the image
    
    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    
    # Map the prediction to the respective class names
    class_names = ['AI_GENERATED', 'REAL']
    predicted_class_name = class_names[predicted_class[0]]
    
    return predicted_class_name

def main():
    st.title('AI Vs. Human Artwork Detector')
    st.text('This is the web app for the AI artwork detector')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Convert the file to an image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Display the image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Classify the image
        prediction = classify_image(image, model)
        
        # Display the classification result
        st.write(f"Model Prediction: {prediction}")

if __name__ == "__main__":
    main()
