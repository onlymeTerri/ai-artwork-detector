import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import torchvision.models as models
from torchmetrics import F1Score
import torch.nn as nn
# Constants
IMG_SIZE = (150, 150)

class BinaryClassificationResNet50(pl.LightningModule):
    def __init__(self, num_classes=2, learning_rate=1e-3):
        super(BinaryClassificationResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, num_classes)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        # Define F1 score metric for each phase
        self.train_f1 = F1Score(task='binary', average='macro')
        self.val_f1 = F1Score(task='binary', average='macro')
        self.test_f1 = F1Score(task='binary', average='macro')

    def forward(self, x):
        return self.resnet50(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.train_f1(preds, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.val_f1(preds, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.test_f1(preds, labels)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    # Class definition as provided...

# Function to load the model from checkpoint
@st.cache(allow_output_mutation=True)
def load_modelResNet(checkpoint_path):
    model = BinaryClassificationResNet50.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze() 
    return model

# Define a function to load models safely
def load_model(model_path):
    try:
        # Check the file extension and load appropriately
        _, file_extension = os.path.splitext(model_path)
        if file_extension == '.keras':
            model = tf.keras.models.load_model(model_path)
        elif file_extension == '.ckpt':
            model = load_modelResNet(model_path)
        elif file_extension == '.pth':
            device = torch.device('cpu')
            model = models.efficientnet_b0(pretrained=False)  # Initialize the model structure
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, 2)  # Adjust the classifier as per your setup

            model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model weights
            model.eval() 
        else:
            st.error("Unsupported file extension for model loading.")
            return None
        
        st.write(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        st.error(f"Error loading the model from {model_path}: {e}")
        return None

# Image preprocessing function
def preprocess_image(image_file):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_file)
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, uploaded_file, model_choice):
    preprocessed_image = preprocess_image(uploaded_file)
    if model_choice == 'MobileNetV2': 
        img = image.load_img(uploaded_file, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)  # Make sure to match the training preprocessing
        img_array = np.expand_dims(img_array, axis=0)  # Model expects a batch of images
        predictions = model.predict(img_array)
        st.write(predictions)
        return predictions[0][0]  # Assuming binary classification: [AI-generated, Real]
    
    if model_choice == 'ResNet50':
        prediction = model(preprocessed_image)
        pred_label = torch.argmax(prediction, dim=1)
        st.write(f"Prediction: {'Class 1' if pred_label.item() == 1 else 'Class 0'}")
        prediction = 1 if pred_label.item() == 1 else 0
        return prediction
    
    if model_choice == 'EfficientNet': 
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize the image to 224x224 (as per EfficientNet's requirements)
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # Normalize with the same parameters used during training
        ])
        img_path = uploaded_file # Specify the path to your image
        image = Image.open(img_path).convert('RGB')
        device = torch.device('cpu') 
        image = transform(image)
        image = image.unsqueeze(0)  
        image = image.to(device)  
        with torch.no_grad():  
            outputs = model(image)  
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Apply softmax to convert to probabilities
            predicted_class = probabilities.argmax(dim=1)  # Get the class with the highest probability
        st.write(f"Predicted class index: {'1' if predicted_class.item() == 1 else '0'}")
        return predicted_class.item()

     


# Define model paths
MODEL_PATHS = {
    "MobileNetV2": "model_mobilenetv2.keras",
    "ResNet50": "best_model.ckpt",
    "EfficientNet": "best_model_eff.pth"
    # Add or update model paths as needed
}

# Attempt to load models and create a radio button for model selection
models = {name: load_model(path) for name, path in MODEL_PATHS.items()}
if not models:
    st.error("No models loaded successfully. Please check the model paths and file extensions.")
    st.stop()

model_choice = st.sidebar.radio("Choose a model for identification:", list(models.keys()))

#UI Componnet
st.title("AI-Generated Image Detector")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


if uploaded_file:
    st.image(uploaded_file, caption='Uploaded Image', width=200)  # Fix image width to 150
    selected_model = models[model_choice]

    if st.button('Identify'):
        with st.spinner('Analyzing the image...'):
            prediction = predict(selected_model, uploaded_file, model_choice)
            # st.write(model_choice)
            is_ai_generated = prediction < 0.5  # Adjust threshold as needed
            if is_ai_generated:
                st.error("The image is suspected to be AI-generated.")
            else:
                st.success("The image is suspected to be real.")
          
