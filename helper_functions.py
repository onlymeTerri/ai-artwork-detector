import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import seaborn as sns

IMG_SIZE = 512

def view_random_image(root_path, class_name):
    """
    Displays a random image from a specified class folder.
    """
    path = os.path.join(root_path, class_name)
    random_filename = random.choice(os.listdir(path))
    img_path = os.path.join(path, random_filename)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(f"Class: {class_name}\nFilename: {random_filename}")
    plt.axis('off')
    plt.show()

def pre_process_image(image_path, img_size=IMG_SIZE):
    """
    Pre-processes an image to the format required by the model.
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img /= 255.0  # Normalize to [0,1] as done during training
    return img

def supervised_metrics(y_true, y_pred):
    """
    Prints common metrics for evaluating classification models.
    """
    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("F1 Score:", f1_score(y_true, y_pred, average='binary'))
    print("Recall:", recall_score(y_true, y_pred, average='binary'))
    print("Precision:", precision_score(y_true, y_pred, average='binary'))

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 7)):
    """
    Plots a confusion matrix using seaborn.
    """
    cm = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.show()

def display_prediction(model, image_path, class_names=['AI_GENERATED', 'REAL']):
    """
    Pre-processes an uploaded image and uses the model to predict the class.
    Displays the image along with the prediction.
    """
    img = pre_process_image(image_path)
    img_array = tf.expand_dims(img, 0)  # Model expects a batch

    prediction = model.predict(img_array)
    predicted_class = class_names[int(tf.round(prediction)[0][0])]  # Binary classification

    plt.imshow(img)
    plt.title(f"Model Prediction: {predicted_class}")
    plt.axis('off')
    plt.show()
