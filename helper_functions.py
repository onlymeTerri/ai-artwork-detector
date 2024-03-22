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

    path = os.path.join(root_path, class_name)
    random_filename = random.choice(os.listdir(path))
    img_path = os.path.join(path, random_filename)
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(f"Class: {class_name}\nFilename: {random_filename}")
    plt.axis('off')
    plt.show()

def pre_process_image(image, img_size=IMG_SIZE):
    
    image_tensor = tf.convert_to_tensor(np.array(image), dtype=tf.float32)
    image_resized = tf.image.resize(image_tensor, [img_size, img_size])
    image_normalized = image_resized / 255.0
    
    return image_normalized

def supervised_metrics(y_true, y_pred):

    print("Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))
    print("F1 Score:", f1_score(y_true, y_pred, average='binary'))
    print("Recall:", recall_score(y_true, y_pred, average='binary'))
    print("Precision:", precision_score(y_true, y_pred, average='binary'))

def plot_confusion_matrix(y_true, y_pred, class_names, figsize=(10, 7)):

    cm = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predictions')
    plt.ylabel('Actual')
    plt.show()



