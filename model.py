import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as lyrs
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.regularizers import l2

# Constants
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 150
SCENERY_TYPES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
DATASET_PATH = "./data_cleaned"
CLASS_NAMES = ['ai_generated', 'real']

# Function to load images from a directory
def load_images_from_directory(directory, label, target_size=(IMG_SIZE, IMG_SIZE)):
    images = []
    labels = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.lower().endswith(".jpg"):
                img_path = os.path.join(root, filename)
                img = image.load_img(img_path, target_size=target_size)
                img_array = image.img_to_array(img)
                img_array = preprocess_input(img_array)  # Preprocess for MobileNetV2
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

def create_datasets(scenery_types, class_names, dataset_path, target_size=(IMG_SIZE, IMG_SIZE), test_size=0.2):
    all_train_images = []
    all_train_labels = []
    all_test_images = []
    all_test_labels = []
    test_datasets = {}

    for scenery_type in scenery_types:
        for class_name in class_names:
            images, labels = load_images_from_directory(
                os.path.join(dataset_path, class_name, scenery_type),
                label=class_names.index(class_name),
                target_size=target_size
            )
            # Split the images into training and testing
            train_images, test_images, train_labels, test_labels = train_test_split(
                images, labels, test_size=test_size, random_state=42
            )
            # Extend the lists that will be turned into TensorFlow datasets
            all_train_images.extend(train_images)
            all_train_labels.extend(train_labels)
            all_test_images.extend(test_images)
            all_test_labels.extend(test_labels)
            # Store the test set for evaluation later
            test_datasets[scenery_type] = (test_images, test_labels)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((np.array(all_train_images), np.array(all_train_labels)))
    train_dataset = train_dataset.shuffle(buffer_size=len(all_train_images)).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((np.array(all_test_images), np.array(all_test_labels)))
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset, test_datasets

def build_model():
    model_base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)
    model_base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = model_base(inputs, training=False)  # Ensures the base is frozen
    x = lyrs.GlobalAveragePooling2D()(x)
    x = lyrs.BatchNormalization()(x)
    x = lyrs.Dropout(0.5)(x)
    outputs = lyrs.Dense(1, activation="sigmoid")(x)  # Output for binary classification

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, train_dataset, validation_dataset, epochs=EPOCHS):
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )

    # Evaluate the model on the validation dataset
    val_loss, val_accuracy = model.evaluate(validation_dataset)
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}%')

    
    return history

def evaluate_on_scenery(model, images, labels):
    predictions = model.predict(images, batch_size=BATCH_SIZE)
    predicted_classes = (predictions > 0.5).astype(int)
    f1 = f1_score(labels, predicted_classes)
    precision = precision_score(labels, predicted_classes)
    recall = recall_score(labels, predicted_classes)
    return f1, precision, recall

def print_dataset_distribution(test_datasets):
    for scenery_type, dataset in test_datasets.items():
        print(f"{scenery_type}: {len(dataset[0])} images")

def main():
    train_dataset, validation_dataset, test_datasets = create_datasets(
        SCENERY_TYPES, CLASS_NAMES, DATASET_PATH, (IMG_SIZE,IMG_SIZE)
    )
    
    print("Test dataset distribution:")
    print_dataset_distribution(test_datasets)

    # Build and compile the model
    model = build_model()

    # Train and evaluate the model
    history = train_and_evaluate_model(model, train_dataset, validation_dataset, epochs=EPOCHS)

    # Evaluate model performance on the test set for each scenery type
    for scenery, (images, labels) in test_datasets.items():
        images = np.array(images)  
        labels = np.array(labels)  
        f1, precision, recall = evaluate_on_scenery(model, images, labels)
        print(f"{scenery} - F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # Save the model
    model_filename = 'model_mobilenetv2.keras'
    if os.path.isfile(model_filename):  
        os.remove(model_filename)  

    model.save(model_filename)  
    print(f"Model saved as {model_filename}")
    print(model.summary())

if __name__ == '__main__':
    main()
