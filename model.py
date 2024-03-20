import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import tensorflow.keras.layers as lyrs
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# Define constants
IMG_SIZE = 512
BATCH_SIZE = 32
EPOCHS = 20
DATASET_PATH = "./data"

def build_model():
    model_base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False)
    model_base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = model_base(inputs)
    x = lyrs.GlobalAveragePooling2D()(x)
    x = lyrs.BatchNormalization()(x)
    x = lyrs.Dropout(0.5)(x)
    outputs = lyrs.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_data_generators(data_path, img_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2  # Use 20% of the data for validation
    )

    train_generator = datagen.flow_from_directory(
        directory=data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        directory=data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

def train_model():
    model = build_model()
    train_datagen, val_datagen = get_data_generators(DATASET_PATH, IMG_SIZE, BATCH_SIZE)

    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # checkpointer = ModelCheckpoint('model_best.hdf5', verbose=1, save_best_only=True)
    checkpointer = ModelCheckpoint(
    filepath='model_best.keras',  # Note the lack of file extension
    verbose=1, 
    save_best_only=True
)



    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(train_datagen,
              validation_data=val_datagen,
              epochs=EPOCHS,
              callbacks=[checkpointer, early_stopping])

    # Assuming you meant to load and possibly do something with the model here,
    # you need to use tf.keras.models.load_model to load the model correctly.
    # model = tf.keras.models.load_model('model_best.hdf5')

    return model


# If you want to train the model immediately when the script is run
if __name__ == '__main__':
    model = train_model()
    # Optionally, save the final model
    model.save('model_best')
