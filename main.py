import os

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# Define hyperparameters
batch_size = 32
epochs = 8

# Generate data for training and validation
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    directory='directory/train',
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory='directory/train',
    target_size=(32, 32),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

activation_mode = input('Select training or testing mode ("train" or "test"): ')

if activation_mode == 'train':
    # Create a neural network model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f'Test accuracy: {test_acc}')

    print("Evaluation complete")

    # Сохранение модели
    model.save('trained_model.keras')

elif activation_mode == 'test':
    # Load the trained model
    model = tf.keras.models.load_model('trained_model.keras')

    # Create a data generator for the test set
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        directory='archive/test',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    # Function to load and preprocess an image
    def load_and_preprocess_image(image_pth):
        # Load the images of the test set
        img = Image.open(image_pth)
        img = img.resize((32, 32))
        img = img.convert('RGB')
        img = np.array(img) / 255.0  # Normalize the image
        return img


    # Path to the image to test
    image_path = input('Paste the path to the image/images you want to test: ')

    if not os.path.isfile(image_path):
        print("Invalid path, please enter a valid path to an image.")

    # Load and preprocess the image
    input_image = load_and_preprocess_image(image_path)
    input_image = np.expand_dims(input_image, axis=0)

    # Predict the class of the image
    prediction = model.predict(input_image)

    # Define the threshold for classification
    threshold = 0.5

    # Classify the image
    classified_image = "REAL" if prediction > threshold else "FAKE"

    # Print the result
    print(f"Image: {classified_image} - {prediction[0][0] * 100:.2f} % (Probability)")

else:
    print('Invalid mode, please select "train" or "test"')
