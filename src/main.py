import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

def create_model(input_shape=(32, 32, 3)):
    """Create and return a CNN model for binary image classification."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),  # Proper input layer specification
        layers.Conv2D(32, (3, 3), activation='relu'),
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
    return model

def train_model(batch_size=32, epochs=8):
    """Train the model and save it."""
    # Generate data for training and validation
    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        rotation_range=20,      # Add data augmentation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    train_generator = datagen.flow_from_directory(
        directory='archive/train',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        directory='archive/train',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    # Create model
    model = create_model()

    # Add early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f'Test accuracy: {test_acc:.4f}')

    # Save the model
    model.save('trained_model.keras')
    print("Model saved as 'trained_model.keras'")

    return model, history

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """Load and preprocess an image for prediction."""
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def test_model(model_path='trained_model.keras'):
    """Load a trained model and use it to classify an image."""
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Path to the image to test
    image_path = input('Enter the path to the image you want to test: ')

    if not os.path.isfile(image_path):
        print("Invalid path, please enter a valid path to an image.")
        return

    # Load and preprocess the image
    input_image = load_and_preprocess_image(image_path)
    if input_image is None:
        return

    # Predict the class of the image
    prediction = model.predict(input_image, verbose=0)

    # Define the threshold for classification
    threshold = 0.5

    # Classify the image
    classification = "REAL" if prediction[0][0] > threshold else "FAKE"
    confidence = prediction[0][0] if prediction[0][0] > threshold else 1 - prediction[0][0]

    # Print the result
    print(f"Classification: {classification}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Raw prediction value: {prediction[0][0]:.4f}")

def main():
    """Main function to run the program."""
    # Set memory growth to avoid memory allocation errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Error setting memory growth: {e}")

    # Define hyperparameters
    batch_size = 32
    epochs = 10

    while True:
        activation_mode = input('Select mode (train/test/exit): ').lower()

        if activation_mode == 'train':
            train_model(batch_size, epochs)
        elif activation_mode == 'test':
            test_model()
        elif activation_mode == 'exit':
            print("Exiting program.")
            break
        else:
            print('Invalid mode, please select "train", "test", or "exit"')

if __name__ == "__main__":
    main()