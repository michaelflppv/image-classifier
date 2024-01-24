import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image

# Задаем параметры обучения
batch_size = 32
epochs = 10

# Создаем генератор изображений для автоматической загрузки и предобработки данных
datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

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

train_mode = False

if train_mode:
    # Создаем модель нейронной сети
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

    # Компилируем модель
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Обучаем модель
    model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

    # Оцениваем модель на тестовых данных
    test_loss, test_acc = model.evaluate(validation_generator)
    print(f'Test accuracy: {test_acc}')

    print("Обучение завершено")

    # Сохранение модели
    model.save('trained_model.keras')

else:
    # Загрузка сохраненной модели
    model = tf.keras.models.load_model('trained_model.keras')

    # Создаем генератор изображений для тестовых данных
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        directory='archive/test',
        target_size=(32, 32),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )

    # Предсказываем классы для изображений в тестовом наборе
    predictions = model.predict(test_generator)

    # Функция для загрузки и обработки входного изображения
    def load_and_preprocess_image(image_pth):
        # Загрузка изображения и изменение размера на 32x32 пикселя
        img = Image.open(image_pth)
        img = img.resize((32, 32))
        img = img.convert('RGB')
        img = np.array(img) / 255.0  # Нормализация значений пикселей
        return img


    # Путь к изображению для тестирования
    image_path = 'archive/test/FAKE/1.jpg'

    # Загрузка и предобработка изображения
    input_image = load_and_preprocess_image(image_path)

    # Порог для классификации (например, 0.5 - если предсказание больше 0.5, то классифицируем как REAL, иначе как FAKE)
    threshold = 0.5

    # Классифицируем изображения
    classified_images = ["REAL" if prediction > threshold else "FAKE" for prediction in predictions]

    # Выводим результаты классификации
    for i, image_path in enumerate(test_generator.filepaths):
        print(f"Image {i + 1}: {classified_images[i]} - {predictions[i][0] * 100:.2f} % (Probability)")
