import os
import emnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical # преобразование target
from logging_info import setup_logging

# Настраиваем логирование
logger = setup_logging()

"""
def take_data(): получение данных
def take_dict(): создание словаря
def preprocessing(): векторизация и нормализация
def model_keras(): обучение модели keras
"""

path = os.path.dirname(os.path.abspath(__file__))  # текущий путь к файлу

# data
def take_data():
    images_train, labels_train = emnist.extract_training_samples('balanced')
    images_test, labels_test = emnist.extract_test_samples('balanced')
    logger.info(f'Modeling.data taken')

    return images_train, labels_train, images_test, labels_test



def take_dict():
    label_path = os.path.join(path, '../emnist-balanced-mapping.txt')
    # Загружаем соответствие лейблов и символов
    with open(label_path, 'r') as f:
        label_mapping = f.readlines()
    # Создаем словарь соответствий
    label_dict = {}
    for entry in label_mapping:
        label, ascii_code = map(int, entry.split())
        label_dict[label] = chr(ascii_code)
        logger.info(f'Modeling.dictionary taken')

    return label_dict



def preprocessing():
    # данные
    images_train, labels_train, images_test, labels_test = take_data()
    # словарь
    label_dict = take_dict()
    num_train_samples = images_train.shape[0]
    num_test_samples = images_test.shape[0]
    # Изменение формы изображений для работы с Keras
    X_train = images_train.reshape(num_train_samples, 28, 28, 1)
    X_test = images_test.reshape(num_test_samples, 28, 28, 1)
    # Нормализация данных
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    # Преобразование меток в категориальный формат
    labels_train_categorical = to_categorical(labels_train, num_classes=len(label_dict))
    labels_test_categorical = to_categorical(labels_test, num_classes=len(label_dict))
    logger.info(f'Modeling.data preprocessed')

    return X_train, X_test, labels_train_categorical, labels_test_categorical



def model_keras():
    # данные для моделирования и теста
    X_train, X_test, labels_train_categorical, labels_test_categorical = preprocessing()
    # Создание модели CNN
    model_k = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(47, activation='softmax')  # 47 - количество классов в EMNIST Balanced
    ])
    # Компиляция модели
    model_k.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Обучение модели
    model_k.fit(X_train, labels_train_categorical, epochs=9, batch_size=32, validation_split=0.1)
    # Оценка модели
    test_loss, test_accuracy = model_k.evaluate(X_test, labels_test_categorical)
    print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')
    # Сохранение модели в формате keras
    # проверяем метрики, с непроходной метрикой модель не записываем
    if test_accuracy < 0.68:
        print('Метрика модели ниже требуемой, модель не записана.')
        logger.warning('Metric is lower than required, the model is not saved')
        return
    model_path = os.path.join(path, 'model.keras')
    model_k.save(model_path)
    logger.info(f'Modeling.model saved')



if __name__ == "__main__":
    model_keras()