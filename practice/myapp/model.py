import os
import numpy as np
from tensorflow import keras

"""
Описание класса Model
def predict()
"""

path = os.path.dirname(os.path.abspath(__file__))  # текущий путь к файлу

from myapp.logging_info import setup_logging
# Настраиваем логирование
logger = setup_logging()


class Model:
    def __init__(self):
        '''
        Модель
        Словарь соответствий
        '''
        model_path = os.path.join(path, 'model.keras')
        self.model = keras.models.load_model(model_path)
        logger.info(f'Model.Model loaded')


        label_path = os.path.join(path, '../emnist-balanced-mapping.txt')
        with open(label_path, 'r') as f:
            label_mapping = f.readlines()

            # Создаем словарь соответствий
            self.label_dict = {}
            for entry in label_mapping:
                label, ascii_code = map(int, entry.split())
                self.label_dict[label] = chr(ascii_code)
                logger.info(f'Model.Dictionary loaded')




    def predict(self, x):
        '''
        Parameters
        ----------
        x : np.ndarray
        Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
        Символ-предсказание
        '''
        # Подготовка данных
        x = x.reshape(1, 28, 28, 1)  # Добавляем размерность канала
        x = x.astype('float32') / 255.0  # Нормализация данных
        predict = self.model.predict(x)
        # Получаем индекс максимального значения (предсказанный класс)
        predicted_class = np.argmax(predict, axis=1)[0]

        # Получение символа из словаря label_dict
        pred = self.label_dict[predicted_class]
        logger.info(f'Model.Prediction taken: {pred}')
        print(pred)

        return {'prediction': pred}


if __name__ == "__main__":
    model = Model()