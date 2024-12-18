import numpy as np
import os
from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

path = os.path.dirname(os.path.abspath(__file__))  # текущий путь к файлу

# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):
    # Преобразуем строку в массив чисел
    image = np.array(list(map(int, image[1:-1].split(','))))

    # Проверка на правильность размера
    if image.shape != (28 * 28,):  # Убедитесь, что размер соответствует (28*28)
        return {'error': 'Input image must be a flat array of size 784.'}

    pred = model.predict(image)
    logger.info(f'main.Prediction taken: {pred}')

    return pred


static_path = os.path.join(path, '../static')  # путь к папке 'static'
# статические файлы
app.mount('/', StaticFiles(directory=static_path, html=True), name='static')