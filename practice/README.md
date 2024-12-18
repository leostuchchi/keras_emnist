# Распознавание рукописных символов.

## 1. Описание решения
_Проект компьютерного зрения, выводит предсказанные значения рукописных букв и цифр:_
- _Задача многоклассовой классификации_
- _Для обучения модели применялись данные EMNIST. Набор данных представляет собой сбалансированное подмножество рукописных букв и цифр, в котором все классы представлены в равных количествах._
- _Итоговой моделью является tensorflow.keras с параметрами:_
- _Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(47, activation='softmax'_
- _Метрика итоовой модели на тестовых данных Test accuracy: 0.87_


## 2. Установка и запуск сервиса

_Для запуска сервиса необходимо клонировать репозиторий и создать docker-образ из Dockerfile, сервис с уже обученной моделью запустится в docker-контейнере в своем виртуальном окружении, перейдите на адрес запущенного docker-контейнера._

```bash
git clone git@github.com:leostuchchi/keras_emnist.git
cd keras_emnist/practice
docker build -t cv .
docker run -d -p 8000:8000 --name myapp-container cv
```
