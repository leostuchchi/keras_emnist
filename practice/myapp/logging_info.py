import os
import logging
from datetime import datetime

path = os.path.dirname(os.path.abspath(__file__))  # текущий путь к файлу


def setup_logging():
    """
    Настраиваем логирование. Создает лог файл в myapp/logs.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Получаем текущую дату и время для названия лог файла
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(path, f'logs/app_{timestamp}.log')

    # Создаем обработчик файлового лога
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)

    # Создаем формат логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Добавляем обработчик к логгеру
    logger.addHandler(file_handler)

    return logger