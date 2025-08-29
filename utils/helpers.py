import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime


def ensure_dir(directory):
    """Створює директорію, якщо вона не існує"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_model(model, filename, directory='models'):
    """Зберігає модель у вказаний файл"""
    ensure_dir(directory)
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    print(f"Модель збережено у {filepath}")


def load_model(filename, directory='models'):
    """Завантажує модель з файлу"""
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"Модель завантажено з {filepath}")
        return model
    else:
        print(f"Файл {filepath} не знайдено")
        return None


def timestamp_to_string(timestamp=None):
    """Повертає поточну дату і час у форматі для назв файлів"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


def calculate_drawdown(portfolio_values):
    """Розраховує максимальну просадку"""
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = drawdown.min()
    return max_drawdown


def moving_average_crossover_signal(short_ma, long_ma):
    """Генерує сигнал на основі перетину ковзних середніх"""
    # Створюємо сигнал
    signal = np.where(short_ma > long_ma, 1, -1)  # 1 для покупки, -1 для продажу

    # Знаходимо точки перетину (зміна сигналу)
    crossover = np.diff(signal)
    crossover = np.insert(crossover, 0, 0)  # Додаємо 0 на початок для збереження розміру

    # Створюємо серію сигналів, де:
    # 1 = новий сигнал на покупку
    # -1 = новий сигнал на продаж
    # 0 = немає нового сигналу
    trading_signal = np.zeros_like(crossover)
    trading_signal[crossover == 2] = 1  # Перетин знизу вгору (з -1 на 1)
    trading_signal[crossover == -2] = -1  # Перетин зверху вниз (з 1 на -1)

    return trading_signal