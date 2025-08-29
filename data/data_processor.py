import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer


def preprocess_data(df, fill_method='ffill'):
    """Попередня обробка даних"""
    # Копіюємо датафрейм
    data = df.copy()

    # Переконуємося, що timestamp є DatetimeIndex
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')

    # Видалення дублікатів
    data = data[~data.index.duplicated(keep='first')]

    # Сортування за часом
    data = data.sort_index()

    # Заповнення пропущених значень
    data = data.fillna(method=fill_method)

    # Перевірка на нескінченні значення
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    # Скидання індексу для подальшої роботи
    data = data.reset_index()

    return data


def split_data(df, test_size=0.2, validation_size=0.1):
    """Розділення даних на тренувальний, валідаційний та тестовий набори"""
    # Загальна кількість записів
    n = len(df)

    # Індекси для розділення
    test_start = int(n * (1 - test_size))
    val_start = int(test_start * (1 - validation_size))

    # Розділення даних
    train_df = df.iloc[:val_start].copy()
    val_df = df.iloc[val_start:test_start].copy()
    test_df = df.iloc[test_start:].copy()

    print(f"Тренувальний набір: {train_df.shape}")
    print(f"Валідаційний набір: {val_df.shape}")
    print(f"Тестовий набір: {test_df.shape}")

    return train_df, val_df, test_df


def scale_features(X_train, X_val, X_test, method='standard'):
    """Масштабування ознак"""
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'yeo-johnson':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        raise ValueError("Непідтримуваний метод масштабування")

    # Навчання скейлера тільки на тренувальних даних
    X_train_scaled = scaler.fit_transform(X_train)

    # Трансформація валідаційних та тестових даних
    X_val_scaled = scaler.transform(X_val) if X_val is not None else None
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def scale_target(y_train, y_val, y_test, method='yeo-johnson'):
    """Масштабування цільової змінної"""
    transformer = PowerTransformer(method=method)

    # Перетворення масивів у 2D формат для трансформації
    y_train_2d = y_train.values.reshape(-1, 1)

    # Навчання трансформера на тренувальних даних
    y_train_scaled = transformer.fit_transform(y_train_2d).flatten()

    # Трансформація валідаційних та тестових даних
    y_val_scaled = transformer.transform(y_val.values.reshape(-1, 1)).flatten() if y_val is not None else None
    y_test_scaled = transformer.transform(y_test.values.reshape(-1, 1)).flatten() if y_test is not None else None

    return y_train_scaled, y_val_scaled, y_test_scaled, transformer