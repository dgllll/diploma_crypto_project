import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_regression_metrics(y_true, y_pred):
    """
    Розраховує метрики для оцінки якості регресійних моделей

    Parameters:
    -----------
    y_true : array-like
        Фактичні значення
    y_pred : array-like
        Прогнозовані значення

    Returns:
    --------
    dict
        Словник з метриками
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Середня абсолютна процентна похибка (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # Середня процентна похибка (MPE) - для визначення зміщення
    mpe = np.mean((y_true - y_pred) / y_true) * 100

    # Theil's U - порівняння з наївною моделлю
    y_naive = np.roll(y_true, 1)  # Наївний прогноз: значення попереднього періоду
    y_naive[0] = y_naive[1]  # Виправлення першого значення

    theil_u_numerator = np.sqrt(np.mean((y_true - y_pred) ** 2))
    theil_u_denominator = np.sqrt(np.mean((y_true - y_naive) ** 2))
    theil_u = theil_u_numerator / theil_u_denominator if theil_u_denominator != 0 else np.inf

    # Оцінка правильності прогнозування напрямку руху
    actual_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))

    # Додаємо 0 в кінець щоб вирівняти розміри масивів
    actual_direction = np.append(actual_direction, 0)
    pred_direction = np.append(pred_direction, 0)

    # Відсоток правильних прогнозів напрямку
    direction_accuracy = np.mean(actual_direction == pred_direction) * 100

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'MPE': mpe,
        'Theil U': theil_u,
        'Direction Accuracy': direction_accuracy
    }


