"""
Конфігураційний файл для системи прогнозування та торгівлі криптовалютами
"""

# Загальні налаштування
CONFIG = {
    # Налаштування даних
    'data': {
        'symbol': 'BTC-USD',
        'timeframe': '1h',
        'period': '4y',
        'test_size': 0.15,
        'validation_size': 0.1
    },

    # Налаштування ознак
    'features': {
        'use_multitimeframe': False,
        'timeframes': ['1H', '4H', '1D'],
        'use_fib_lags': True,
        'max_lags': 34,
        'n_selected_features': 60
    },

    # Налаштування моделей
    'models': {
        'optimize_hyperparams': False,  # Вимкнено для швидкості
        'n_forecast_periods': 1,  # Прогноз на  години вперед
        'n_iterations': 10,  # Зменшена кількість ітерацій
        'cv_folds': 3  # Зменшена кількість фолдів
    },

    # Налаштування торгової стратегії
    'trading': {
        'initial_balance': 100000,  # Початковий капітал в USD
        'position_size': 0.01,  # Частка капіталу для кожної угоди
        'stop_loss': 0.015,  # Стоп-лосс
        'take_profit': 0.04,
        # 🆕 НОВІ ПАРАМЕТРИ ДЛЯ ШОРТ ПОЗИЦІЙ
        'enable_short_positions': False,  # Увімкнення шорт позицій
        'short_stop_loss': 0.015,  # 2.5% стоп-лос для шортів (трохи більший)
        'short_take_profit': 0.04,  # 4% тейк-профіт для шортів
        'max_short_positions': 1,  # Максимум одночасних шорт позицій
        'allow_simultaneous_long_short': False,  # Дозволити одночасні лонг і шорт
        'short_threshold_multiplier': 3.0,  # Вищі вимоги до впевненості для шортів
        'max_short_investment': 0.2,       # Максимум 20% в один шорт
        'enable_emergency_protection': True
    },
    'saving': {
        'models_dir': 'models/saved',
        'results_dir': 'results',
        'data_dir': 'data'
    }
}

# config.py - ОНОВЛЕНА конфігурація з параметрами для шорт позицій

