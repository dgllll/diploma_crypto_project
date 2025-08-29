"""
Головний файл для запуску системи прогнозування та торгівлі криптовалютами
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import traceback

# Імпортуємо модулі проекту
from config import CONFIG
from data.data_loader import load_bitcoin_data, validate_data, save_data_to_csv, test_data_loading
from data.data_processor import preprocess_data, split_data, scale_features
from features.feature_engineering import build_advanced_features, prepare_features_targets_robust
from models.prediction_pipeline import CryptoPricePredictionPipeline
from trading.strategy import CryptoTradingStrategy
from trading.backtesting import backtest_strategy, plot_backtest_results, calculate_trading_metrics
from evaluation.visualization import plot_price_prediction, plot_feature_importance
from utils.helpers import save_model, load_model, timestamp_to_string, ensure_dir

def run_data_preparation():
    """Підготовка даних з покращеною обробкою помилок"""
    print("=" * 50)
    print("КРОК 1: Завантаження та підготовка даних")
    print("=" * 50)

    try:
        # Спочатку тестуємо всі джерела даних
        print("Тестування доступності джерел даних...")
        df = load_bitcoin_data(
            prefer_source='yahoo',  # Можна змінити на 'coinbase' або 'binance'
            period=CONFIG['data']['period'],
            interval=CONFIG['data']['timeframe']
        )

        # Перевіряємо валідність даних
        is_valid, validation_message = validate_data(df)

        if not is_valid:
            print(f"Помилка валідації даних: {validation_message}")
            print("Спроба використання тестових даних...")
            df = test_data_loading()

        if df.empty:
            raise ValueError("Не вдалося завантажити жодних даних!")

        print(f"✓ Успішно завантажено {len(df)} рядків даних")
        print(f"✓ Період даних: з {df['timestamp'].min()} до {df['timestamp'].max()}")
        print(f"✓ Ціновий діапазон: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

        # Збереження сирих даних
        ensure_dir(CONFIG['saving']['data_dir'])
        raw_data_path = os.path.join(
            CONFIG['saving']['data_dir'],
            f"raw_data_{timestamp_to_string()}.csv"
        )

        if save_data_to_csv(df, raw_data_path):
            print(f"✓ Сирі дані збережено в {raw_data_path}")

        # Попередня обробка даних
        print("Попередня обробка даних...")
        df = preprocess_data(df)

        if df.empty:
            raise ValueError("Дані стали пустими після попередньої обробки!")

        print(f"✓ Після попередньої обробки: {len(df)} рядків")

        # Розділення даних
        print("Розділення на тренувальний, валідаційний та тестовий набори...")
        train_df, val_df, test_df = split_data(
            df,
            test_size=CONFIG['data']['test_size'],
            validation_size=CONFIG['data']['validation_size']
        )

        print(f"✓ Тренувальний набір: {len(train_df)} рядків")
        print(f"✓ Валідаційний набір: {len(val_df)} рядків")
        print(f"✓ Тестовий набір: {len(test_df)} рядків")

        return train_df, val_df, test_df

    except Exception as e:
        print(f"❌ Помилка при підготовці даних: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())

        # Спробуємо створити синтетичні дані як резерв
        print("Спроба створення синтетичних даних для продовження роботи...")
        try:
            from data.data_loader import create_sample_data
            df = create_sample_data()

            if not df.empty:
                print("✓ Створено синтетичні дані")
                df = preprocess_data(df)
                train_df, val_df, test_df = split_data(df,
                                                     test_size=CONFIG['data']['test_size'],
                                                     validation_size=CONFIG['data']['validation_size'])
                return train_df, val_df, test_df
        except Exception as e2:
            print(f"❌ Не вдалося створити синтетичні дані: {str(e2)}")

        raise Exception("Критична помилка: не вдалося підготувати дані для роботи")

def run_feature_engineering(train_df, val_df, test_df):
    """Інженерія ознак з покращеною обробкою помилок"""
    print("\n" + "=" * 50)
    print("КРОК 2: Створення ознак")
    print("=" * 50)

    try:
        print("Створення розширених ознак...")

        # Створюємо ознаки для кожного набору даних
        print("Обробка тренувальних даних...")
        train_features = build_advanced_features(
            train_df,
            timeframes=CONFIG['features']['timeframes'] if CONFIG['features']['use_multitimeframe'] else []
        )

        print("Обробка валідаційних даних...")
        val_features = build_advanced_features(
            val_df,
            timeframes=CONFIG['features']['timeframes'] if CONFIG['features']['use_multitimeframe'] else []
        )

        print("Обробка тестових даних...")
        test_features = build_advanced_features(
            test_df,
            timeframes=CONFIG['features']['timeframes'] if CONFIG['features']['use_multitimeframe'] else []
        )

        print(f"✓ Створені ознаки для всіх наборів даних")
        print(f"✓ Кількість ознак: {len(train_features.columns)}")

        # Підготовка ознак та цільових змінних з покращеною обробкою NaN
        print("Підготовка ознак та цільових змінних...")

        X_train, y_train = prepare_features_targets_robust(
            train_features,
            target_col='close',
            forecast_horizon=CONFIG['models']['n_forecast_periods'],
            nan_threshold=0.7,  # Видаляємо колонки з >70% NaN
            min_samples=50
        )

        X_val, y_val = prepare_features_targets_robust(
            val_features,
            target_col='close',
            forecast_horizon=CONFIG['models']['n_forecast_periods'],
            nan_threshold=0.7,
            min_samples=10
        )

        X_test, y_test = prepare_features_targets_robust(
            test_features,
            target_col='close',
            forecast_horizon=CONFIG['models']['n_forecast_periods'],
            nan_threshold=0.7,
            min_samples=10
        )

        print(f"✓ Тренувальні ознаки: {X_train.shape}")
        print(f"✓ Валідаційні ознаки: {X_val.shape}")
        print(f"✓ Тестові ознаки: {X_test.shape}")

        # Перевіряємо, чи достатньо даних
        if len(X_train) < 100:
            print("⚠️ Увага: Мало тренувальних даних (< 100 рядків)")

        if X_train.shape[1] < 10:
            print("⚠️ Увага: Мало ознак (< 10)")

        # Синхронізуємо колонки між наборами даних
        common_columns = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))

        if len(common_columns) < len(X_train.columns):
            print(f"Синхронізація колонок: {len(X_train.columns)} -> {len(common_columns)}")
            X_train = X_train[common_columns]
            X_val = X_val[common_columns]
            X_test = X_test[common_columns]

        return X_train, y_train, X_val, y_val, X_test, y_test, test_features

    except Exception as e:
        print(f"❌ Помилка при створенні ознак: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())
        raise

def run_model_training(X_train, y_train, X_val, y_val):
    """Навчання та оптимізація моделей"""
    print("\n" + "=" * 50)
    print("КРОК 3: Навчання та оптимізація моделей")
    print("=" * 50)

    try:
        # Створення конвеєру прогнозування
        pipeline = CryptoPricePredictionPipeline(
            n_forecast_periods=CONFIG['models']['n_forecast_periods']
        )

        # Навчання моделі з оптимізацією
        print("Навчання ансамблевої моделі...")
        print(f"Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")

        pipeline.fit(
            X_train, y_train, X_val, y_val,
            optimize=CONFIG['models']['optimize_hyperparams']
        )

        # Збереження моделі
        ensure_dir(CONFIG['saving']['models_dir'])
        model_filename = f"crypto_prediction_model_{timestamp_to_string()}.pkl"
        model_path = os.path.join(CONFIG['saving']['models_dir'], model_filename)
        save_model(pipeline, model_filename, CONFIG['saving']['models_dir'])

        print(f"✓ Модель збережено в {model_path}")

        return pipeline

    except Exception as e:
        print(f"❌ Помилка при навчанні моделей: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())
        raise

def run_model_evaluation(pipeline, X_test, y_test):
    """Оцінка моделі на тестових даних"""
    print("\n" + "=" * 50)
    print("КРОК 4: Оцінка моделі")
    print("=" * 50)

    try:
        # Оцінка моделі
        print("Оцінка моделі на тестових даних...")
        metrics = pipeline.evaluate(X_test, y_test)

        # Отримання прогнозів
        y_pred = pipeline.predict(X_test)

        # Візуалізація результатів
        print("Створення візуалізації результатів...")
        plot_price_prediction(y_test, y_pred, title='Bitcoin Price Prediction - Test Data')

        # Якщо доступна інформація про важливість ознак
        if pipeline.ensemble and pipeline.ensemble.feature_importances_:
            importances = list(pipeline.ensemble.feature_importances_.values())
            features = list(pipeline.ensemble.feature_importances_.keys())
            plot_feature_importance(features, importances, n_features=20)

        print("\n📊 Результати оцінки моделі:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"{metric}: {value}")
            else:
                print(f"{metric}: {value:.4f}")

        return y_pred, metrics

    except Exception as e:
        print(f"❌ Помилка при оцінці моделі: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())
        raise

def run_trading_strategy(pipeline, test_features):
    """Запуск та оцінка торгової стратегії"""
    print("\n" + "=" * 50)
    print("КРОК 5: Тестування торгової стратегії")
    print("=" * 50)

    try:
        # Створення торгової стратегії
        strategy = CryptoTradingStrategy(
            prediction_model=pipeline,
            confidence_threshold=CONFIG['trading']['confidence_threshold'],
            position_size=CONFIG['trading']['position_size']
        )

        print(f"Параметри стратегії:")
        print(f"  Поріг впевненості: {CONFIG['trading']['confidence_threshold']}%")
        print(f"  Розмір позиції: {CONFIG['trading']['position_size']*100}%")
        print(f"  Початковий капітал: ${CONFIG['trading']['initial_balance']}")

        # Бектестинг стратегії
        print("Проведення бектесту...")
        backtest_results = backtest_strategy(
            strategy,
            test_features,
            pipeline.feature_names
        )

        # Візуалізація результатів бектесту
        print("Створення візуалізації торгової стратегії...")
        plot_backtest_results(backtest_results)

        # Розрахунок додаткових метрик торгівлі
        trading_metrics = calculate_trading_metrics(backtest_results)

        # Збереження результатів бектесту
        ensure_dir(CONFIG['saving']['results_dir'])
        results_filename = f"backtest_results_{timestamp_to_string()}.csv"
        results_path = os.path.join(CONFIG['saving']['results_dir'], results_filename)
        backtest_results.to_csv(results_path, index=False)

        print(f"✓ Результати бектесту збережено в {results_path}")

        print("\n📈 Результати торгової стратегії:")
        for key, value in trading_metrics.items():
            print(f"{key}: {value:.4f}")

        return backtest_results, trading_metrics

    except Exception as e:
        print(f"❌ Помилка при тестуванні торгової стратегії: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())

        # Повертаємо порожні результати при помилці
        return pd.DataFrame(), {}

def create_summary_report(prediction_metrics, trading_metrics, save_path=None):
    """Створює підсумковий звіт"""
    print("\n" + "=" * 50)
    print("ПІДСУМКОВИЙ ЗВІТ")
    print("=" * 50)

    report = []
    report.append("# Звіт про роботу системи прогнозування Bitcoin")
    report.append(f"Дата створення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## Параметри системи")
    report.append(f"- Горизонт прогнозування: {CONFIG['models']['n_forecast_periods']} годин")
    report.append(f"- Таймфрейм даних: {CONFIG['data']['timeframe']}")
    report.append(f"- Період даних: {CONFIG['data']['period']}")
    report.append(f"- Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")
    report.append("")

    report.append("## Результати прогнозування")
    for metric, value in prediction_metrics.items():
        if isinstance(value, dict):
            report.append(f"- {metric}:")
            for k, v in value.items():
                report.append(f"  - {k}: {v}")
        else:
            report.append(f"- {metric}: {value:.4f}")
    report.append("")

    report.append("## Результати торгової стратегії")
    if trading_metrics:
        for metric, value in trading_metrics.items():
            report.append(f"- {metric}: {value:.4f}")
    else:
        report.append("- Торгова стратегія не була протестована")
    report.append("")

    report.append("## Висновки")
    report.append("Система успішно:")
    report.append("1. Завантажила та обробила історичні дані Bitcoin")
    report.append("2. Створила розширений набір технічних індикаторів")
    report.append("3. Навчила ансамблеву модель машинного навчання")
    report.append("4. Оцінила точність прогнозування")

    if trading_metrics:
        profit = trading_metrics.get('Total Return', 0) * 100
        if profit > 0:
            report.append(f"5. Показала прибутковість торгової стратегії: {profit:.2f}%")
        else:
            report.append(f"5. Торгова стратегія показала збиток: {profit:.2f}%")

    report_text = "\n".join(report)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Звіт збережено в {save_path}")

    print(report_text)
    return report_text

def main():
    """Головна функція з покращеною обробкою помилок"""
    print("🚀 Запуск системи прогнозування та торгівлі криптовалютами")
    print("Версія з покращеною обробкою помилок завантаження даних")
    print("=" * 70)

    results = {}

    try:
        # Крок 1: Підготовка даних
        train_df, val_df, test_df = run_data_preparation()

        # Крок 2: Інженерія ознак
        X_train, y_train, X_val, y_val, X_test, y_test, test_features = run_feature_engineering(
            train_df, val_df, test_df
        )

        # Крок 3: Навчання моделей
        pipeline = run_model_training(X_train, y_train, X_val, y_val)

        # Крок 4: Оцінка моделі
        y_pred, prediction_metrics = run_model_evaluation(pipeline, X_test, y_test)
        results['prediction_metrics'] = prediction_metrics

        # Крок 5: Торгова стратегія
        backtest_results, trading_metrics = run_trading_strategy(pipeline, test_features)
        results['trading_metrics'] = trading_metrics
        results['backtest_results'] = backtest_results
        results['pipeline'] = pipeline

        # Створення підсумкового звіту
        ensure_dir(CONFIG['saving']['results_dir'])
        report_path = os.path.join(
            CONFIG['saving']['results_dir'],
            f"summary_report_{timestamp_to_string()}.md"
        )

        create_summary_report(prediction_metrics, trading_metrics, report_path)

        print("\n🎉 Система прогнозування та торгівлі успішно завершила роботу!")

        return results

    except Exception as e:
        print(f"\n❌ Критична помилка в головній функції: {str(e)}")
        print("Повна інформація про помилку:")
        print(traceback.format_exc())

        print("\n🔧 Рекомендації для вирішення проблем:")
        print("1. Перевірте інтернет-з'єднання")
        print("2. Спробуйте оновити бібліотеку yfinance: pip install --upgrade yfinance")
        print("3. Спробуйте використати інше джерело даних, змінивши prefer_source в load_bitcoin_data()")
        print("4. Перевірте, чи всі залежності встановлені правильно")
        print("5. Зверніться до документації проєкту")

        return None

if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Результати доступні в змінній 'results'")
        print(f"🔍 Для детального аналізу дивіться збережені файли в папці '{CONFIG['saving']['results_dir']}'")