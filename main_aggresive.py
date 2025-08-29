
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import traceback


# Імпортуємо модулі проекту
from config import CONFIG
from aggressive_trading_system import AggressiveTradingSystem, plot_backtest_results_standalone
from data.data_loader import load_bitcoin_data, validate_data, save_data_to_csv, test_data_loading
from data.data_processor import preprocess_data, split_data, scale_features
from features.feature_engineering import build_advanced_features, prepare_features_targets_robust
from models.prediction_pipeline import CryptoPricePredictionPipeline

# 🔥 ВИДАЛЕНО старі імпорти, тепер використовуємо тільки AggressiveTradingSystem
# from aggressive_strategy import AggressiveCryptoTradingStrategy  # ВИДАЛЕНО
# from fixed_aggressive_backtesting import (  # ВИДАЛЕНО
#     backtest_aggressive_strategy_fixed as backtest_aggressive_strategy,
#     plot_aggressive_backtest_results
# )

from evaluation.visualization import plot_price_prediction, plot_feature_importance
from utils.helpers import save_model, load_model, timestamp_to_string, ensure_dir


def fix_random_seeds(seed=42):
    """
    🎲 Фіксує всі джерела випадковості для відтворюваності результатів
    """
    print(f"🔧 Фіксація випадковості з seed={seed}")

    # Python built-in random
    random.seed(seed)

    # NumPy random
    np.random.seed(seed)

    # Python hash seed (для словників)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Для ML бібліотек (якщо використовуються)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        print("✅ TensorFlow seed зафіксовано")
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print("✅ PyTorch seed зафіксовано")
    except ImportError:
        pass

    print("✅ Всі джерела випадковості зафіксовані")


def run_data_preparation():
    """Підготовка даних з покращеною обробкою помилок"""
    print("=" * 50)
    print("КРОК 1: Завантаження та підготовка даних")
    print("=" * 50)

    try:
        # Спочатку тестуємо всі джерела даних
        print("Тестування доступності джерел даних...")
        df = load_bitcoin_data(
            prefer_source='binance',  # Можна змінити на 'coinbase' або 'binance'
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

        # 🔧 ВИПРАВЛЕННЯ: Правильна обробка model_info
        try:
            model_info = pipeline.get_model_info()
            print(f"\n📊 Інформація про ансамбль:")

            if isinstance(model_info, dict):
                # Новий формат (словник)
                print(model_info['description'])
                print(f"Кількість моделей: {model_info['n_models']}")
                print(f"Статус: {model_info['status']}")
                print(f"Загальна вага: {model_info['total_weight']:.4f}")

                # Детальна інформація про моделі
                for name, info in model_info['models'].items():
                    weight_pct = info['weight'] * 100
                    print(f"  {name}: {weight_pct:.1f}%")

            else:
                # Старий формат (рядок) - для зворотної сумісності
                print(model_info)

        except AttributeError:
            print("⚠️ Метод get_model_info() недоступний")
        except Exception as e:
            print(f"⚠️ Помилка отримання інформації про модель: {e}")

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


def ensure_required_columns(test_features, feature_names_for_model):
    """
    🔧 Переконується, що всі необхідні колонки присутні в test_features
    """
    print("🔧 Перевірка та синхронізація колонок...")

    # Список усіх необхідних колонок
    required_columns = [
        'timestamp', 'close', 'open', 'high', 'low', 'volume',  # Базові OHLCV
        'rsi', 'macd_12_26', 'macd_signal_12_26', 'adx', 'volume_ratio_20'  # Технічні індикатори
    ] + feature_names_for_model

    # Видаляємо дублікати
    required_columns = list(set(required_columns))

    # Перевіряємо наявність колонок
    missing_columns = [col for col in required_columns if col not in test_features.columns]

    if missing_columns:
        print(f"⚠️ Відсутні колонки: {missing_columns}")

        # Додаємо відсутні колонки з значеннями за замовчуванням
        for col in missing_columns:
            if col in ['rsi']:
                test_features[col] = 50  # RSI за замовчуванням
            elif col in ['macd_12_26', 'macd', 'macd_signal_12_26', 'macd_signal']:
                test_features[col] = 0   # MACD за замовчуванням
            elif col in ['adx']:
                test_features[col] = 25  # ADX за замовчуванням
            elif col in ['volume_ratio_20']:
                test_features[col] = 1.0 # Volume ratio за замовчуванням
            elif 'lag' in col or 'sma' in col or 'ema' in col:
                # Для лагових та MA індикаторів використовуємо close
                test_features[col] = test_features['close'] if 'close' in test_features.columns else 50000
            else:
                # Для всіх інших - нульові значення
                test_features[col] = 0

        print(f"✓ Додано {len(missing_columns)} відсутніх колонок зі значеннями за замовчуванням")

    # Перевіряємо альтернативні назви для MACD
    if 'macd_12_26' not in test_features.columns and 'macd' in test_features.columns:
        test_features['macd_12_26'] = test_features['macd']
        print("✓ Використано 'macd' як 'macd_12_26'")

    if 'macd_signal_12_26' not in test_features.columns and 'macd_signal' in test_features.columns:
        test_features['macd_signal_12_26'] = test_features['macd_signal']
        print("✓ Використано 'macd_signal' як 'macd_signal_12_26'")

    print(f"✓ Всі необхідні колонки присутні в даних")
    return test_features


def run_aggressive_trading_strategy(pipeline, test_features, feature_names_for_model):
    """
    🔥 Запуск та оцінка АГРЕСИВНОЇ торгової системи (об'єднана версія)
    """
    print("\n" + "=" * 50)
    print("🔥 КРОК 5: АГРЕСИВНА торгова система (об'єднана)")
    print("💰 З комісіями та проскальзуванням (логіка в класі системи)")
    print("=" * 50)

    try:
        # 🔧 Переконуємося, що всі необхідні колонки присутні
        test_features = ensure_required_columns(test_features, feature_names_for_model)

        # 🔥 Створення екземпляра АГРЕСИВНОЇ торгової системи
        trading_system = AggressiveTradingSystem(
            prediction_model=pipeline,
            initial_balance=CONFIG['trading']['initial_balance']
        )

        print(f"🔥 Параметри агресивної системи:")
        print(f"  Базовий поріг впевненості: {trading_system.trading_params['base_confidence_threshold']}")
        print(f"  RSI пороги: {trading_system.trading_params['rsi_oversold']}-{trading_system.trading_params['rsi_overbought']}")
        print(f"  Скальпінг: {'Увімкнено' if trading_system.trading_params['enable_scalping'] else 'Вимкнено'}")
        print(f"  Початковий капітал: ${trading_system.initial_balance}")

        print(f"\n💰 Торгові витрати:")
        print(f"  Комісія тейкер: {trading_system.trading_costs['taker_fee']*100:.4f}%")
        #print(f"  Проскальзування: {trading_system.trading_costs['fixed_slippage_usd']:.4f}:$")
        print(f"  Мінімальний трейд: ${trading_system.trading_costs['min_trade_amount']}")

        # 🔥 Бектестинг системи
        print("\n🔥 Проведення бектесту системи...")
        backtest_results_df = trading_system.run_backtest(
            historical_data=test_features,
            feature_names_for_model=feature_names_for_model
        )

        # 📊 Візуалізація результатів бектесту
        if not backtest_results_df.empty:
            print("📊 Створення візуалізації результатів бектесту...")
            plot_backtest_results_standalone(backtest_results_df, "Агресивна Торгова Система")
        else:
            print("⚠️ Немає результатів для візуалізації.")

        # 📈 Розрахунок розширених метрик торгівлі
        trading_stats = trading_system.get_trading_statistics()

        # Збереження результатів бектесту
        if not backtest_results_df.empty:
            ensure_dir(CONFIG['saving']['results_dir'])
            results_filename = f"aggressive_SYSTEM_backtest_results_{timestamp_to_string()}.csv"
            results_path = os.path.join(CONFIG['saving']['results_dir'], results_filename)
            backtest_results_df.to_csv(results_path, index=False)
            print(f"✓ Результати бектесту системи збережено в {results_path}")

        return backtest_results_df, trading_stats

    except Exception as e:
        print(f"❌ Помилка при тестуванні агресивної торгової системи: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())
        return pd.DataFrame(), {}


def create_aggressive_summary_report(prediction_metrics, trading_metrics, save_path=None):
    """
    📊 Створює підсумковий звіт для агресивної стратегії
    """
    print("\n" + "=" * 50)
    print("🔥 ПІДСУМКОВИЙ ЗВІТ АГРЕСИВНОЇ СИСТЕМИ")
    print("=" * 50)

    report = []
    report.append("# 🔥 Звіт про роботу агресивної системи прогнозування Bitcoin")
    report.append(f"Дата створення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## ⚙️ Параметри системи")
    report.append(f"- Горизонт прогнозування: {CONFIG['models']['n_forecast_periods']} годин")
    report.append(f"- Таймфрейм даних: {CONFIG['data']['timeframe']}")
    report.append(f"- Період даних: {CONFIG['data']['period']}")
    report.append(f"- Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")
    report.append("")

    report.append("## 🎯 Цілі агресивної стратегії")
    report.append("- **Основна ціль**: Збільшити кількість трейдів з 7 до 50+")
    report.append("- **Додаткові цілі**: Врахувати комісії та проскальзування")
    report.append("- **Реалістичність**: Максимально наблизити до реальної торгівлі")
    report.append("")

    report.append("## 📊 Результати прогнозування")
    for metric, value in prediction_metrics.items():
        if isinstance(value, dict):
            report.append(f"- {metric}:")
            for k, v in value.items():
                report.append(f"  - {k}: {v}")
        else:
            report.append(f"- {metric}: {value:.4f}")
    report.append("")

    report.append("## 🔥 Результати агресивної торгової стратегії")
    if trading_metrics:
        total_trades = trading_metrics.get('total_trades', 0)
        total_costs = trading_metrics.get('total_trading_costs', 0)

        # Результат досягнення цілі
        if total_trades >= 50:
            goal_status = "✅ ЦІЛЬ ДОСЯГНУТО"
        elif total_trades >= 30:
            goal_status = "⚡ БЛИЗЬКО ДО ЦІЛІ"
        else:
            goal_status = "❌ ЦІЛЬ НЕ ДОСЯГНУТА"

        report.append(f"### 🎯 Досягнення мети: {goal_status}")
        report.append(f"- Кількість трейдів: {total_trades} (мета: 50+)")
        report.append("")

        report.append("### 📈 Детальна статистика:")
        for metric, value in trading_metrics.items():
            if isinstance(value, float):
                report.append(f"- {metric}: {value:.4f}")
            else:
                report.append(f"- {metric}: {value}")

        report.append("")
        report.append(f"### 💰 Торгові витрати:")
        report.append(f"- Загальні витрати: ${total_costs:.2f}")
        report.append(f"- Витрати у відсотках: {trading_metrics.get('cost_ratio', 0):.2f}%")

    else:
        report.append("- Агресивна торгова стратегія не була протестована")
    report.append("")

    report_text = "\n".join(report)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Звіт збережено в {save_path}")

    print(report_text)
    return report_text


def main():
    """🔥 Головна функція з агресивною торговою стратегією (ВИПРАВЛЕНА)"""

    # 🎯 КРИТИЧНО ВАЖЛИВО: Фіксація випадковості на самому початку!
    fix_random_seeds(42)

    print("🔥 Запуск системи прогнозування та АГРЕСИВНОЇ торгівлі криптовалютами")
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

        # Отримуємо список ознак для моделі
        feature_names_for_model = list(X_test.columns)
        print(f"✓ Ознаки для моделі: {len(feature_names_for_model)} колонок")

        # 🔥 Крок 5: АГРЕСИВНА торгова стратегія
        backtest_results, trading_metrics = run_aggressive_trading_strategy(
            pipeline, test_features, feature_names_for_model
        )
        results['trading_metrics'] = trading_metrics
        results['backtest_results'] = backtest_results
        results['pipeline'] = pipeline

        # Створення підсумкового звіту
        ensure_dir(CONFIG['saving']['results_dir'])
        report_path = os.path.join(
            CONFIG['saving']['results_dir'],
            f"aggressive_summary_report_{timestamp_to_string()}.md"
        )

        create_aggressive_summary_report(prediction_metrics, trading_metrics, report_path)

        print("\n🎉 Система агресивного прогнозування та торгівлі успішно завершила роботу!")

        # Фінальний підсумок мети
        total_trades = trading_metrics.get('total_trades', 0)
        if total_trades >= 50:
            print(f"🏆 ГОЛОВНА ЦІЛЬ ДОСЯГНУТА: {total_trades} трейдів!")
        else:
            print(f"⚠️  Потрібно налаштування: {total_trades} трейдів з 50")

        return results

    except Exception as e:
        print(f"\n❌ Критична помилка в головній функції: {str(e)}")
        print("Повна інформація про помилку:")
        print(traceback.format_exc())

        print("\n🔧 Рекомендації для вирішення проблем:")
        print("1. Перевірте інтернет-з'єднання")
        print("2. Спробуйте оновити бібліотеку yfinance: pip install --upgrade yfinance")
        print("3. Спробуйте використати інше джерело даних")
        print("4. Перевірте, чи всі залежності встановлені правильно")

        return None


if __name__ == "__main__":
    results = main()
    if results:
        print(f"\n✅ Результати доступні в змінній 'results'")
        print(f"🔍 Для детального аналізу дивіться збережені файли в папці '{CONFIG['saving']['results_dir']}'")

        # Короткий підсумок результатів
        trading_stats = results.get('trading_metrics', {})
        if trading_stats:
            total_trades = trading_stats.get('total_trades', 0)
            print(f"\n📊 ШВИДКИЙ ПІДСУМОК:")
            print(f"   Трейдів: {total_trades}")
            print(f"   Комісій сплачено: ${trading_stats.get('total_fees_paid', 0):.2f}")
            print(f"   Скальпінг трейдів: {trading_stats.get('scalping_trades', 0)}")