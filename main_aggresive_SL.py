"""
Головний файл для запуску системи прогнозування та агресивної торгівлі криптовалютами
🔥 ОНОВЛЕННЯ: Використання розділених модулів торгової системи та бектестингу
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
# 🆕 ІМПОРТИ ДЛЯ PLOTLY ВІЗУАЛІЗАЦІЇ

from datetime import datetime
import traceback
# Імпортуємо модулі проекту
from config import CONFIG
from aggressive_trading_system_SL import AggressiveTradingSystem
try:
    from aggressive_backtesting_SL import (
        run_aggressive_backtest,
        plot_backtest_results,
        calculate_trading_metrics,
        save_backtest_results
    )
except ImportError as e:
    try:
        # Fallback
        from aggressive_backtesting_SL import (
            run_aggressive_backtest,
            plot_backtest_results,
            calculate_trading_metrics,
            save_backtest_results
        )
    except ImportError as e2:
        print(f"❌ Помилка імпорту модулів бектестингу: {e}, {e2}")
        print("Перевірте наявність файлів aggressive_backtesting.py або aggressive_backtesting_fixed.py")

# Імпорти для підготовки даних та моделей
from data.data_loader import load_bitcoin_data, validate_data, save_data_to_csv, test_data_loading
from data.data_processor import preprocess_data, split_data, scale_features
from features.feature_engineering import build_advanced_features, prepare_features_targets_robust
from models.prediction_pipeline import CryptoPricePredictionPipeline
from evaluation.visualization import plot_price_prediction, plot_feature_importance
from utils.helpers import save_model, load_model, timestamp_to_string, ensure_dir


def comprehensive_seed_fix(seed=42):
    """
    🎲 Максимально повна фіксація всіх джерел випадковості
    """
    print(f"🔧 Комплексна фіксація випадковості з seed={seed}")

    # 1. Python built-in random
    random.seed(seed)

    # 2. NumPy random
    np.random.seed(seed)

    # 3. Python hash seed (для словників та множин)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 4. Scikit-learn random states
    os.environ['SKLEARN_SEED'] = str(seed)

    # 5. XGBoost
    try:
        import xgboost as xgb
        # Фіксуємо глобальні налаштування XGBoost
        xgb.set_config(verbosity=0)
        print("✅ XGBoost seed зафіксовано")
    except ImportError:
        pass

    # 6. LightGBM
    try:
        import lightgbm as lgb
        # LightGBM використовує параметр random_state в моделях
        print("✅ LightGBM готовий до фіксації")
    except ImportError:
        pass

    print("✅ Всі доступні джерела випадковості зафіксовані")


def ensure_deterministic_training():
    """
    🔒 Додаткові налаштування для детермінованого навчання
    """
    # Для pandas
    import pandas as pd

    # Фіксуємо numpy для pandas операцій
    pd.set_option('mode.chained_assignment', None)

    # Для sklearn
    try:
        from sklearn.utils import check_random_state
        print("✅ Sklearn готовий до детермінованого навчання")
    except ImportError:
        pass

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

        return pipeline

    except Exception as e:
        print(f"❌ Помилка при навчанні моделей: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())
        raise


def run_model_evaluation(pipeline, X_test, y_test):
    """Оновлена оцінка моделі з детальними прогнозами"""
    print("\n" + "=" * 50)
    print("КРОК 4: Оцінка моделі")
    print("=" * 50)

    try:
        # Стандартна оцінка
        print("Оцінка моделі на тестових даних...")
        metrics = pipeline.evaluate(X_test, y_test)
        y_pred = pipeline.predict(X_test)

        # 🆕 ДОДАТИ детальні прогнози
        print("\n" + "🎯" * 20)
        detailed_results = pipeline.show_detailed_predictions(X_test, y_test, n_predictions=15)

        # 🆕 ДОДАТИ прогноз вперед
        print("\n" + "🔮" * 20)
        future_predictions = pipeline.get_next_predictions(X_test, n_steps=5)

        # Візуалізація (існуючий код)
        print("Створення візуалізації результатів...")
        plot_price_prediction(y_test, y_pred, title='Bitcoin Price Prediction - Test Data')

        if pipeline.ensemble and pipeline.ensemble.feature_importances_:
            importances = list(pipeline.ensemble.feature_importances_.values())
            features = list(pipeline.ensemble.feature_importances_.keys())
            plot_feature_importance(features, importances, n_features=20)

        # Додаємо нові результати до метрик
        metrics['Detailed_Predictions'] = detailed_results
        metrics['Future_Predictions'] = future_predictions

        print("\n📊 Результати оцінки моделі:")
        for metric, value in metrics.items():
            if metric not in ['Detailed_Predictions', 'Future_Predictions']:
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
                test_features[col] = 0  # MACD за замовчуванням
            elif col in ['adx']:
                test_features[col] = 25  # ADX за замовчуванням
            elif col in ['volume_ratio_20']:
                test_features[col] = 1.0  # Volume ratio за замовчуванням
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
    🔥 Запуск та оцінка АГРЕСИВНОЇ торгової системи (нова версія з розділеними модулями)
    """
    print("\n" + "=" * 50)
    print("🔥 КРОК 5: АГРЕСИВНА торгова система (з розділеними модулями)")
    print("💰 З комісіями та проскальзуванням + стоп-лос/тейк-профіт")
    print("=" * 50)

    try:
        # 🔧 Переконуємося, що всі необхідні колонки присутні
        test_features = ensure_required_columns(test_features, feature_names_for_model)

        # 🔥 Створення екземпляра АГРЕСИВНОЇ торгової системи
        trading_system = AggressiveTradingSystem(
            prediction_model=pipeline,
            initial_balance=CONFIG['trading']['initial_balance']
        )
        # В run_aggressive_trading_strategy() після створення системи
        trading_system.show_trading_predictions_summary(test_features, feature_names_for_model, n_examples=10)

        print(f"🔥 Параметри агресивної системи:")
        print(f"  Базовий поріг впевненості: {trading_system.trading_params['base_confidence_threshold']}")
        print(
            f"  RSI пороги: {trading_system.trading_params['rsi_oversold']}-{trading_system.trading_params['rsi_overbought']}")
        print(f"  Скальпінг: {'Увімкнено' if trading_system.trading_params['enable_scalping'] else 'Вимкнено'}")
        print(f"  Початковий капітал: ${trading_system.initial_balance}")

        print(f"\n💰 Торгові витрати:")
        print(f"  Комісія тейкер: {trading_system.trading_costs['taker_fee'] * 100:.4f}%")
        #print(f"  Проскальзування: {trading_system.trading_costs['slippage_pct'] * 100:.4f}%")
        print(f"  Мінімальний трейд: ${trading_system.trading_costs['min_trade_amount']}")

        print(f"\n🛡️ Ризик-менеджмент:")
        print(f"  Стоп-лос: {trading_system.risk_params['stop_loss_pct'] * 100:.1f}%")
        print(f"  Тейк-профіт: {trading_system.risk_params['take_profit_pct'] * 100:.1f}%")
        print(f"  Макс. позицій: {trading_system.risk_params['max_open_positions']}")

        # 🔥 Бектестинг системи з новим модулем
        print("\n🔥 Проведення бектесту системи...")
        backtest_results_df = run_aggressive_backtest(
            trading_system=trading_system,
            historical_data=test_features,
            feature_names_for_model=feature_names_for_model
        )

        # 📊 Візуалізація результатів бектесту
        if not backtest_results_df.empty:
            print("📊 Створення візуалізації результатів бектесту...")
            plot_backtest_results(backtest_results_df, "Агресивна Торгова Система з Стоп-лосом")
        else:
            print("⚠️ Немає результатів для візуалізації.")

        # 📈 Розрахунок розширених метрик торгівлі
        trading_metrics = calculate_trading_metrics(backtest_results_df)
        trading_stats = trading_system.get_trading_statistics()

        # Збереження результатів бектесту
        if not backtest_results_df.empty:
            print("💾 Збереження результатів...")
            results_path, stats_path = save_backtest_results(
                backtest_results_df,
                trading_stats,
                f"aggressive_system_with_stops_{timestamp_to_string()}"
            )
            if results_path:
                print(f"✓ Результати збережено")

        return backtest_results_df, trading_metrics, trading_stats

    except Exception as e:
        print(f"❌ Помилка при тестуванні агресивної торгової системи: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())
        return pd.DataFrame(), {}, {}


def create_aggressive_summary_report(prediction_metrics, trading_metrics, save_path=None):
    """
    📊 ВИПРАВЛЕНА функція створення підсумкового звіту
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
    report.append(
        f"- Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")
    report.append("")

    report.append("## 🎯 Цілі агресивної стратегії")
    report.append("- **Основна ціль**: Збільшити кількість трейдів з 7 до 50+")
    report.append("- **Додаткові цілі**: Врахувати комісії та проскальзування")
    report.append("- **Реалістичність**: Максимально наблизити до реальної торгівлі")
    report.append("")

    report.append("## 📊 Результати прогнозування")
    if prediction_metrics:
        for metric, value in prediction_metrics.items():
            # 🔧 ВИПРАВЛЕННЯ: Перевіряємо тип даних перед форматуванням
            if isinstance(value, dict):
                report.append(f"- {metric}:")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        report.append(f"  - {k}: {v:.4f}")
                    else:
                        report.append(f"  - {k}: {v}")
            elif isinstance(value, (list, tuple)):
                # Для списків та кортежів
                report.append(f"- {metric}: {value}")
            elif isinstance(value, (int, float)):
                # Тільки для чисел застосовуємо .4f
                report.append(f"- {metric}: {value:.4f}")
            else:
                # Для всіх інших типів - як є
                report.append(f"- {metric}: {value}")
    else:
        report.append("- Метрики прогнозування недоступні")
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
            # 🔧 ВИПРАВЛЕННЯ: Така ж перевірка для торгових метрик
            if isinstance(value, dict):
                report.append(f"- {metric}:")
                for k, v in value.items():
                    if isinstance(v, (int, float)):
                        report.append(f"  - {k}: {v:.4f}")
                    else:
                        report.append(f"  - {k}: {v}")
            elif isinstance(value, (list, tuple)):
                report.append(f"- {metric}: {value}")
            elif isinstance(value, (int, float)):
                report.append(f"- {metric}: {value:.4f}")
            else:
                report.append(f"- {metric}: {value}")

        report.append("")
        report.append(f"### 💰 Торгові витрати:")
        if isinstance(total_costs, (int, float)):
            report.append(f"- Загальні витрати: ${total_costs:.2f}")
        else:
            report.append(f"- Загальні витрати: {total_costs}")

        cost_ratio = trading_metrics.get('cost_ratio', 0)
        if isinstance(cost_ratio, (int, float)):
            report.append(f"- Витрати у відсотках: {cost_ratio:.2f}%")
        else:
            report.append(f"- Витрати у відсотках: {cost_ratio}")

    else:
        report.append("- Агресивна торгова стратегія не була протестована")
    report.append("")

    report_text = "\n".join(report)

    if save_path:
        try:
            ensure_dir(os.path.dirname(save_path))
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ Звіт збережено в {save_path}")
        except Exception as e:
            print(f"❌ Помилка збереження звіту: {e}")

    print(report_text)
    return report_text


# 🔧 АЛЬТЕРНАТИВНЕ ВИПРАВЛЕННЯ: Більш надійна функція форматування

def safe_format_value(value, decimal_places=4):
    """
    🛡️ Безпечне форматування значень різних типів
    """
    if isinstance(value, (int, float)):
        if decimal_places == 0:
            return f"{value:.0f}"
        else:
            return f"{value:.{decimal_places}f}"
    elif isinstance(value, (list, tuple)):
        if len(value) <= 5:  # Короткі списки показуємо повністю
            return str(value)
        else:  # Довгі списки скорочуємо
            return f"[{value[0]}, {value[1]}, ..., {value[-1]}] (total: {len(value)} items)"
    elif isinstance(value, dict):
        if len(value) <= 3:  # Маленькі словники показуємо повністю
            return str(value)
        else:  # Великі словники скорочуємо
            keys = list(value.keys())[:3]
            return f"{{{keys[0]}: {value[keys[0]]}, {keys[1]}: {value[keys[1]]}, ...}} (total: {len(value)} keys)"
    else:
        return str(value)


# Використання safe_format_value:
def create_aggressive_summary_report_safe(prediction_metrics, trading_metrics, save_path=None):
    """
    📊 НАЙБЕЗПЕЧНІША версія звіту
    """
    print("\n" + "=" * 50)
    print("🔥 ПІДСУМКОВИЙ ЗВІТ АГРЕСИВНОЇ СИСТЕМИ")
    print("=" * 50)

    report = []
    report.append("# 🔥 Звіт про роботу агресивної системи прогнозування Bitcoin")
    report.append(f"Дата створення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Параметри системи
    report.append("## ⚙️ Параметри системи")
    try:
        report.append(f"- Горизонт прогнозування: {CONFIG['models']['n_forecast_periods']} годин")
        report.append(f"- Таймфрейм даних: {CONFIG['data']['timeframe']}")
        report.append(f"- Період даних: {CONFIG['data']['period']}")
        report.append(
            f"- Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")
    except:
        report.append("- Не вдалося отримати параметри системи")
    report.append("")

    # Результати прогнозування
    report.append("## 📊 Результати прогнозування")
    if prediction_metrics:
        for metric, value in prediction_metrics.items():
            formatted_value = safe_format_value(value)
            report.append(f"- {metric}: {formatted_value}")
    else:
        report.append("- Метрики прогнозування недоступні")
    report.append("")

    # Торгові результати
    report.append("## 🔥 Результати агресивної торгової стратегії")
    if trading_metrics:
        for metric, value in trading_metrics.items():
            formatted_value = safe_format_value(value)
            report.append(f"- {metric}: {formatted_value}")
    else:
        report.append("- Агресивна торгова стратегія не була протестована")
    report.append("")

    report_text = "\n".join(report)

    if save_path:
        try:
            ensure_dir(os.path.dirname(save_path))
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ Звіт збережено в {save_path}")
        except Exception as e:
            print(f"❌ Помилка збереження звіту: {e}")

    print(report_text)
    return report_text

def main():
    """🔥 Головна функція з агресивною торговою стратегією та стоп-лосом (нова версія)"""
    comprehensive_seed_fix(42)
    ensure_deterministic_training()
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

        # 🔥 Крок 5: АГРЕСИВНА торгова стратегія з новими модулями
        backtest_results, trading_metrics, trading_stats = run_aggressive_trading_strategy(
            pipeline, test_features, feature_names_for_model
        )
        results['trading_metrics'] = trading_metrics
        results['trading_stats'] = trading_stats
        results['backtest_results'] = backtest_results
        results['pipeline'] = pipeline

        # Створення підсумкового звіту
        ensure_dir(CONFIG['saving']['results_dir'])
        report_path = os.path.join(
            CONFIG['saving']['results_dir'],
            f"aggressive_summary_report_with_stops_{timestamp_to_string()}.md"
        )

        create_aggressive_summary_report_safe(prediction_metrics, trading_metrics, report_path)
        print("\n🎉 Система агресивного прогнозування та торгівлі з стоп-лосом успішно завершила роботу!")

        # Фінальний підсумок мети
        total_trades = trading_stats.get('total_trades', 0)
        stop_loss_count = trading_stats.get('stop_loss_triggered', 0)
        take_profit_count = trading_stats.get('take_profit_triggered', 0)

        print(f"\n🏆 ФІНАЛЬНІ РЕЗУЛЬТАТИ:")
        print(f"   📊 Кількість трейдів: {total_trades}")
        print(f"   🛡️ Стоп-лос спрацював: {stop_loss_count} разів")
        print(f"   🎯 Тейк-профіт спрацював: {take_profit_count} разів")

        if total_trades >= 30:
            print(f"🏆 ВІДМІННИЙ РЕЗУЛЬТАТ: {total_trades} трейдів з ризик-менеджментом!")
        elif total_trades >= 15:
            print(f"✅ ХОРОШИЙ РЕЗУЛЬТАТ: {total_trades} трейдів, можна покращити налаштування")
        else:
            print(f"⚠️  Потрібно налаштування: {total_trades} трейдів з {total_trades}")

        # Оцінка ефективності ризик-менеджменту
        total_risk_actions = stop_loss_count + take_profit_count
        if total_risk_actions > 0:
            success_rate = take_profit_count / total_risk_actions * 100
            print(f"🛡️ Ефективність ризик-менеджменту: {success_rate:.1f}% успішних закриттів")
        # В main() в самому кінці
        print("\n🔮 ФІНАЛЬНІ ПРОГНОЗИ:")
        latest_predictions = pipeline.get_next_predictions(X_test, n_steps=3)
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
        print("5. Перевірте наявність файлів aggressive_trading_system.py та aggressive_backtesting.py")

        return None


def quick_test_new_modules():
    """
    ⚡ Швидкий тест нових модулів для перевірки їх роботи
    """
    print("⚡ ШВИДКИЙ ТЕСТ НОВИХ МОДУЛІВ")
    print("=" * 40)

    try:
        # Тест імпортів
        print("🔍 Тестування імпортів...")
        from aggressive_trading_system_SL import AggressiveTradingSystem
        from aggressive_backtesting_SL import run_aggressive_backtest, plot_backtest_results
        print("✅ Імпорти успішні")

        # Створення фіктивної моделі
        class TestModel:
            def predict(self, features):
                return [50000]  # Завжди повертає $50,000

        # Створення системи
        print("🔍 Тестування створення торгової системи...")
        test_model = TestModel()
        trading_system = AggressiveTradingSystem(
            prediction_model=test_model,
            initial_balance=10000
        )
        print("✅ Торгова система створена")

        # Створення мінімальних тестових даних
        print("🔍 Створення тестових даних...")
        dates = pd.date_range('2023-01-01', periods=24, freq='H')
        test_data = pd.DataFrame({
            'timestamp': dates,
            'close': [50000 + i * 10 for i in range(24)],
            'open': [50000 + i * 10 - 5 for i in range(24)],
            'high': [50000 + i * 10 + 20 for i in range(24)],
            'low': [50000 + i * 10 - 20 for i in range(24)],
            'volume': [1000] * 24,
            'rsi': [50] * 24,
            'macd': [0] * 24,
            'macd_signal': [0] * 24,
            'adx': [25] * 24,
            'volume_ratio_20': [1.0] * 24,
            'close_lag_1': [50000 + i * 10 - 10 for i in range(24)],
            'sma_20': [50000 + i * 10 for i in range(24)]
        })

        feature_names = ['close_lag_1', 'sma_20', 'rsi', 'macd', 'volume_ratio_20']
        print("✅ Тестові дані створені")

        # Тест бектесту
        print("🔍 Тестування бектесту...")
        results = run_aggressive_backtest(trading_system, test_data, feature_names)

        if not results.empty:
            print("✅ Бектест пройшов успішно!")
            print(f"   Оброблено {len(results)} точок")
            executed_trades = len(results[results['executed'] == True])
            print(f"   Виконано {executed_trades} трейдів")

            # Тест статистики
            stats = trading_system.get_trading_statistics()
            print(f"   Статистика: {len(stats)} метрик")

        else:
            print("⚠️ Бектест повернув порожні результати")

        print("\n🎉 ВСІ ТЕСТИ ПРОЙШЛИ УСПІШНО!")
        print("🚀 Нові модулі готові до використання")

        return True

    except Exception as e:
        print(f"❌ Помилка в тестуванні: {e}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Спочатку проводимо швидкий тест нових модулів
    print("🔧 Перевірка нових модулів перед запуском...")
    if quick_test_new_modules():
        print("\n" + "=" * 50)
        # Запускаємо основну програму
        results = main()
        if results:
            print(f"\n✅ Результати доступні в змінній 'results'")
            print(f"🔍 Для детального аналізу дивіться збережені файли в папці '{CONFIG['saving']['results_dir']}'")

            # Короткий підсумок результатів
            trading_stats = results.get('trading_stats', {})
            if trading_stats:
                total_trades = trading_stats.get('total_trades', 0)
                total_fees = trading_stats.get('total_fees_paid', 0)
                stop_loss_count = trading_stats.get('stop_loss_triggered', 0)
                take_profit_count = trading_stats.get('take_profit_triggered', 0)
                print(f"\n📊 ШВИДКИЙ ПІДСУМОК:")
                print(f"   Трейдів виконано: {total_trades}")
                print(f"   Комісій сплачено: ${total_fees:.2f}")
                print(f"   Стоп-лос спрацював: {stop_loss_count}")
                print(f"   Тейк-профіт спрацював: {take_profit_count}")

                # Оцінка загальної ефективності
                if total_trades > 20 and take_profit_count > stop_loss_count:
                    print("🎉 СТРАТЕГІЯ ПОКАЗУЄ ГАРНІ РЕЗУЛЬТАТИ!")
                elif total_trades > 10:
                    print("✅ СТРАТЕГІЯ ПРАЦЮЄ, МОЖНА ПОКРАЩИТИ")
                else:
                    print("⚠️ ПОТРІБНЕ ДОДАТКОВЕ НАЛАШТУВАННЯ")
    else:
        print("❌ Тест нових модулів не пройшов. Перевірте файли:")
        print("   - aggressive_trading_system.py")
        print("   - aggressive_backtesting.py")
        print("   - Переконайтеся, що вони знаходяться в тій же папці")