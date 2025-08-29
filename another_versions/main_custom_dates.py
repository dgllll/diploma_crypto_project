# main_simple_2024.py - Спрощена версія без мультитаймфрейму

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import ccxt

    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("⚠️ УВАГА: ccxt не встановлено. Встановіть: pip install ccxt")

from config import CONFIG
from aggressive_trading_system import AggressiveTradingSystem, plot_backtest_results_standalone


def run_simple_2024():
    """
    🎯 Спрощений запуск для 2024 року БЕЗ мультитаймфрейм ознак
    """

    if not BINANCE_AVAILABLE:
        print("❌ ccxt не встановлено. Встановіть: pip install ccxt")
        return None

    print("🎯 СПРОЩЕНИЙ ЗАПУСК ДЛЯ 2024 РОКУ")
    print("📅 БЕЗ мультитаймфрейм ознак для уникнення конфліктів")
    print("=" * 60)

    try:
        # 1. Завантаження даних з Binance
        from custom_data_loader import load_bitcoin_binance_dates

        print("📊 Завантаження даних 2024...")
        df = load_bitcoin_binance_dates('2024-01-01', '2024-12-31', '1h')

        if df.empty:
            print("❌ Не вдалося завантажити дані")
            return None

        # 2. Обробка даних
        print("🔧 Обробка даних...")
        from data.data_processor import preprocess_data, split_data

        df = preprocess_data(df)
        train_df, val_df, test_df = split_data(df, test_size=0.3, validation_size=0.1)

        print(f"✅ Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # 3. Feature engineering БЕЗ мультитаймфрейму
        print("🔧 Створення ознак (БЕЗ мультитаймфрейму)...")
        from features.feature_engineering import build_advanced_features, prepare_features_targets_robust

        # Створюємо ознаки БЕЗ мультитаймфрейм
        train_features = build_advanced_features(train_df, timeframes=[])  # Пустий список!
        val_features = build_advanced_features(val_df, timeframes=[])
        test_features = build_advanced_features(test_df, timeframes=[])

        # Підготовка з однаковими параметрами
        nan_threshold = 0.9  # Дуже м'який поріг

        X_train, y_train = prepare_features_targets_robust(
            train_features, 'close', 24, nan_threshold, min_samples=50)
        X_val, y_val = prepare_features_targets_robust(
            val_features, 'close', 24, nan_threshold, min_samples=10)
        X_test, y_test = prepare_features_targets_robust(
            test_features, 'close', 24, nan_threshold, min_samples=10)

        print(f"✅ Ознаки: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        # Синхронізація колонок
        common_columns = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
        X_train = X_train[common_columns]
        X_val = X_val[common_columns]
        X_test = X_test[common_columns]

        print(f"✅ Після синхронізації: {len(common_columns)} спільних колонок")

        # 4. Навчання моделі
        print("🤖 Навчання моделей...")
        from models.prediction_pipeline import CryptoPricePredictionPipeline

        pipeline = CryptoPricePredictionPipeline(n_forecast_periods=24)
        pipeline.fit(X_train, y_train, X_val, y_val, optimize=False)  # Без оптимізації для швидкості

        print("✅ Модель навчена!")

        # 5. Агресивна торгівля з дуже агресивними параметрами
        print("🔥 Агресивна торгова система...")

        trading_system = AggressiveTradingSystem(
            prediction_model=pipeline,
            initial_balance=50000  # Менший капітал для 2024
        )

        # 🔥 СУПЕР АГРЕСИВНІ ПАРАМЕТРИ для максимуму трейдів
        trading_system.trading_params.update({
            'base_confidence_threshold': 0.1,  # Дуже низький
            'min_confidence_threshold': 0.02,  # Екстремально низький
            'rsi_oversold': 55,  # Чутливіший
            'rsi_overbought': 45,
            'enable_scalping': True,
            'scalping_threshold': 0.05,  # Дуже агресивний скальпінг
            'position_size_base': 0.1,  # Менші позиції = більше трейдів
        })

        trading_system.trading_costs.update({
            'min_trade_amount': 5,  # Мінімальні угоди
            'maker_fee': 0.0002,  # Менші комісії
            'taker_fee': 0.0004,
            'slippage_pct': 0.0002,
        })

        print("🔥 Параметри торгівлі:")
        print(f"  Поріг впевненості: {trading_system.trading_params['base_confidence_threshold']}")
        print(
            f"  RSI пороги: {trading_system.trading_params['rsi_oversold']}/{trading_system.trading_params['rsi_overbought']}")
        print(f"  Мін. угода: ${trading_system.trading_costs['min_trade_amount']}")

        # 6. Бектест
        print("⚖️ Бектест системи...")
        feature_names = list(X_test.columns)
        backtest_results = trading_system.run_backtest(test_features, feature_names)

        # 7. Результати
        if not backtest_results.empty:
            print("📊 Створення візуалізації...")
            plot_backtest_results_standalone(backtest_results, "Спрощена Агресивна Система 2024")

        trading_stats = trading_system.get_trading_statistics()

        print(f"\n🎉 РЕЗУЛЬТАТИ СПРОЩЕНОЇ СИСТЕМИ 2024:")
        print(f"🎯 Трейдів: {trading_stats.get('total_trades', 0)}")
        print(f"💰 BUY: {trading_stats.get('buy_trades', 0)}")
        print(f"💰 SELL: {trading_stats.get('sell_trades', 0)}")
        print(f"⚡ Скальпінг: {trading_stats.get('scalping_trades', 0)}")
        print(f"💸 Комісій: ${trading_stats.get('total_fees_paid', 0):.2f}")

        # Оцінка досягнення мети
        total_trades = trading_stats.get('total_trades', 0)
        if total_trades >= 100:
            print(f"🏆 СУПЕР УСПІХ: {total_trades} трейдів!")
        elif total_trades >= 50:
            print(f"🎯 ЦІЛЬ ДОСЯГНУТА: {total_trades} трейдів!")
        elif total_trades >= 25:
            print(f"⚡ БЛИЗЬКО ДО ЦІЛІ: {total_trades} трейдів")
        else:
            print(f"⚠️ ПОТРІБНЕ НАЛАШТУВАННЯ: {total_trades} трейдів")

        return {
            'backtest_results': backtest_results,
            'trading_stats': trading_stats,
            'pipeline': pipeline
        }

    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        import traceback
        print("Детальна помилка:")
        print(traceback.format_exc())
        return None


def run_ultra_aggressive_2024():
    """
    🔥 УЛЬТРА-АГРЕСИВНА версія з мінімальними порогами
    """
    print("🔥 УЛЬТРА-АГРЕСИВНА СИСТЕМА 2024")
    print("🎯 МЕТА: 100+ трейдів")
    print("=" * 50)

    # Змінюємо конфігурацію на льоту
    result = run_simple_2024()

    if result and result['trading_stats']:
        total = result['trading_stats'].get('total_trades', 0)

        if total < 50:
            print(f"\n🔧 ПОТРІБНЕ ДОДАТКОВЕ НАЛАШТУВАННЯ")
            print("💡 Рекомендації:")
            print("1. Зменшити base_confidence_threshold до 0.05")
            print("2. Зменшити min_trade_amount до 1")
            print("3. Збільшити position_size_base до 0.15")
            print("4. Змінити RSI пороги на 52/48")

    return result


if __name__ == "__main__":
    print("🚀 ВИБІР РЕЖИМУ:")
    print("1. Спрощена система (звичайна)")
    print("2. Ультра-агресивна система")

    # Можете змінити тут:
    mode = 1  # 1 або 2

    if mode == 1:
        result = run_simple_2024()
    else:
        result = run_ultra_aggressive_2024()

    if result:
        print("\n✅ СИСТЕМА ЗАВЕРШИЛА РОБОТУ")
    else:
        print("\n❌ СИСТЕМА ЗАВЕРШИЛАСЬ З ПОМИЛКОЮ")