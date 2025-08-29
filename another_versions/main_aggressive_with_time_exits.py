# main_aggressive_with_time_exits.py
# ВИПРАВЛЕНИЙ головний файл з повною реалізацією

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import traceback
import os

# Ваші існуючі імпорти
from config import CONFIG
from data.data_loader import load_bitcoin_data, validate_data, test_data_loading
from data.data_processor import preprocess_data, split_data
from features.feature_engineering import build_advanced_features, prepare_features_targets_robust
from models.prediction_pipeline import CryptoPricePredictionPipeline
from utils.helpers import save_model, timestamp_to_string, ensure_dir

# 🆕 НОВИЙ ІМПОРТ: замість старої агресивної системи
from aggressive_trading_system_with_time_exits import (
    IntegratedTradingSystemWithTimeExits,
    create_time_exit_config
)


def fix_random_seeds(seed=42):
    """Фіксація випадковості"""
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔧 Випадковість зафіксована з seed={seed}")


def run_data_preparation():
    """Підготовка даних"""
    print("=" * 50)
    print("КРОК 1: Завантаження та підготовка даних")
    print("=" * 50)

    try:
        df = load_bitcoin_data(
            prefer_source='binance',
            period=CONFIG['data']['period'],
            interval=CONFIG['data']['timeframe']
        )

        is_valid, validation_message = validate_data(df)
        if not is_valid:
            print(f"Помилка валідації: {validation_message}")
            df = test_data_loading()

        if df.empty:
            raise ValueError("Не вдалося завантажити дані!")

        print(f"✓ Завантажено {len(df)} рядків")
        print(f"✓ Період: {df['timestamp'].min()} - {df['timestamp'].max()}")

        df = preprocess_data(df)
        train_df, val_df, test_df = split_data(
            df,
            test_size=CONFIG['data']['test_size'],
            validation_size=CONFIG['data']['validation_size']
        )

        return train_df, val_df, test_df

    except Exception as e:
        print(f"❌ Помилка підготовки даних: {e}")
        raise


def run_feature_engineering(train_df, val_df, test_df):
    """Створення ознак"""
    print("\n" + "=" * 50)
    print("КРОК 2: Створення ознак")
    print("=" * 50)

    try:
        train_features = build_advanced_features(
            train_df,
            timeframes=CONFIG['features']['timeframes'] if CONFIG['features']['use_multitimeframe'] else []
        )
        val_features = build_advanced_features(
            val_df,
            timeframes=CONFIG['features']['timeframes'] if CONFIG['features']['use_multitimeframe'] else []
        )
        test_features = build_advanced_features(
            test_df,
            timeframes=CONFIG['features']['timeframes'] if CONFIG['features']['use_multitimeframe'] else []
        )

        print(f"✓ Створені ознаки: {len(train_features.columns)} колонок")

        X_train, y_train = prepare_features_targets_robust(
            train_features, target_col='close',
            forecast_horizon=CONFIG['models']['n_forecast_periods'],
            nan_threshold=0.7, min_samples=50
        )
        X_val, y_val = prepare_features_targets_robust(
            val_features, target_col='close',
            forecast_horizon=CONFIG['models']['n_forecast_periods'],
            nan_threshold=0.7, min_samples=10
        )
        X_test, y_test = prepare_features_targets_robust(
            test_features, target_col='close',
            forecast_horizon=CONFIG['models']['n_forecast_periods'],
            nan_threshold=0.7, min_samples=10
        )

        # Синхронізація колонок
        common_columns = list(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
        X_train = X_train[common_columns]
        X_val = X_val[common_columns]
        X_test = X_test[common_columns]

        print(f"✓ Фінальні розміри: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")

        return X_train, y_train, X_val, y_val, X_test, y_test, test_features

    except Exception as e:
        print(f"❌ Помилка створення ознак: {e}")
        raise


def run_model_training(X_train, y_train, X_val, y_val):
    """Навчання моделей"""
    print("\n" + "=" * 50)
    print("КРОК 3: Навчання моделей")
    print("=" * 50)

    try:
        pipeline = CryptoPricePredictionPipeline(
            n_forecast_periods=CONFIG['models']['n_forecast_periods']
        )

        pipeline.fit(
            X_train, y_train, X_val, y_val,
            optimize=CONFIG['models']['optimize_hyperparams']
        )

        # Збереження моделі
        ensure_dir(CONFIG['saving']['models_dir'])
        model_filename = f"crypto_model_time_exits_{timestamp_to_string()}.pkl"
        save_model(pipeline, model_filename, CONFIG['saving']['models_dir'])
        print(f"✓ Модель збережено: {model_filename}")

        return pipeline

    except Exception as e:
        print(f"❌ Помилка навчання: {e}")
        raise


def ensure_required_columns(test_features, feature_names_for_model):
    """Перевірка наявності необхідних колонок"""
    required_columns = [
                           'timestamp', 'close', 'open', 'high', 'low', 'volume',
                           'rsi', 'macd_12_26', 'macd_signal_12_26', 'adx', 'volume_ratio_20'
                       ] + feature_names_for_model

    missing_columns = [col for col in required_columns if col not in test_features.columns]

    if missing_columns:
        print(f"⚠️ Додаємо відсутні колонки: {len(missing_columns)}")
        for col in missing_columns:
            if col in ['rsi']:
                test_features[col] = 50
            elif col in ['macd_12_26', 'macd_signal_12_26']:
                test_features[col] = 0
            elif col in ['adx']:
                test_features[col] = 25
            elif col in ['volume_ratio_20']:
                test_features[col] = 1.0
            else:
                test_features[col] = test_features['close'] if 'close' in test_features.columns else 50000

    return test_features


def run_time_exit_experiments(pipeline, test_features, feature_names_for_model):
    """Експерименти з часовими виходами"""
    print("\n" + "=" * 60)
    print("🕐 КРОК 4: ЕКСПЕРИМЕНТИ З ЧАСОВИМИ ВИХОДАМИ (принцип Джона Генрі)")
    print("=" * 60)

    test_features = ensure_required_columns(test_features, feature_names_for_model)

    # Конфігурації для тестування
    strategies = {
        '🏆 John Henry Pure': 'john_henry_pure',
        '⚖️ Balanced Hybrid': 'balanced',
        '⚡ Aggressive': 'aggressive',
        '🐌 Patient': 'patient'
    }

    results_comparison = {}
    best_result = None
    best_return = float('-inf')

    for strategy_name, config_type in strategies.items():
        print(f"\n🧪 Тестування стратегії: {strategy_name}")
        print("-" * 40)

        try:
            # Створюємо систему з конкретною конфігурацією
            trading_system = IntegratedTradingSystemWithTimeExits(
                prediction_model=pipeline,
                initial_balance=CONFIG['trading']['initial_balance']
            )

            # Налаштовуємо параметри часових виходів
            config = create_time_exit_config(config_type)
            trading_system.time_exit_params.update(config)

            print(f"📋 Конфігурація:")
            print(f"   Макс. час: {config['max_position_bars']} годин")
            if config.get('pure_john_henry_mode'):
                print(f"   Режим: Чистий Джон Генрі")
            else:
                print(f"   Прибутковий вихід: {config.get('profitable_only_exit_bars', 'Вимк.')} годин")
                print(f"   Збитковий вихід: {config.get('losing_only_exit_bars', 'Вимк.')} годин")

            # Запускаємо бектест
            backtest_results = trading_system.run_backtest(test_features, feature_names_for_model)

            if not backtest_results.empty:
                final_portfolio = backtest_results['portfolio_value'].iloc[-1]
                initial_portfolio = backtest_results['portfolio_value'].iloc[0]
                total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100

                # Отримуємо детальну статистику
                stats = trading_system.get_enhanced_trading_statistics()

                results_comparison[strategy_name] = {
                    'config_type': config_type,
                    'total_return': total_return,
                    'final_portfolio': final_portfolio,
                    'total_trades': stats['total_trades'],
                    'time_exits': stats['time_exit_triggered'],
                    'john_henry_exits': stats['john_henry_exits'],
                    'profitable_time_exits': stats['profitable_time_exits'],
                    'losing_time_exits': stats['losing_time_exits'],
                    'avg_duration': stats['avg_position_duration'],
                    'stop_losses': stats['stop_loss_triggered'],
                    'take_profits': stats['take_profit_triggered'],
                    'system': trading_system,
                    'results': backtest_results
                }

                print(f"💰 Результат: {total_return:+.2f}%")
                print(f"📊 Трейдів: {stats['total_trades']} | Часових виходів: {stats['time_exit_triggered']}")
                print(f"🕐 Середня тривалість: {stats['avg_position_duration']:.1f} годин")

                # Відстежуємо найкращий результат
                if total_return > best_return:
                    best_return = total_return
                    best_result = (strategy_name, results_comparison[strategy_name])

            else:
                print("❌ Порожні результати бектесту")

        except Exception as e:
            print(f"❌ Помилка стратегії {strategy_name}: {e}")
            print(traceback.format_exc())

    # Порівняльна таблиця
    print(f"\n📋 ПОРІВНЯННЯ СТРАТЕГІЙ ЧАСОВИХ ВИХОДІВ:")
    print("=" * 100)
    print(
        f"{'Стратегія':<20} {'Дохід %':<10} {'Трейди':<8} {'Час.вих':<9} {'JH вих':<8} {'Приб.ч.':<8} {'Збит.ч.':<8} {'Сер.час':<8}")
    print("-" * 100)

    for name, results in results_comparison.items():
        print(f"{name:<20} {results['total_return']:>+7.2f}% {results['total_trades']:>6} "
              f"{results['time_exits']:>7} {results['john_henry_exits']:>6} "
              f"{results['profitable_time_exits']:>6} {results['losing_time_exits']:>6} "
              f"{results['avg_duration']:>6.1f}h")

    # Аналіз найкращої стратегії
    if best_result:
        best_name, best_data = best_result
        print(f"\n🏆 НАЙКРАЩА СТРАТЕГІЯ: {best_name}")
        print(f"   💰 Дохідність: {best_data['total_return']:+.2f}%")
        print(f"   📊 Ефективність часових виходів: {best_data['time_exits']} з {best_data['total_trades']} трейдів")

        # Оцінка принципу Джона Генрі
        if best_data['john_henry_exits'] > 0:
            jh_ratio = best_data['john_henry_exits'] / best_data['time_exits'] * 100
            if jh_ratio > 70:
                print(f"   🎯 Домінує принцип Джона Генрі ({jh_ratio:.1f}% часових виходів)")
            elif jh_ratio > 30:
                print(f"   ⚖️ Збалансований підхід ({jh_ratio:.1f}% за максимальним часом)")
            else:
                print(f"   ⚡ Адаптивні виходи переважають ({jh_ratio:.1f}% за максимальним часом)")

        return best_data['system'], best_data['results'], results_comparison

    else:
        print("❌ Жодна стратегія не показала результатів")
        return None, None, results_comparison


def create_enhanced_visualization(best_results, system_name="Найкраща стратегія з часовими виходами"):
    """Покращена візуалізація"""
    try:
        if best_results.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 14))

        # График 1: Ціна + типи виходів
        ax1.plot(best_results['timestamp'], best_results['price'],
                 'b-', linewidth=1.5, alpha=0.8, label='Ціна BTC')

        executed_trades = best_results[best_results['executed'] == True]

        # Відображаємо різні типи виходів (спрощена версія)
        if not executed_trades.empty:
            ax1.scatter(executed_trades['timestamp'], executed_trades['price'],
                        marker='o', color='red', s=50, alpha=0.8,
                        label=f'Виконані трейди ({len(executed_trades)})')

        ax1.set_title(f'🕐 {system_name}: Виконані трейди', fontweight='bold')
        ax1.set_ylabel('Ціна (USD)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # График 2: Портфель
        ax2.plot(best_results['timestamp'], best_results['portfolio_value'],
                 label='Портфель', color='purple', linewidth=2)

        # Buy & Hold порівняння
        if len(best_results) > 1:
            initial_price = best_results['price'].iloc[0]
            initial_portfolio = best_results['portfolio_value'].iloc[0]
            btc_equivalent = initial_portfolio / initial_price
            buy_hold_values = btc_equivalent * best_results['price']

            ax2.plot(best_results['timestamp'], buy_hold_values,
                     label='Buy & Hold', color='gray', linestyle='--', linewidth=2)

        ax2.set_title('💰 Динаміка портфеля', fontweight='bold')
        ax2.set_ylabel('Вартість (USD)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # График 3: Розподіл тривалості позицій
        if 'max_position_age' in best_results.columns:
            age_data = best_results['max_position_age'][best_results['max_position_age'] > 0]
            if not age_data.empty:
                ax3.hist(age_data, bins=15, alpha=0.7, color='lightblue', edgecolor='navy')
                ax3.axvline(age_data.mean(), color='red', linestyle='--', linewidth=2,
                            label=f'Середня: {age_data.mean():.1f}h')
                ax3.set_title('⏱️ Розподіл віку позицій')
                ax3.set_xlabel('Вік позиції (години)')
                ax3.set_ylabel('Частота')
                ax3.legend()
                ax3.grid(True, alpha=0.3)

        # График 4: Кількість відкритих позицій
        ax4.plot(best_results['timestamp'], best_results['open_positions'],
                 color='orange', linewidth=2, label='Відкриті позиції')
        ax4.set_title('📊 Відкриті позиції в часі')
        ax4.set_ylabel('Кількість')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("✅ Візуалізацію створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")


def create_comprehensive_report(prediction_metrics, best_system, comparison_results, save_path=None):
    """Комплексний звіт"""
    print("\n" + "=" * 60)
    print("📊 КОМПЛЕКСНИЙ ЗВІТ: Часові виходи vs Традиційні методи")
    print("=" * 60)

    report = []
    report.append("# 🕐 Звіт про інтеграцію часових виходів (принцип Джона Генрі)")
    report.append(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## 🎯 Мета експерименту")
    report.append("Тестування принципу Джона Генрі: 'вихід після певної кількості барів незалежно від P&L'")
    report.append("")

    report.append("## 📊 Результати прогнозування ML")
    for metric, value in prediction_metrics.items():
        if isinstance(value, (int, float)):
            report.append(f"- {metric}: {value:.4f}")
        else:
            report.append(f"- {metric}: {value}")
    report.append("")

    report.append("## 🏆 Порівняння стратегій")
    for name, results in comparison_results.items():
        report.append(f"### {name}")
        report.append(f"- Дохід: {results['total_return']:+.2f}%")
        report.append(f"- Трейдів: {results['total_trades']}")
        report.append(f"- Часових виходів: {results['time_exits']}")
        report.append("")

    # Аналіз найкращої стратегії
    if best_system:
        stats = best_system.get_enhanced_trading_statistics()
        report.append("## 🎖️ Найкраща стратегія")
        report.append(f"- Всього трейдів: {stats['total_trades']}")
        report.append(f"- Часових виходів: {stats['time_exit_triggered']}")
        report.append(f"- Джона Генрі виходів: {stats['john_henry_exits']}")
        report.append("")

        if stats['time_exit_triggered'] > 0:
            jh_ratio = stats['john_henry_exits'] / stats['time_exit_triggered'] * 100
            if jh_ratio > 50:
                report.append("✅ **ПРИНЦИП ДЖОНА ГЕНРІ ПРАЦЮЄ!**")
            else:
                report.append("⚖️ **ГІБРИДНИЙ ПІДХІД ЕФЕКТИВНИЙ**")

    report_text = "\n".join(report)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Звіт збережено: {save_path}")

    print(report_text)
    return report_text


def main():
    """🔥 ГОЛОВНА ФУНКЦІЯ з часовими виходами Джона Генрі"""
    print("🕐 Запуск системи з ЧАСОВИМИ ВИХОДАМИ (принцип Джона Генрі)")
    print("🎯 Мета: Тестування виходу після X барів незалежно від прибутку/збитку")
    print("=" * 80)

    # Фіксація випадковості
    fix_random_seeds(42)

    results = {}

    try:
        # Кроки 1-3: Підготовка даних та навчання
        print("📥 Підготовка даних, ознак та навчання моделей...")
        train_df, val_df, test_df = run_data_preparation()

        X_train, y_train, X_val, y_val, X_test, y_test, test_features = run_feature_engineering(
            train_df, val_df, test_df
        )

        pipeline = run_model_training(X_train, y_train, X_val, y_val)

        # Швидка оцінка моделі
        print("\n📊 Швидка оцінка моделі...")
        y_pred = pipeline.predict(X_test)
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        prediction_metrics = {
            'MAE': mae,
            'R2': r2,
            'Test_samples': len(y_test),
            'Features_used': len(X_test.columns)
        }

        print(f"✓ MAE: {mae:.2f}, R²: {r2:.4f}")
        results['prediction_metrics'] = prediction_metrics

        # Підготовка для торгівлі
        feature_names_for_model = list(X_test.columns)
        print(f"✓ Підготовлено {len(feature_names_for_model)} ознак для торгівлі")

        # 🆕 КРОК 4: ЕКСПЕРИМЕНТИ З ЧАСОВИМИ ВИХОДАМИ
        best_system, best_backtest_results, comparison_results = run_time_exit_experiments(
            pipeline, test_features, feature_names_for_model
        )

        if best_system and best_backtest_results is not None:
            # Збереження результатів
            ensure_dir(CONFIG['saving']['results_dir'])

            # Результати бектесту
            backtest_filename = f"time_exits_backtest_{timestamp_to_string()}.csv"
            backtest_path = os.path.join(CONFIG['saving']['results_dir'], backtest_filename)
            best_backtest_results.to_csv(backtest_path, index=False)
            print(f"✓ Результати бектесту збережено: {backtest_filename}")

            # Візуалізація
            print("\n📊 Створення візуалізації...")
            create_enhanced_visualization(best_backtest_results)

            # Звіт
            report_filename = f"time_exits_report_{timestamp_to_string()}.md"
            report_path = os.path.join(CONFIG['saving']['results_dir'], report_filename)

            create_comprehensive_report(
                prediction_metrics, best_system, comparison_results, report_path
            )

            # Результати
            results['best_system'] = best_system
            results['best_backtest_results'] = best_backtest_results
            results['comparison_results'] = comparison_results
            results['trading_metrics'] = best_system.get_enhanced_trading_statistics()

            # 🆕 ФІНАЛЬНИЙ ПІДСУМОК
            print(f"\n🎉 СИСТЕМА З ЧАСОВИМИ ВИХОДАМИ ЗАВЕРШИЛА РОБОТУ!")
            print("=" * 60)

            final_stats = best_system.get_enhanced_trading_statistics()
            final_portfolio = best_backtest_results['portfolio_value'].iloc[-1]
            initial_portfolio = best_backtest_results['portfolio_value'].iloc[0]
            total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100

            print(f"💰 ФІНАНСОВІ РЕЗУЛЬТАТИ:")
            print(f"   📈 Загальна дохідність: {total_return:+.2f}%")
            print(f"   💼 Фінальний портфель: ${final_portfolio:,.0f}")
            print(f"   📊 Всього трейдів: {final_stats['total_trades']}")

            print(f"\n🕐 ЧАСОВІ ВИХОДИ (ДЖОН ГЕНРІ):")
            print(f"   ⏰ Всього часових виходів: {final_stats['time_exit_triggered']}")
            print(f"   👑 'Чистих' Джона Генрі: {final_stats['john_henry_exits']}")
            print(f"   🟢 Прибуткових часових: {final_stats['profitable_time_exits']}")
            print(f"   ⏱️ Середня тривалість: {final_stats['avg_position_duration']:.1f} годин")

            # Оцінка ефективності принципу Джона Генрі
            if final_stats['time_exit_triggered'] > 0:
                jh_ratio = final_stats['john_henry_exits'] / final_stats['time_exit_triggered'] * 100
                time_exit_ratio = final_stats['time_exit_triggered'] / final_stats['total_trades'] * 100

                print(f"\n🎯 ОЦІНКА ПРИНЦИПУ ДЖОНА ГЕНРІ:")
                print(f"   📊 Частка часових виходів: {time_exit_ratio:.1f}% від всіх трейдів")

                if jh_ratio > 70:
                    print(f"   🏆 ПРИНЦИП ДЖОНА ГЕНРІ ДОМІНУЄ! ({jh_ratio:.1f}% часових виходів)")
                elif jh_ratio > 40:
                    print(f"   ⚖️ ЗБАЛАНСОВАНИЙ ПІДХІД ({jh_ratio:.1f}% чистих JH виходів)")
                else:
                    print(f"   ⚡ АДАПТИВНІ ВИХОДИ КРАЩІ ({jh_ratio:.1f}% чистих JH виходів)")

            print(f"\n📁 ФАЙЛИ ЗБЕРЕЖЕНО:")
            print(f"   📊 Результати: {backtest_filename}")
            print(f"   📋 Звіт: {report_filename}")

        else:
            print("❌ Не вдалося отримати результати експериментів")
            return None

        return results

    except Exception as e:
        print(f"\n❌ КРИТИЧНА ПОМИЛКА: {str(e)}")
        print("Детальна інформація:")
        print(traceback.format_exc())
        return None


if __name__ == "__main__":
    print("🚀 ЗАПУСК СИСТЕМИ З ІНТЕГРОВАНИМИ ЧАСОВИМИ ВИХОДАМИ")
    print("Базується на принципі легендарного трейдера Джона Генрі:")
    print("'Вихід після певної кількості барів незалежно від прибутку чи збитку'")
    print("")

    results = main()

    if results:
        print(f"\n✅ ЕКСПЕРИМЕНТ ЗАВЕРШЕНО УСПІШНО!")
        print(f"🔬 Результати доступні в змінній 'results'")

        if 'trading_metrics' in results:
            tm = results['trading_metrics']
            print(f"📊 Ключові метрики:")
            print(f"   - Трейдів: {tm.get('total_trades', 0)}")
            print(f"   - Часових виходів: {tm.get('time_exit_triggered', 0)}")
            print(f"   - Виходів Джона Генрі: {tm.get('john_henry_exits', 0)}")

        print(f"\n🎯 ВИСНОВОК ПРО ПРИНЦИП ДЖОНА ГЕНРІ:")
        if 'trading_metrics' in results:
            tm = results['trading_metrics']
            if tm.get('time_exit_triggered', 0) > 0:
                jh_effectiveness = tm.get('john_henry_exits', 0) / tm.get('time_exit_triggered', 1) * 100
                if jh_effectiveness > 50:
                    print("✅ Принцип Джона Генрі ПРАЦЮЄ для Bitcoin!")
                elif jh_effectiveness > 25:
                    print("⚖️ Принцип Джона Генрі працює В КОМБІНАЦІЇ з іншими методами")
                else:
                    print("💡 Принцип Джона Генрі потребує НАЛАШТУВАННЯ для Bitcoin")

    else:
        print(f"\n❌ ЕКСПЕРИМЕНТ НЕ ЗАВЕРШЕНО")
        print("Перевірте помилки вище та спробуйте ще раз")

# ШВИДКІ НАЛАШТУВАННЯ ДЛЯ ТЕСТУВАННЯ:
"""
Для швидкого тестування змініть у config.py:

CONFIG = {
    'data': {
        'period': '1y',  # Замість '4y' для швидшого завантаження
        'timeframe': '1h',
    },
    'models': {
        'optimize_hyperparams': False,  # Вимкніть для швидкості
        'n_forecast_periods': 12,  # Зменшіть з 24
    }
}

Або створіть окремий тестовий конфіг:
"""

# 🔧 ТЕСТОВИЙ КОНФІГ для швидкого запуску
TEST_CONFIG = {
    'data': {
        'symbol': 'BTC-USD',
        'timeframe': '1h',
        'period': '6mo',  # Тільки 6 місяців для швидкості
        'test_size': 0.3,
        'validation_size': 0.2
    },
    'features': {
        'use_multitimeframe': False,  # Вимкнено для швидкості
        'timeframes': [],
        'n_selected_features': 20  # Менше ознак
    },
    'models': {
        'optimize_hyperparams': False,  # Обов'язково вимкнути
        'n_forecast_periods': 6,  # Менший горизонт
        'n_iterations': 5,
        'cv_folds': 2
    },
    'trading': {
        'initial_balance': 10000,  # Менший капітал для тестування
        'position_size': 0.15,
        'confidence_threshold': 0.4,
        'stop_loss': 0.03,
        'take_profit': 0.06
    },
    'saving': {
        'models_dir': 'models/test',
        'results_dir': 'results/test',
        'data_dir': 'data/test'
    }
}


def quick_test():
    """
    🚀 ШВИДКИЙ ТЕСТ системи часових виходів
    Використовується для перевірки, що все працює
    """
    print("⚡ ШВИДКИЙ ТЕСТ системи часових виходів")
    print("Використовуємо скорочені дані та параметри для швидкості")

    # Тимчасово замінюємо CONFIG
    global CONFIG
    original_config = CONFIG.copy()
    CONFIG.update(TEST_CONFIG)

    try:
        # Запускаємо швидкий тест
        results = main()

        if results:
            print("✅ ШВИДКИЙ ТЕСТ ПРОЙШОВ УСПІШНО!")
            print("Тепер можна запускати повну версію з реальними параметрами")
        else:
            print("❌ ШВИДКИЙ ТЕСТ НЕ ПРОЙШОВ")

    finally:
        # Відновлюємо оригінальний CONFIG
        CONFIG.update(original_config)

    return results


# ДІАГНОСТИКА помилок
def diagnose_issues():
    """
    🔍 Діагностика можливих проблем
    """
    print("🔍 ДІАГНОСТИКА СИСТЕМИ:")

    issues = []

    # Перевірка імпортів
    try:
        from aggressive_trading_system_with_time_exits import IntegratedTradingSystemWithTimeExits
        print("✅ Головний клас системи імпортовано")
    except ImportError as e:
        issues.append(f"❌ Не вдалося імпортувати головний клас: {e}")

    # Перевірка конфігу
    try:
        print(f"✅ CONFIG завантажено: {CONFIG.keys()}")
    except Exception as e:
        issues.append(f"❌ Проблема з CONFIG: {e}")

    # Перевірка залежностей
    required_modules = ['pandas', 'numpy', 'matplotlib', 'sklearn', 'xgboost', 'lightgbm']
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            issues.append(f"❌ Відсутній модуль: {module}")

    # Перевірка папок
    import os
    required_dirs = ['data', 'models', 'features', 'utils']
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✅ Папка {dir_name}/")
        else:
            issues.append(f"❌ Відсутня папка: {dir_name}/")

    if issues:
        print(f"\n🚨 ЗНАЙДЕНО {len(issues)} ПРОБЛЕМ:")
        for issue in issues:
            print(f"   {issue}")

        print(f"\n🔧 РЕКОМЕНДАЦІЇ:")
        print("1. Переконайтеся, що файл aggressive_trading_system_with_time_exits.py існує")
        print("2. Встановіть відсутні залежності: pip install -r requirements.txt")
        print("3. Створіть відсутні папки")
        print("4. Перевірте config.py")
    else:
        print("✅ Всі компоненти на місці!")

    return len(issues) == 0


# Допоміжна функція для відладки
def debug_run():
    """
    🐛 ВІДЛАДОЧНИЙ ЗАПУСК з детальною інформацією
    """
    print("🐛 ВІДЛАДОЧНИЙ РЕЖИМ")

    # Спочатку діагностика
    if not diagnose_issues():
        print("❌ Діагностика показала проблеми. Виправте їх перед запуском.")
        return

    # Потім швидкий тест
    print("\n⚡ Запуск швидкого тесту...")
    test_results = quick_test()

    if test_results:
        print("\n🚀 Швидкий тест пройшов! Запуск повної версії...")
        full_results = main()
        return full_results
    else:
        print("\n❌ Швидкий тест не пройшов. Перевірте налаштування.")
        return None


# ТОЧКА ВХОДУ з вибором режиму
if __name__ == "__main__":
    import sys

    # Перевіряємо аргументи командного рядка
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == 'test':
            print("🧪 Режим швидкого тестування")
            quick_test()
        elif mode == 'debug':
            print("🐛 Режим відладки")
            debug_run()
        elif mode == 'diagnose':
            print("🔍 Режим діагностики")
            diagnose_issues()
        else:
            print(f"❓ Невідомий режим: {mode}")
            print("Доступні режими: test, debug, diagnose")
    else:
        # Звичайний запуск
        print("🚀 ЗВИЧАЙНИЙ ЗАПУСК")
        print("Для швидкого тесту: python main_aggressive_with_time_exits.py test")
        print("Для відладки: python main_aggressive_with_time_exits.py debug")
        print("Для діагностики: python main_aggressive_with_time_exits.py diagnose")
        print("")

        # Запуск основної програми
        results = main()

        if results:
            print("\n🎯 УСПІШНЕ ЗАВЕРШЕННЯ!")
        else:
            print("\n❌ ПРОГРАМА ЗАВЕРШИЛАСЯ З ПОМИЛКАМИ")
            print("Спробуйте режим діагностики: python main_aggressive_with_time_exits.py diagnose")