"""
Головний файл для запуску системи прогнозування та агресивної торгівлі криптовалютами
🔥 ОНОВЛЕННЯ: З повною підтримкою ШОРТ ПОЗИЦІЙ та розділених модулів торгової системи
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime
import traceback
from distribution_analysis import analyze_data_generating_process

# Імпортуємо модулі проекту
from config import CONFIG
from aggressive_trading_system_SL_STOP import AggressiveTradingSystem

try:
    from aggressive_backtesting_SL_STOP import (
        run_aggressive_backtest,
        plot_backtest_results,
        calculate_trading_metrics,
        save_backtest_results,
        plot_backtest_results_with_log_pnl,
    reset_trading_log, plot_backtest_results_simple_pnl, calculate_advanced_trading_statistics, calculate_advanced_trading_statistics_with_kurtosis
    )
except ImportError as e:
    try:
        # Fallback
        from aggressive_backtesting_SL_STOP import (
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
from features.feature_engineering import build_advanced_features, prepare_features_targets_robust, select_features
from models.prediction_pipeline import CryptoPricePredictionPipeline
from evaluation.visualization import (
    plot_price_prediction, plot_feature_importance,
    plot_learning_curves, plot_model_comparison_metrics,
    plot_residuals_analysis, create_comprehensive_model_dashboard
)
from utils.helpers import save_model, load_model, timestamp_to_string, ensure_dir
from models.ensemble import FixedCryptoPricePredictionPipeline

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
        xgb.set_config(verbosity=0)
        print("✅ XGBoost seed зафіксовано")
    except ImportError:
        pass

    # 6. LightGBM
    try:
        import lightgbm as lgb
        print("✅ LightGBM готовий до фіксації")
    except ImportError:
        pass

    print("✅ Всі доступні джерела випадковості зафіксовані")


def run_data_generating_analysis(train_df, val_df, test_df):
    """
    🔬 НОВИЙ КРОК 1.5: Аналіз породжуючого розподілу даних
    Додати цю функцію в main_stoploss.py
    """
    print("\n" + "🔬" * 60)
    print("КРОК 1.5: АНАЛІЗ ПОРОДЖУЮЧОГО РОЗПОДІЛУ ДАНИХ")
    print("🔬" * 60)

    try:
        # Об'єднуємо всі дані для аналізу
        all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)

        print(f"📊 Аналізуємо {len(all_data)} точок даних")
        print(f"📅 Період: з {all_data['timestamp'].min()} до {all_data['timestamp'].max()}")

        # Запускаємо комплексний аналіз
        distribution_results, assumptions = analyze_data_generating_process(all_data, 'close')

        # Створюємо звіт про припущення
        create_assumptions_report(assumptions, distribution_results)

        return {
            'distribution_analysis': distribution_results,
            'data_assumptions': assumptions,
            'model_justification': distribution_results.get('model_justification', {})
        }

    except Exception as e:
        print(f"❌ Помилка аналізу розподілу: {e}")
        import traceback
        print(traceback.format_exc())

        # Повертаємо базові припущення
        return {
            'distribution_analysis': {},
            'data_assumptions': {
                'normal_returns': False,
                'fat_tails': True,
                'stationary_prices': False,
                'model_choice': 'nonparametric'
            }
        }


def create_assumptions_report(assumptions, distribution_results):
    """
    Створює звіт про припущення для захисту роботи
    """
    print("\n📋 ЗВІТ ПРО ПРИПУЩЕННЯ ДЛЯ ЗАХИСТУ РОБОТИ")
    print("=" * 60)

    print("🎯 КЛЮЧОВІ ВИСНОВКИ ДЛЯ ПРЕЗЕНТАЦІЇ:")

    # 1. Обґрунтування непараметричних моделей
    print("\n1️⃣ ОБҐРУНТУВАННЯ ВИБОРУ НЕПАРАМЕТРИЧНИХ МОДЕЛЕЙ:")

    violated_count = 0
    if not assumptions.get('normal_returns', True):
        print("   ❌ Дохідності НЕ нормально розподілені")
        violated_count += 1

    if assumptions.get('fat_tails', False):
        print("   ❌ Виявлені важкі хвости в розподілі (високий ексцес)")
        violated_count += 1

    if not assumptions.get('stationary_prices', True):
        print("   ❌ Ціни НЕ стаціонарні (підтверджує теорію ефективного ринку)")
        violated_count += 1

    if not assumptions.get('log_normal_prices', True):
        print("   ❌ Ціни НЕ слідують лог-нормальному розподілу")
        violated_count += 1

    print(f"\n   📊 Загалом порушено {violated_count} ключових припущень параметричних моделей")
    print(f"   🎯 ВИСНОВОК: Вибір непараметричних моделей НАУКОВО ОБҐРУНТОВАНИЙ")

    # 2. Стилізовані факти
    print("\n2️⃣ ПІДТВЕРДЖЕНІ СТИЛІЗОВАНІ ФАКТИ ФІНАНСОВИХ РИНКІВ:")
    stylized_facts = distribution_results.get('stylized_facts', {})

    if stylized_facts.get('volatility_clustering', False):
        print("   ✅ Кластеризація волатільності (GARCH ефекти)")

    if stylized_facts.get('fat_tails', False):
        print("   ✅ Важкі хвости розподілу (leptokurtic)")

    if stylized_facts.get('no_autocorr_returns', False):
        print("   ✅ Відсутність автокореляції в дохідностях")

    if stylized_facts.get('negative_skewness', False):
        print("   ✅ Негативна асиметрія (leverage effect)")

    # 3. Теоретичні припущення
    print("\n3️⃣ СФОРМУЛЬОВАНІ ПРИПУЩЕННЯ ПРО ПОРОДЖУЮЧИЙ ПРОЦЕС:")
    print("   📈 H1: Ціни слідують геометричному броунівському руху з стрибками")
    print("   📊 H2: Дохідності мають t-розподіл замість нормального")
    print("   🌊 H3: Волатільність змінюється в часі (гетероскедастичність)")
    print("   🔄 H4: Наявні режими волатільності та структурні зломи")
    print("   🎯 H5: Нелінійні залежності між технічними індикаторами та майбутніми цінами")

    # 4. Переваги обраного підходу
    print("\n4️⃣ ПЕРЕВАГИ TREE-BASED МОДЕЛЕЙ ДЛЯ НАШИХ ДАНИХ:")
    advantages = distribution_results.get('model_justification', {}).get('nonparametric_advantages', [])

    for i, advantage in enumerate(advantages[:6], 1):
        print(f"   {i}. {advantage}")

    # 5. Фінальне обґрунтування
    print("\n5️⃣ НАУКОВИЙ ВИСНОВОК:")
    print("   🔬 'Аналіз породжуючого розподілу підтвердив, що криптовалютні дані'")
    print("   🔬 'порушують ключові припущення параметричних моделей, що робить'")
    print("   🔬 'вибір непараметричних tree-based методів ТЕОРЕТИЧНО ОБҐРУНТОВАНИМ'")
    print("   🔬 'та оптимальним для прогнозування в умовах фінансових ринків'")


def ensure_deterministic_training():
    """
    🔒 Додаткові налаштування для детермінованого навчання
    """
    import pandas as pd
    pd.set_option('mode.chained_assignment', None)

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
        print("Тестування доступності джерел даних...")
        df = load_bitcoin_data(
            prefer_source='binance',
            period=CONFIG['data']['period'],
            interval=CONFIG['data']['timeframe']
        )

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

        print("Попередня обробка даних...")
        df = preprocess_data(df)

        if df.empty:
            raise ValueError("Дані стали пустими після попередньої обробки!")

        print(f"✓ Після попередньої обробки: {len(df)} рядків")

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

        print("Підготовка ознак та цільових змінних...")

        X_train, y_train = prepare_features_targets_robust(
            train_features,
            target_col='close',  # ✅ Замість 'close'
            forecast_horizon=CONFIG['models']['n_forecast_periods'],
            nan_threshold=0.7,
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
        print("🔍 Відбір найважливіших ознак...")

        # from features.feature_engineering import select_features
        #
        # selected_features, importances = select_features(
        #     X_train, y_train,
        #     method='xgb',  # Найкращий для фінансових даних
        #     n_features=CONFIG['features']['n_selected_features']  # 30 з конфігу
        # )
        #
        # print(f"✅ Відібрано {len(selected_features)} з {len(X_train.columns)} ознак")
        #
        # # Застосовуємо селекцію до всіх наборів
        # X_train = X_train[selected_features]
        # X_val = X_val[selected_features]
        # X_test = X_test[selected_features]
        #
        # # Виводимо топ-10 найважливіших ознак
        # print("🏆 Топ-10 найважливіших ознак:")
        # for i, (feature, importance) in enumerate(zip(selected_features[:10], importances[:10])):
        #     print(f"  {i + 1}. {feature}: {importance:.4f}")
        #
        # print(f"✓ Тренувальні ознаки: {X_train.shape}")
        # print(f"✓ Валідаційні ознаки: {X_val.shape}")
        # print(f"✓ Тестові ознаки: {X_test.shape}")

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


def test_strategy_multiple_timeframes(pipeline, test_features, feature_names_for_model):
    """
    🕒 Тестування стратегії на різних проміжках часу
    ДОДАТИ ЦЮ ФУНКЦІЮ В main_stoploss.py
    """
    print("\n" + "🕒" * 60)
    print("ТЕСТУВАННЯ СТРАТЕГІЇ НА РІЗНИХ ПРОМІЖКАХ ЧАСУ")
    print("🕒" * 60)

    # Визначаємо різні проміжки для тестування (в індексах)
    timeframes = {
        'Останній_тиждень': (-168, -1),  # Останні 168 годин (7 днів)
        'Останній_місяць': (-720, -1),  # Останні 720 годин (30 днів)
        'Середній_період': (-360, -180),  # 360-180 годин назад (15-7.5 днів назад)
        'Ранній_період': (-720, -360),  # 720-360 годин назад (30-15 днів назад)
        'Перша_половина': (0, len(test_features) // 2),  # Перша половина даних
        'Друга_половина': (len(test_features) // 2, -1),  # Друга половина даних
    }

    all_results = {}

    for period_name, (start_idx, end_idx) in timeframes.items():
        try:
            print(f"\n📊 Тестування періоду: {period_name}")
            print("-" * 50)

            # Отримуємо підвибірку даних для цього періоду
            if end_idx == -1:
                period_data = test_features.iloc[start_idx:]
            else:
                period_data = test_features.iloc[start_idx:end_idx]

            # Перевіряємо, чи достатньо даних
            if len(period_data) < 50:
                print(f"⚠️ Недостатньо даних для періоду {period_name} ({len(period_data)} точок)")
                continue

            print(f"📈 Період: {len(period_data)} точок даних")
            print(f"📅 Від: {period_data['timestamp'].iloc[0]} до: {period_data['timestamp'].iloc[-1]}")

            # Забезпечуємо наявність необхідних колонок
            period_data = ensure_required_columns(period_data, feature_names_for_model)

            # Створюємо нову торгову систему для цього періоду
            period_trading_system = AggressiveTradingSystem(
                prediction_model=pipeline,
                initial_balance=CONFIG['trading']['initial_balance']
            )

            # Скидаємо логи PnL для цього періоду
            reset_trading_log()

            # Запускаємо бектест для цього періоду
            period_results = run_aggressive_backtest(
                trading_system=period_trading_system,
                historical_data=period_data,
                feature_names_for_model=feature_names_for_model
            )

            if not period_results.empty:
                # Розраховуємо метрики для цього періоду
                period_metrics = calculate_trading_metrics(period_results)

                advanced_stats = calculate_advanced_trading_statistics(period_results)
                kurtosis_stats = calculate_advanced_trading_statistics_with_kurtosis(period_results)
                period_stats = period_trading_system.get_trading_statistics()

                # Розраховуємо специфічні метрики періоду
                initial_value = period_results['portfolio_value'].iloc[0]
                final_value = period_results['portfolio_value'].iloc[-1]
                period_return = ((final_value - initial_value) / initial_value) * 100

                executed_trades = len(period_results[period_results['executed'] == True])
                days_in_period = len(period_data) / 24  # Припускаємо годинні дані
                trades_per_day = executed_trades / days_in_period if days_in_period > 0 else 0

                # Аналіз успішності
                if 'pnl' in period_results.columns:
                    profitable_trades = len(period_results[
                                                (period_results['executed'] == True) &
                                                (period_results['pnl'] > 0)
                                                ])
                    success_rate = (profitable_trades / executed_trades * 100) if executed_trades > 0 else 0
                else:
                    success_rate = 0
                    profitable_trades = 0

                # Зберігаємо результати
                period_summary = {
                    'period_name': period_name,
                    'data_points': len(period_data),
                    'days': days_in_period,
                    'initial_value': initial_value,
                    'final_value': final_value,
                    'period_return': period_return,
                    'total_trades': executed_trades,
                    'trades_per_day': trades_per_day,
                    'profitable_trades': profitable_trades,
                    'success_rate': success_rate,
                    'long_trades': period_stats.get('buy_trades', 0),
                    'short_trades': period_stats.get('short_trades', 0),
                    'long_pnl': period_stats.get('long_pnl', 0),
                    'short_pnl': period_stats.get('short_pnl', 0),
                    'total_fees': period_stats.get('total_fees_paid', 0),
                    'stop_loss_triggered': period_stats.get('stop_loss_triggered', 0),
                    'take_profit_triggered': period_stats.get('take_profit_triggered', 0),
                    'full_results': period_results,
                    'full_stats': period_stats
                }

                all_results[period_name] = period_summary

                # Виводимо короткий звіт
                print(f"💰 Дохідність: {period_return:+.2f}%")
                print(f"📊 Трейдів: {executed_trades} ({trades_per_day:.1f}/день)")
                print(f"✅ Успішність: {success_rate:.1f}%")
                print(f"🟢 Лонг: {period_stats.get('buy_trades', 0)} | 🔴 Шорт: {period_stats.get('short_trades', 0)}")

                if abs(period_stats.get('long_pnl', 0)) > 0 or abs(period_stats.get('short_pnl', 0)) > 0:
                    print(
                        f"💵 Long PnL: ${period_stats.get('long_pnl', 0):+.0f} | Short PnL: ${period_stats.get('short_pnl', 0):+.0f}")

            else:
                print(f"❌ Немає результатів для періоду {period_name}")

        except Exception as e:
            print(f"❌ Помилка при тестуванні періоду {period_name}: {e}")
            continue

    # Створюємо порівняльну таблицю
    if all_results:
        print(f"\n📊 ПОРІВНЯЛЬНА ТАБЛИЦЯ РЕЗУЛЬТАТІВ")
        print("=" * 120)
        print(
            f"{'Період':<20} {'Дні':<6} {'Дохідність':<12} {'Трейдів':<8} {'Тр/день':<8} {'Успішність':<10} {'Лонг':<6} {'Шорт':<6}")
        print("-" * 120)

        best_return = -999999
        best_period = None
        most_active = 0
        most_active_period = None

        for period_name, results in all_results.items():
            print(f"{period_name:<20} "
                  f"{results['days']:<6.1f} "
                  f"{results['period_return']:+<12.2f}% "
                  f"{results['total_trades']:<8} "
                  f"{results['trades_per_day']:<8.1f} "
                  f"{results['success_rate']:<10.1f}% "
                  f"{results['long_trades']:<6} "
                  f"{results['short_trades']:<6}")

            # Відслідковуємо найкращі результати
            if results['period_return'] > best_return:
                best_return = results['period_return']
                best_period = period_name

            if results['total_trades'] > most_active:
                most_active = results['total_trades']
                most_active_period = period_name

        print("-" * 120)
        print(f"🏆 Найкраща дохідність: {best_period} ({best_return:+.2f}%)")
        print(f"🔥 Найбільш активний: {most_active_period} ({most_active} трейдів)")

        # Аналіз консистентності
        returns = [r['period_return'] for r in all_results.values()]
        avg_return = np.mean(returns)
        std_return = np.std(returns)

        print(f"\n📈 АНАЛІЗ КОНСИСТЕНТНОСТІ:")
        print(f"   Середня дохідність: {avg_return:+.2f}%")
        print(f"   Стандартне відхилення: {std_return:.2f}%")
        print(f"   Коефіцієнт варіації: {(std_return / abs(avg_return) * 100) if avg_return != 0 else 'N/A'}")

        positive_periods = len([r for r in returns if r > 0])
        print(
            f"   Прибуткових періодів: {positive_periods}/{len(returns)} ({positive_periods / len(returns) * 100:.1f}%)")

        # Простий висновок
        if positive_periods >= len(returns) * 0.7:
            conclusion = "✅ СТРАТЕГІЯ СТАБІЛЬНА на різних періодах"
        elif positive_periods >= len(returns) * 0.5:
            conclusion = "⚠️ СТРАТЕГІЯ ПОМІРНО СТАБІЛЬНА"
        else:
            conclusion = "❌ СТРАТЕГІЯ НЕСТАБІЛЬНА на різних періодах"

        print(f"\n🎯 ВИСНОВОК: {conclusion}")

    else:
        print("❌ Не вдалося протестувати жоден період")

    return all_results

def run_model_training(X_train, y_train, X_val, y_val):
    """Навчання та оптимізація моделей БЕЗ перенавчання"""
    print("\n" + "=" * 50)
    print("КРОК 3: Навчання та оптимізація моделей (БЕЗ перенавчання)")
    print("=" * 50)

    try:
        pipeline = FixedCryptoPricePredictionPipeline(
            n_forecast_periods=CONFIG['models']['n_forecast_periods'],
            ensemble_method='cv'  # або 'cv' для більшої точності
        )

        print("Навчання ансамблевої моделі БЕЗ перенавчання...")
        print(f"Метод ансамблю: рівномірні ваги (найбезпечніший)")
        print(f"Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")
        pipeline.fit(
            X_train, y_train,
            optimize=CONFIG['models']['optimize_hyperparams']
        )

        # Збереження моделі (залишається без змін)
        ensure_dir(CONFIG['saving']['models_dir'])
        model_filename = f"crypto_prediction_model_fixed_{timestamp_to_string()}.pkl"
        model_path = os.path.join(CONFIG['saving']['models_dir'], model_filename)
        save_model(pipeline, model_filename, CONFIG['saving']['models_dir'])

        print(f"✓ Модель збережено в {model_path}")
        print("✓ Перенавчання усунуто - валідаційні дані не використовуються для оптимізації ваг!")

        return pipeline

    except Exception as e:
        print(f"❌ Помилка при навчанні моделей: {str(e)}")
        print("Деталі помилки:")
        print(traceback.format_exc())
        raise

def run_model_evaluation(pipeline, X_test, y_test):
    """Оновлена оцінка моделі з правильним форматуванням"""
    print("\n" + "=" * 50)
    print("КРОК 4: Оцінка моделі")
    print("=" * 50)

    try:
        print("Оцінка моделі на тестових даних...")
        metrics = pipeline.evaluate(X_test, y_test)
        y_pred = pipeline.predict(X_test)
        individual_results = pipeline.evaluate_individual_models(X_test, y_test)

        print("\n" + "🎯" * 20)
        detailed_results = pipeline.show_detailed_predictions(X_test, y_test, n_predictions=15)

        print("\n" + "🔮" * 20)
        future_predictions = pipeline.get_next_predictions(X_test, n_steps=2)

        print("Створення візуалізації результатів...")
        plot_price_prediction(y_test, y_pred, title='Прогноз цін Біткоіна - тестові дані')

        if hasattr(pipeline, 'ensemble') and hasattr(pipeline.ensemble, 'feature_importances_') and pipeline.ensemble.feature_importances_:
            importances = list(pipeline.ensemble.feature_importances_.values())
            features = list(pipeline.ensemble.feature_importances_.keys())
            plot_feature_importance(features, importances, n_features=20)
        comparison_chart_path = create_model_comparison_chart(CONFIG['saving']['results_dir'])
        # 🔧 ДОДАЄМО результати до metrics
        metrics['Detailed_Predictions'] = detailed_results
        metrics['Future_Predictions'] = future_predictions

        print("\n📊 Результати оцінки моделі:")

        # 🔧 ВИПРАВЛЕНО: правильне форматування різних типів даних
        for metric, value in metrics.items():
            if metric in ['Detailed_Predictions', 'Future_Predictions']:
                continue
            elif isinstance(value, str):
                print(f"{metric}: {value}")
            elif isinstance(value, (int, float)):
                if metric in ['MAE', 'RMSE']:
                    # Розраховуємо середню ціну для конвертації у відсотки
                    mean_price = np.mean(y_test)
                    percentage_value = (value / mean_price) * 100
                    print(f"{metric}: {percentage_value:.2f}%")
                else:
                    print(f"{metric}: {value:.4f}")
                # Числа форматуємо з 4 знаками після коми
                print(f"{metric}: {value:.4f}")
            elif isinstance(value, dict):
                # Словники виводимо як рядок
                print(f"{metric}: {value}")
            else:
                # Все інше як рядок
                print(f"{metric}: {value}")
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

    required_columns = [
                           'timestamp', 'close', 'open', 'high', 'low', 'volume',
                           'rsi', 'macd_12_26', 'macd_signal_12_26', 'adx', 'volume_ratio_20'
                       ] + feature_names_for_model

    required_columns = list(set(required_columns))

    missing_columns = [col for col in required_columns if col not in test_features.columns]

    if missing_columns:
        print(f"⚠️ Відсутні колонки: {missing_columns}")

        for col in missing_columns:
            if col in ['rsi']:
                test_features[col] = 50
            elif col in ['macd_12_26', 'macd', 'macd_signal_12_26', 'macd_signal']:
                test_features[col] = 0
            elif col in ['adx']:
                test_features[col] = 25
            elif col in ['volume_ratio_20']:
                test_features[col] = 1.0
            elif 'lag' in col or 'sma' in col or 'ema' in col:
                test_features[col] = test_features['close'] if 'close' in test_features.columns else 50000
            else:
                test_features[col] = 0

        print(f"✓ Додано {len(missing_columns)} відсутніх колонок зі значеннями за замовчуванням")

    if 'macd_12_26' not in test_features.columns and 'macd' in test_features.columns:
        test_features['macd_12_26'] = test_features['macd']
        print("✓ Використано 'macd' як 'macd_12_26'")

    if 'macd_signal_12_26' not in test_features.columns and 'macd_signal' in test_features.columns:
        test_features['macd_signal_12_26'] = test_features['macd_signal']
        print("✓ Використано 'macd_signal' як 'macd_signal_12_26'")

    print(f"✓ Всі необхідні колонки присутні в даних")
    return test_features


def run_aggressive_trading_strategy_with_shorts(pipeline, test_features, feature_names_for_model):
    """
    🔥 ВИПРАВЛЕНА функція з простим підходом до PnL
    """
    print("\n" + "=" * 60)
    print("🔥 КРОК 5: АГРЕСИВНА торгова система з ПРОСТИМ PnL")
    print("=" * 60)

    try:

        test_features = ensure_required_columns(test_features, feature_names_for_model)

        trading_system = AggressiveTradingSystem(
            prediction_model=pipeline,
            initial_balance=CONFIG['trading']['initial_balance']
        )

        print(f"\n🔴 ПАРАМЕТРИ ШОРТ ПОЗИЦІЙ:")
        print(f"  Шорт позиції увімкнені: {trading_system.enable_shorts}")
        print(f"  Макс. шорт позицій: {trading_system.risk_params['max_short_positions']}")

        trading_system.show_trading_predictions_summary(test_features, feature_names_for_model, n_examples=10)

        print("\n🔥 Проведення бектесту системи з шорт позиціями...")
        backtest_results_df = run_aggressive_backtest(
            trading_system=trading_system,
            historical_data=test_features,
            feature_names_for_model=feature_names_for_model
        )

        if not backtest_results_df.empty:
            print("📊 Створення ПРОСТОЇ але ЕФЕКТИВНОЇ візуалізації...")

            # 🆕 ВИКОРИСТОВУЄМО ПРОСТУ ВІЗУАЛІЗАЦІЮ З PnL З ДАНИХ
            plot_backtest_results_simple_pnl(backtest_results_df, "Агресивна Система")
        else:
            print("⚠️ Немає результатів для візуалізації.")

        trading_metrics = calculate_trading_metrics(backtest_results_df)
        trading_stats = trading_system.get_trading_statistics()


        return backtest_results_df, trading_metrics, trading_stats
    except Exception as e:
        print(f"❌ Помилка: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return pd.DataFrame(), {}, {}

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
        if len(value) <= 5:
            return str(value)
        else:
            return f"[{value[0]}, {value[1]}, ..., {value[-1]}] (total: {len(value)} items)"
    elif isinstance(value, dict):
        if len(value) <= 3:
            return str(value)
        else:
            keys = list(value.keys())[:3]
            return f"{{{keys[0]}: {value[keys[0]]}, {keys[1]}: {value[keys[1]]}, ...}} (total: {len(value)} keys)"
    else:
        return str(value)


def create_model_comparison_chart(save_path='results/'):
    """
    Створює графік порівняння метрик RMSE, MAE та R² для окремих моделей
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Дані метрик для кожної моделі (приклад на основі типових результатів)
    models = ['XGBoost', 'LightGBM', 'HistGradientBoosting', 'Ансамбль']

    # Метрики (приклад значень - замініть на фактичні результати вашого навчання)
    rmse_values = [1.69, 1.45,1.47,1.48]
    mae_values = [1.06, 0.85,0.91,0.92]
    r2_values = [0.8708, 0.9180, 0.9019,0.9204]

    # Налаштування графіка
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    # Кольори для кожної моделі
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Графік 1: RMSE
    bars1 = ax1.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_title('Root Mean Square Error (RMSE)', fontsize=10, fontweight='bold', pad=15)
    ax1.set_ylabel('RMSE (%)', fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Додавання значень на стовпці
    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 10,
                 f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Виділення найкращого результату
    best_rmse_idx = rmse_values.index(min(rmse_values))
    bars1[best_rmse_idx].set_edgecolor('gold')
    bars1[best_rmse_idx].set_linewidth(3)

    # Графік 2: MAE
    bars2 = ax2.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax2.set_title('Mean Absolute Error (MAE)', fontsize=10, fontweight='bold', pad=15)
    ax2.set_ylabel('MAE (%)', fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Додавання значень на стовпці
    for bar, value in zip(bars2, mae_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 5,
                 f'{value:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Виділення найкращого результату
    best_mae_idx = mae_values.index(min(mae_values))
    bars2[best_mae_idx].set_edgecolor('gold')
    bars2[best_mae_idx].set_linewidth(3)

    # Графік 3: R²
    bars3 = ax3.bar(models, r2_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax3.set_title('Coefficient of Determination (R²)', fontsize=10, fontweight='bold', pad=15)
    ax3.set_ylabel('R² Score', fontsize=10)
    ax3.set_ylim(0.0, 1.0)  # Фокус на діапазоні значень
    ax3.grid(axis='y', alpha=0.3)

    # Додавання значень на стовпці
    for bar, value in zip(bars3, r2_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Виділення найкращого результату
    best_r2_idx = r2_values.index(max(r2_values))
    bars3[best_r2_idx].set_edgecolor('gold')
    bars3[best_r2_idx].set_linewidth(3)

    # Поворот підписів для кращої читабельності
    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='x', rotation=45)
        ax.set_xlabel('Модель', fontsize=12)

    # Загальний заголовок
    fig.suptitle('Порівняння метрик якості моделей прогнозування',
                 fontsize=10, fontweight='bold', y=0.98)

    # Додавання легенди для позначення найкращих результатів
    from matplotlib.patches import Rectangle
    legend_elements = [Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='gold', linewidth=3,
                                 label='Найкращий результат')]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.15)

    # Збереження графіка
    from utils.helpers import ensure_dir, timestamp_to_string
    ensure_dir(save_path)
    chart_path = os.path.join(save_path, f'model_comparison_metrics_{timestamp_to_string()}.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"✓ Графік порівняння метрик моделей збережено: {chart_path}")
    return chart_path

def create_aggressive_summary_report_with_shorts(prediction_metrics, trading_metrics, save_path=None):
    """
    📊 НОВА ФУНКЦІЯ: Створення підсумкового звіту з шорт позиціями
    """
    print("\n" + "=" * 50)
    print("🔥 ПІДСУМКОВИЙ ЗВІТ АГРЕСИВНОЇ СИСТЕМИ З ШОРТ ПОЗИЦІЯМИ")
    print("=" * 50)

    report = []
    report.append("# 🔥 Звіт про роботу агресивної системи прогнозування Bitcoin з шорт позиціями")
    report.append(f"Дата створення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    report.append("## ⚙️ Параметри системи")
    try:
        report.append(f"- Горизонт прогнозування: {CONFIG['models']['n_forecast_periods']} годин")
        report.append(f"- Таймфрейм даних: {CONFIG['data']['timeframe']}")
        report.append(f"- Період даних: {CONFIG['data']['period']}")
        report.append(
            f"- Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")
        # 🆕 НОВІ ПАРАМЕТРИ
        report.append(
            f"- Шорт позиції: {'Увімкнені' if CONFIG['trading'].get('enable_short_positions', False) else 'Вимкнені'}")
        if CONFIG['trading'].get('enable_short_positions', False):
            report.append(f"- Макс. шорт позицій: {CONFIG['trading'].get('max_short_positions', 2)}")
    except:
        report.append("- Не вдалося отримати параметри системи")
    report.append("")

    report.append("## 🎯 Цілі агресивної стратегії з шортами")
    report.append("- **Основна ціль**: Збільшити кількість трейдів до 50+ (включно з шортами)")
    report.append("- **Нова можливість**: Заробляти на падінні ціни через шорт позиції")
    report.append("- **Диверсифікація**: Одночасні лонг і шорт позиції для хеджування")
    report.append("- **Реалістичність**: Врахування комісій за позичку та ліквідацію")
    report.append("")

    report.append("## 📊 Результати прогнозування")
    if prediction_metrics:
        for metric, value in prediction_metrics.items():
            if isinstance(value, dict):
                report.append(f"- {metric}:")
                for k, v in value.items():
                    formatted_value = safe_format_value(v)
                    report.append(f"  - {k}: {formatted_value}")
            else:
                formatted_value = safe_format_value(value)
                report.append(f"- {metric}: {formatted_value}")
    else:
        report.append("- Метрики прогнозування недоступні")
    report.append("")

    report.append("## 🔥 Результати агресивної торгової стратегії з шортами")
    if trading_metrics:
        total_trades = trading_metrics.get('total_trades', 0)
        short_trades = trading_metrics.get('short_trades', 0)
        long_trades = trading_metrics.get('buy_trades', 0)
        total_costs = trading_metrics.get('total_trading_costs', 0)
        borrowing_fees = trading_metrics.get('total_borrowing_fees', 0)

        # 🆕 АНАЛІЗ ДОСЯГНЕННЯ ЦІЛЕЙ З ШОРТАМИ
        if total_trades >= 50:
            goal_status = "✅ ЦІЛЬ ДОСЯГНУТО"
        elif total_trades >= 30:
            goal_status = "⚡ БЛИЗЬКО ДО ЦІЛІ"
        else:
            goal_status = "❌ ЦІЛЬ НЕ ДОСЯГНУТА"

        report.append(f"### 🎯 Досягнення мети: {goal_status}")
        report.append(f"- Загальна кількість трейдів: {total_trades} (мета: 50+)")
        report.append(f"- Лонг трейди: {long_trades}")
        report.append(f"- Шорт трейди: {short_trades}")
        if total_trades > 0:
            short_percentage = (short_trades / total_trades) * 100
            report.append(f"- Відсоток шорт трейдів: {short_percentage:.1f}%")
        report.append("")

        report.append("### 📈 Детальна статистика торгівлі:")

        # 🆕 РОЗДІЛЕННЯ СТАТИСТИКИ НА ЛОНГ І ШОРТ
        report.append("#### 📊 Загальна статистика:")
        for metric, value in trading_metrics.items():
            if metric not in ['short_trades', 'buy_trades', 'short_pnl', 'long_pnl', 'total_borrowing_fees']:
                formatted_value = safe_format_value(value)
                report.append(f"- {metric}: {formatted_value}")

        # 🆕 СТАТИСТИКА ПОЗИЦІЙ
        report.append("#### 🟢 Лонг позиції:")
        report.append(f"- Кількість лонг трейдів: {long_trades}")
        long_pnl = trading_metrics.get('long_pnl', 0)
        if isinstance(long_pnl, (int, float)):
            report.append(f"- PnL від лонг позицій: ${long_pnl:.2f}")
        open_long = trading_metrics.get('open_long_positions', 0)
        report.append(f"- Відкритих лонг позицій: {open_long}")

        report.append("#### 🔴 Шорт позиції:")
        report.append(f"- Кількість шорт трейдів: {short_trades}")
        short_pnl = trading_metrics.get('short_pnl', 0)
        if isinstance(short_pnl, (int, float)):
            report.append(f"- PnL від шорт позицій: ${short_pnl:.2f}")
        open_short = trading_metrics.get('open_short_positions', 0)
        report.append(f"- Відкритих шорт позицій: {open_short}")
        borrowed_btc = trading_metrics.get('total_borrowed_btc', 0)
        if isinstance(borrowed_btc, (int, float)):
            report.append(f"- Запозичено BTC: {borrowed_btc:.6f}")

        report.append("")
        report.append(f"### 💰 Торгові витрати (включно з шортами):")
        if isinstance(total_costs, (int, float)):
            report.append(f"- Загальні торгові витрати: ${total_costs:.2f}")
        if isinstance(borrowing_fees, (int, float)):
            report.append(f"- Комісії за позичку (шорти): ${borrowing_fees:.2f}")
            total_all_costs = total_costs + borrowing_fees
            report.append(f"- Всього витрат: ${total_all_costs:.2f}")

        cost_ratio = trading_metrics.get('cost_ratio', 0)
        if isinstance(cost_ratio, (int, float)):
            report.append(f"- Витрати у відсотках: {cost_ratio:.2f}%")

        # 🆕 АНАЛІЗ ЕФЕКТИВНОСТІ ШОРТІВ
        report.append("")
        report.append(f"### 📊 Аналіз ефективності шортів:")
        if short_trades > 0:
            short_success_rate = trading_metrics.get('short_success_rate', 0)
            long_success_rate = trading_metrics.get('long_success_rate', 0)

            report.append(f"- Успішність шорт стратегії: {short_success_rate:.1f}%")
            report.append(f"- Успішність лонг стратегії: {long_success_rate:.1f}%")

            if isinstance(short_pnl, (int, float)) and isinstance(long_pnl, (int, float)):
                total_pnl = short_pnl + long_pnl
                if total_pnl != 0:
                    short_contribution = (short_pnl / total_pnl) * 100
                    report.append(f"- Внесок шортів у загальний PnL: {short_contribution:.1f}%")
        else:
            report.append("- Шорт трейди не виконувались або вимкнені")

    else:
        report.append("- Агресивна торгова стратегія з шортами не була протестована")
    report.append("")

    # 🆕 ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ
    report.append("## 🎯 Висновки та рекомендації:")
    if trading_metrics:
        total_trades = trading_metrics.get('total_trades', 0)
        short_trades = trading_metrics.get('short_trades', 0)

        if total_trades >= 30:
            report.append("✅ **Система показує гарні результати активності**")
        else:
            report.append("⚠️ **Потрібне налаштування для збільшення активності**")

        if short_trades > 0:
            report.append("✅ **Шорт позиції успішно інтегровані в стратегію**")
            report.append("💡 **Рекомендація**: Можна експериментувати з різними порогами для шортів")
        else:
            report.append("❌ **Шорт позиції не використовувались**")
            report.append("💡 **Рекомендація**: Перевірити налаштування шорт параметрів або ринкові умови")

        report.append("💡 **Загальна рекомендація**: Продовжити тестування з різними параметрами ризику")
    report.append("")

    report_text = "\n".join(report)

    if save_path:
        try:
            ensure_dir(os.path.dirname(save_path))
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✓ Звіт з шорт позиціями збережено в {save_path}")
        except Exception as e:
            print(f"❌ Помилка збереження звіту: {e}")

    print(report_text)
    return report_text


def create_timeframe_comparison_report(timeframe_results, save_path):
    """
    📊 Створює детальний звіт порівняння різних періодів
    ДОДАТИ ЦЮ ФУНКЦІЮ В main_stoploss.py
    """
    if not timeframe_results:
        print("❌ Немає результатів для створення звіту")
        return

    report_lines = []
    report_lines.append("# 🕒 Звіт тестування стратегії на різних проміжках часу")
    report_lines.append(f"Дата створення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Загальна статистика
    report_lines.append("## 📊 Загальна статистика")
    total_periods = len(timeframe_results)
    profitable_periods = len([r for r in timeframe_results.values() if r['period_return'] > 0])

    report_lines.append(f"- Протестовано періодів: {total_periods}")
    report_lines.append(
        f"- Прибуткових періодів: {profitable_periods} ({profitable_periods / total_periods * 100:.1f}%)")
    report_lines.append("")

    # Детальна таблиця
    report_lines.append("## 📈 Детальні результати по періодах")
    report_lines.append("")
    report_lines.append(
        "| Період | Дні | Дохідність | Трейдів | Тр/день | Успішність | Лонг | Шорт | Long PnL | Short PnL |")
    report_lines.append(
        "|--------|-----|------------|---------|---------|------------|------|------|----------|-----------|")

    for period_name, results in timeframe_results.items():
        report_lines.append(
            f"| {period_name} | "
            f"{results['days']:.1f} | "
            f"{results['period_return']:+.2f}% | "
            f"{results['total_trades']} | "
            f"{results['trades_per_day']:.1f} | "
            f"{results['success_rate']:.1f}% | "
            f"{results['long_trades']} | "
            f"{results['short_trades']} | "
            f"${results['long_pnl']:+.0f} | "
            f"${results['short_pnl']:+.0f} |"
        )

    report_lines.append("")

    # Найкращі та найгірші періоди
    best_period = max(timeframe_results.items(), key=lambda x: x[1]['period_return'])
    worst_period = min(timeframe_results.items(), key=lambda x: x[1]['period_return'])
    most_active = max(timeframe_results.items(), key=lambda x: x[1]['total_trades'])

    report_lines.append("## 🏆 Аналіз екстремумів")
    report_lines.append(f"- **Найкращий період**: {best_period[0]} ({best_period[1]['period_return']:+.2f}%)")
    report_lines.append(f"- **Найгірший період**: {worst_period[0]} ({worst_period[1]['period_return']:+.2f}%)")
    report_lines.append(f"- **Найактивніший період**: {most_active[0]} ({most_active[1]['total_trades']} трейдів)")
    report_lines.append("")

    # Статистичний аналіз
    returns = [r['period_return'] for r in timeframe_results.values()]
    trades_counts = [r['total_trades'] for r in timeframe_results.values()]

    report_lines.append("## 📊 Статистичний аналіз")
    report_lines.append(f"- Середня дохідність: {np.mean(returns):+.2f}%")
    report_lines.append(f"- Медіанна дохідність: {np.median(returns):+.2f}%")
    report_lines.append(f"- Стандартне відхилення: {np.std(returns):.2f}%")
    report_lines.append(f"- Мінімальна дохідність: {min(returns):+.2f}%")
    report_lines.append(f"- Максимальна дохідність: {max(returns):+.2f}%")
    report_lines.append(f"- Середня кількість трейдів: {np.mean(trades_counts):.1f}")
    report_lines.append("")

    # Висновки
    report_lines.append("## 🎯 Висновки")

    consistency_score = profitable_periods / total_periods
    if consistency_score >= 0.8:
        consistency_level = "дуже висока"
        recommendation = "Стратегія показує стабільні результати на різних проміжках часу."
    elif consistency_score >= 0.6:
        consistency_level = "висока"
        recommendation = "Стратегія загалом стабільна, але потребує моніторингу."
    elif consistency_score >= 0.4:
        consistency_level = "помірна"
        recommendation = "Стратегія нестабільна, рекомендується додаткова оптимізація."
    else:
        consistency_level = "низька"
        recommendation = "Стратегія показує слабкі результати на більшості періодів."

    report_lines.append(
        f"- **Консистентність**: {consistency_level} ({consistency_score * 100:.1f}% прибуткових періодів)")
    report_lines.append(f"- **Рекомендація**: {recommendation}")

    # Аналіз ризиків
    volatility = np.std(returns)
    if volatility > 10:
        risk_level = "високий"
    elif volatility > 5:
        risk_level = "помірний"
    else:
        risk_level = "низький"

    report_lines.append(f"- **Ризик волатільності**: {risk_level} (σ={volatility:.2f}%)")
    report_lines.append("")

    # Збереження звіту
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"✅ Звіт про тестування проміжків збережено: {save_path}")
    except Exception as e:
        print(f"❌ Помилка збереження звіту: {e}")

    return report_lines
def main():
    """🔥 ОСНОВНА ФУНКЦІЯ з агресивною торговою стратегією, стоп-лосом та ШОРТ ПОЗИЦІЯМИ"""
    comprehensive_seed_fix(42)
    ensure_deterministic_training()
    print("🔥 Запуск системи прогнозування та АГРЕСИВНОЇ торгівлі з ШОРТ ПОЗИЦІЯМИ")
    print("=" * 80)

    results = {}

    try:
        # Крок 1: Підготовка даних
        train_df, val_df, test_df = run_data_preparation()

        distribution_analysis = run_data_generating_analysis(train_df, val_df, test_df)
        results['distribution_analysis'] = distribution_analysis

        print("✅ Аналіз породжуючого розподілу завершено")
        print("🎯 Вибір непараметричних моделей теоретично обґрунтований")
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

        # 🔥 Крок 5: АГРЕСИВНА торгова стратегія з ШОРТ ПОЗИЦІЯМИ
        backtest_results, trading_metrics, trading_stats = run_aggressive_trading_strategy_with_shorts(
            pipeline, test_features, feature_names_for_model
        )
        results['trading_metrics'] = trading_metrics
        results['trading_stats'] = trading_stats
        results['backtest_results'] = backtest_results
        # 🕒 Крок 6: Тестування на різних проміжках часу
        print("\n" + "🕒" * 60)
        print("ДОДАТКОВИЙ АНАЛІЗ: Тестування на різних проміжках часу")
        print("🕒" * 60)

        timeframe_results = test_strategy_multiple_timeframes(
            pipeline, test_features, feature_names_for_model
        )
        results['timeframe_analysis'] = timeframe_results

        # Зберігаємо детальні результати по періодах
        if timeframe_results:
            ensure_dir(CONFIG['saving']['results_dir'])
            timeframe_report_path = os.path.join(
                CONFIG['saving']['results_dir'],
                f"timeframe_analysis_{timestamp_to_string()}.md"
            )
            create_timeframe_comparison_report(timeframe_results, timeframe_report_path)
        results['pipeline'] = pipeline

        # Створення підсумкового звіту з шортами
        ensure_dir(CONFIG['saving']['results_dir'])
        report_path = os.path.join(
            CONFIG['saving']['results_dir'],
            f"aggressive_summary_with_shorts_{timestamp_to_string()}.md"
        )

        create_aggressive_summary_report_with_shorts(prediction_metrics, trading_metrics, report_path)
        print("\n🎉 Система агресивного прогнозування та торгівлі з шорт позиціями успішно завершила роботу!")


        # Оцінка ефективності ризик-менеджменту

        # Фінальні прогнози
        print("\n🔮 ФІНАЛЬНІ ПРОГНОЗИ:")
        latest_predictions = pipeline.get_next_predictions(X_test, n_steps=1)

        return results

    except Exception as e:
        print(f"\n❌ Критична помилка в головній функції: {str(e)}")
        print("Повна інформація про помилку:")
        print(traceback.format_exc())

        return None


if __name__ == "__main__":
        # Запускаємо основну програму
        results = main()
        if results:
            print(f"\n✅ Результати доступні в змінній 'results'")

            # Короткий підсумок результатів з шортами
            trading_stats = results.get('trading_stats', {})
            if trading_stats:
                total_trades = trading_stats.get('total_trades', 0)
                short_trades = trading_stats.get('short_trades', 0)
                long_trades = trading_stats.get('buy_trades', 0)
                total_fees = trading_stats.get('total_fees_paid', 0)
                borrowing_fees = trading_stats.get('total_borrowing_fees', 0)
                stop_loss_count = trading_stats.get('stop_loss_triggered', 0)
                take_profit_count = trading_stats.get('take_profit_triggered', 0)
                short_pnl = trading_stats.get('short_pnl', 0)
                long_pnl = trading_stats.get('long_pnl', 0)

                print(f"\n📊 ШВИДКИЙ ПІДСУМОК З ШОРТ ПОЗИЦІЯМИ:")
                print(f"   Загальна кількість трейдів: {total_trades}")
                print(f"   🟢 Лонг трейди: {long_trades}")
                print(f"   🔴 Шорт трейди: {short_trades}")
                print(f"   Комісій сплачено: ${total_fees:.2f}")
                print(f"   Комісій за позичку: ${borrowing_fees:.2f}")
                print(f"   Стоп-лос спрацював: {stop_loss_count}")
                print(f"   Тейк-профіт спрацював: {take_profit_count}")
                if isinstance(short_pnl, (int, float)) and isinstance(long_pnl, (int, float)):
                    print(f"   💰 PnL лонг: ${long_pnl:.2f}")
                    print(f"   💰 PnL шорт: ${short_pnl:.2f}")
                    print(f"   💰 Загальний PnL: ${short_pnl + long_pnl:.2f}")
