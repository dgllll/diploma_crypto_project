# main_with_training_analysis.py - Головний файл з аналізом навчання

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import traceback

# Ваші імпорти
from config import CONFIG
from data.data_loader import load_bitcoin_data, validate_data, test_data_loading
from data.data_processor import preprocess_data, split_data
from features.feature_engineering import build_advanced_features, prepare_features_targets_robust
from models.enhanced_prediction_pipeline import EnhancedCryptoPricePredictionPipeline  # 🆕 Новий pipeline
from utils.helpers import save_model, timestamp_to_string, ensure_dir


def fix_random_seeds(seed=42):
    """Фіксація випадковості"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔧 Випадковість зафіксована з seed={seed}")


def run_data_preparation():
    """Підготовка даних (ваш існуючий код)"""
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
    """Створення ознак (ваш існуючий код)"""
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


def run_enhanced_model_training(X_train, y_train, X_val, y_val, track_training=True):
    """
    🆕 ПОКРАЩЕНЕ навчання моделей з відстеженням train/val curves
    """
    print("\n" + "=" * 50)
    print("🔍 КРОК 3: НАВЧАННЯ З АНАЛІЗОМ ПЕРЕОБУЧЕННЯ")
    print("=" * 50)

    try:
        # Створюємо покращений pipeline
        pipeline = EnhancedCryptoPricePredictionPipeline(
            n_forecast_periods=CONFIG['models']['n_forecast_periods']
        )

        # Створюємо папку для результатів аналізу
        analysis_dir = os.path.join(CONFIG['saving']['results_dir'], 'training_analysis')
        ensure_dir(analysis_dir)

        print(f"📊 Режими навчання:")
        print(f"   Відстеження train/val curves: {'✅' if track_training else '❌'}")
        print(f"   Оптимізація гіперпараметрів: {'✅' if CONFIG['models']['optimize_hyperparams'] else '❌'}")
        print(f"   Папка для аналізу: {analysis_dir}")

        # 🔥 ГОЛОВНЕ: Навчання з відстеженням
        pipeline.fit(
            X_train, y_train, X_val, y_val,
            optimize=CONFIG['models']['optimize_hyperparams'],
            track_training=track_training,  # 🆕 Ключовий параметр
            save_plots=True,
            results_dir=analysis_dir
        )

        # Зберігаємо модель
        ensure_dir(CONFIG['saving']['models_dir'])
        model_filename = f"enhanced_crypto_model_{timestamp_to_string()}.pkl"
        save_model(pipeline, model_filename, CONFIG['saving']['models_dir'])
        print(f"✓ Модель збережено: {model_filename}")

        # 🆕 Аналіз навчання (якщо відстеження увімкнено)
        if track_training:
            print(f"\n🔍 АНАЛІЗ ПРОЦЕСУ НАВЧАННЯ:")
            print("=" * 40)

            # Виводимо детальний аналіз переобучення
            pipeline.print_overfitting_summary()

            # Створюємо порівняльні графіки
            print(f"\n📊 Створення порівняльних графіків...")
            pipeline.plot_ensemble_training_comparison()

            # Зберігаємо повний аналіз
            saved_files = pipeline.save_training_analysis(analysis_dir)

            return pipeline, saved_files
        else:
            print(f"\n⚠️ Детальний аналіз недоступний без відстеження навчання")
            return pipeline, None

    except Exception as e:
        print(f"❌ Помилка навчання: {e}")
        print(traceback.format_exc())
        raise


def run_enhanced_model_evaluation(pipeline, X_test, y_test):
    """
    🆕 РОЗШИРЕНА оцінка моделі з урахуванням аналізу навчання
    """
    print("\n" + "=" * 50)
    print("📊 КРОК 4: РОЗШИРЕНА ОЦІНКА МОДЕЛІ")
    print("=" * 50)

    try:
        # Стандартна оцінка + аналіз навчання
        metrics = pipeline.evaluate(X_test, y_test)

        # Отримання прогнозів
        y_pred = pipeline.predict(X_test)

        print(f"\n📈 ОСНОВНІ МЕТРИКИ:")
        print(f"   MAE: {metrics['MAE']:.2f}")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   R²: {metrics['R2']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")

        # Додатковий аналіз якщо доступний
        if 'training_analysis' in metrics:
            training_analysis = metrics['training_analysis']
            if training_analysis['status'] == 'completed':
                print(f"\n🎯 ВИСНОВКИ ПРО ПЕРЕОБУЧЕННЯ:")

                for model_name, analysis in training_analysis['overfitting_analysis'].items():
                    status = analysis['status']
                    if 'overfitting' in status:
                        print(f"   ⚠️  {model_name}: {analysis['message']}")
                    elif status == 'good_fit':
                        print(f"   ✅ {model_name}: {analysis['message']}")
                    else:
                        print(f"   🔵 {model_name}: {analysis['message']}")

        return y_pred, metrics

    except Exception as e:
        print(f"❌ Помилка оцінки: {e}")
        raise


def create_training_analysis_report(pipeline, metrics, saved_files, save_path=None):
    """
    📋 Створює підсумковий звіт з аналізом навчання
    """
    print("\n" + "=" * 50)
    print("📋 СТВОРЕННЯ ЗВІТУ ПРО НАВЧАННЯ")
    print("=" * 50)

    report = []
    report.append("# 🔍 Звіт про аналіз навчання моделей Bitcoin")
    report.append(f"Дата створення: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Конфігурація
    report.append("## ⚙️ Параметри навчання")
    report.append(f"- Горизонт прогнозування: {CONFIG['models']['n_forecast_periods']} годин")
    report.append(
        f"- Оптимізація гіперпараметрів: {'Увімкнена' if CONFIG['models']['optimize_hyperparams'] else 'Вимкнена'}")
    report.append(f"- Відстеження train/val curves: Увімкнене")
    report.append("")

    # Результати тестування
    report.append("## 📊 Результати тестування")
    for metric, value in metrics.items():
        if metric != 'training_analysis' and isinstance(value, (int, float)):
            report.append(f"- {metric}: {value:.4f}")
    report.append("")

    # Аналіз переобучення
    if hasattr(pipeline, 'training_completed') and pipeline.training_completed:
        report.append("## 🔍 Аналіз переобучення")

        for model_name, analysis in pipeline.overfitting_analyses.items():
            report.append(f"### {model_name.upper()}")
            report.append(f"- **Статус**: {analysis['message']}")
            report.append(f"- **Фінальний Val RMSE**: {analysis['final_val_loss']:.2f}")
            report.append(f"- **Val R²**: {analysis['val_r2']:.4f}")
            report.append(f"- **Loss Gap**: {analysis['loss_gap']:+.2f}")

            # Рекомендації на основі статусу
            if analysis['status'] == 'severe_overfitting':
                report.append("- **Рекомендації**: Збільшити регуляризацію, зменшити learning rate")
            elif analysis['status'] == 'moderate_overfitting':
                report.append("- **Рекомендації**: Додати раннє зупинення, перевірити дані")
            elif analysis['status'] == 'underfitting':
                report.append("- **Рекомендації**: Збільшити складність моделі, зменшити регуляризацію")
            elif analysis['status'] == 'good_fit':
                report.append("- **Статус**: ✅ Модель добре генералізує!")
            report.append("")

        # Загальні висновки
        overfitting_count = sum(1 for analysis in pipeline.overfitting_analyses.values()
                                if 'overfitting' in analysis['status'])

        report.append("## 🎯 Загальні висновки")
        if overfitting_count == 0:
            report.append("✅ **Всі моделі показують хорошу генералізацію!**")
        elif overfitting_count == 1:
            report.append("⚠️ **Одна модель показує ознаки переобучення**")
            report.append("💡 Рекомендується налаштування її параметрів")
        else:
            report.append(f"🔴 **{overfitting_count} моделей показують переобучення**")
            report.append("💡 Необхідне ретельне налаштування ансамблю")
        report.append("")

    # Файли аналізу
    if saved_files:
        report.append("## 📁 Збережені файли аналізу")
        for file_type, file_path in saved_files.items():
            report.append(f"- {file_type}: `{file_path}`")
        report.append("")

    report_text = "\n".join(report)

    if save_path:
        ensure_dir(os.path.dirname(save_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"✓ Звіт збережено: {save_path}")

    print(report_text)
    return report_text


def main_with_training_analysis():
    """
    🔍 ГОЛОВНА ФУНКЦІЯ з повним аналізом навчання та детекцією переобучення
    """
    print("🔍 ЗАПУСК СИСТЕМИ З АНАЛІЗОМ НАВЧАННЯ МОДЕЛЕЙ")
    print("🎯 Мета: Виявити та проаналізувати переобучення у gradient boosting моделях")
    print("=" * 80)

    # Фіксація випадковості
    fix_random_seeds(42)

    results = {}

    try:
        # Крок 1: Підготовка даних
        train_df, val_df, test_df = run_data_preparation()

        # Крок 2: Створення ознак
        X_train, y_train, X_val, y_val, X_test, y_test, test_features = run_feature_engineering(
            train_df, val_df, test_df
        )

        # 🔍 Крок 3: НАВЧАННЯ З АНАЛІЗОМ (ключовий крок)
        print(f"\n🔥 УВАГА: Буде створено детальні графіки train/validation curves")
        print(f"📊 Це допоможе виявити переобучення у кожній моделі")

        # Вибір режиму (можна змінити на False для швидкого навчання)
        ENABLE_TRAINING_TRACKING = True  # 🎛️ Головний перемикач

        pipeline, analysis_files = run_enhanced_model_training(
            X_train, y_train, X_val, y_val
        )

        results['pipeline'] = pipeline
        results['analysis_files'] = analysis_files

        # Крок 4: Розширена оцінка
        y_pred, prediction_metrics = run_enhanced_model_evaluation(pipeline, X_test, y_test)
        results['prediction_metrics'] = prediction_metrics
        results['y_pred'] = y_pred

        # Створення підсумкового звіту
        ensure_dir(CONFIG['saving']['results_dir'])
        report_path = os.path.join(
            CONFIG['saving']['results_dir'],
            f"training_analysis_report_{timestamp_to_string()}.md"
        )

        create_training_analysis_report(pipeline, prediction_metrics, analysis_files, report_path)
        results['report_path'] = report_path

        # 🎉 Фінальний підсумок
        print(f"\n🎉 АНАЛІЗ НАВЧАННЯ ЗАВЕРШЕНО УСПІШНО!")
        print("=" * 50)

        if ENABLE_TRAINING_TRACKING and hasattr(pipeline, 'training_completed') and pipeline.training_completed:
            print(f"✅ Створено детальні графіки train/validation curves")
            print(f"✅ Проведено аналіз переобучення для всіх моделей")
            print(f"✅ Збережено повний звіт та графіки")

            # Швидкий підсумок по моделях
            print(f"\n📊 ШВИДКИЙ ПІДСУМОК ПО МОДЕЛЯХ:")
            for model_name, analysis in pipeline.overfitting_analyses.items():
                status_emoji = {
                    'severe_overfitting': '🔴',
                    'moderate_overfitting': '🟡',
                    'mild_overfitting': '🟠',
                    'good_fit': '✅',
                    'underfitting': '🔵'
                }.get(analysis['status'], '❓')

                print(
                    f"   {status_emoji} {model_name}: Val RMSE {analysis['final_val_loss']:.2f}, R² {analysis['val_r2']:.3f}")

            # Головний висновок
            overfitting_models = [name for name, analysis in pipeline.overfitting_analyses.items()
                                  if 'overfitting' in analysis['status']]

            if len(overfitting_models) == 0:
                print(f"\n🏆 ВІДМІННО: Всі моделі показують хорошу генералізацію!")
            elif len(overfitting_models) == 1:
                print(f"\n⚠️  УВАГА: Модель {overfitting_models[0]} показує переобучення")
                print(f"💡 Рекомендація: Налаштуйте параметри регуляризації")
            else:
                print(f"\n🔴 ПРОБЛЕМА: {len(overfitting_models)} моделей показують переобучення")
                print(f"💡 Моделі: {', '.join(overfitting_models)}")
                print(f"💡 Рекомендація: Переглянути стратегію навчання")

        else:
            print(f"⚠️  Детальний аналіз навчання не проводився")
            print(f"💡 Для увімкнення встановіть ENABLE_TRAINING_TRACKING = True")

        print(f"\n📁 ЗБЕРЕЖЕНІ ФАЙЛИ:")
        if analysis_files:
            for file_type, file_path in analysis_files.items():
                print(f"   📊 {file_type}: {file_path}")
        print(f"   📋 Звіт: {report_path}")

        return results

    except Exception as e:
        print(f"\n❌ КРИТИЧНА ПОМИЛКА: {str(e)}")
        print("Детальна інформація:")
        print(traceback.format_exc())
        return None


def compare_training_modes():
    """
    🔬 ЕКСПЕРИМЕНТАЛЬНА ФУНКЦІЯ: Порівняння навчання з оптимізацією та без
    """
    print("🔬 ЕКСПЕРИМЕНТ: Порівняння режимів навчання")
    print("=" * 60)

    # Підготовка даних (одноразово)
    train_df, val_df, test_df = run_data_preparation()
    X_train, y_train, X_val, y_val, X_test, y_test, _ = run_feature_engineering(train_df, val_df, test_df)

    results_comparison = {}

    # Тест 1: Без оптимізації, з відстеженням
    print(f"\n🧪 ТЕСТ 1: Без оптимізації + відстеження")
    print("-" * 40)

    pipeline_no_opt = EnhancedCryptoPricePredictionPipeline(n_forecast_periods=24)
    pipeline_no_opt.fit(X_train, y_train, X_val, y_val,
                        optimize=False, track_training=True,
                        results_dir="comparison/no_optimization")

    metrics_no_opt = pipeline_no_opt.evaluate(X_test, y_test)
    results_comparison['no_optimization'] = {
        'pipeline': pipeline_no_opt,
        'metrics': metrics_no_opt
    }

    # Тест 2: З оптимізацією, з відстеженням
    print(f"\n🧪 ТЕСТ 2: З оптимізацією + відстеження")
    print("-" * 40)

    pipeline_with_opt = EnhancedCryptoPricePredictionPipeline(n_forecast_periods=24)
    pipeline_with_opt.fit(X_train, y_train, X_val, y_val,
                          optimize=True, track_training=True,
                          results_dir="comparison/with_optimization")

    metrics_with_opt = pipeline_with_opt.evaluate(X_test, y_test)
    results_comparison['with_optimization'] = {
        'pipeline': pipeline_with_opt,
        'metrics': metrics_with_opt
    }

    # Порівняльний аналіз
    print(f"\n📊 ПОРІВНЯЛЬНИЙ АНАЛІЗ:")
    print("=" * 50)

    for mode, data in results_comparison.items():
        metrics = data['metrics']
        pipeline = data['pipeline']

        print(f"\n{mode.upper().replace('_', ' ')}:")
        print(f"   RMSE: {metrics['RMSE']:.2f}")
        print(f"   R²: {metrics['R2']:.4f}")
        print(f"   MAPE: {metrics['MAPE']:.2f}%")

        if hasattr(pipeline, 'overfitting_analyses'):
            overfitting_count = sum(1 for analysis in pipeline.overfitting_analyses.values()
                                    if 'overfitting' in analysis['status'])
            print(f"   Моделей з переобученням: {overfitting_count}/3")

    # Висновки
    rmse_improvement = metrics_no_opt['RMSE'] - metrics_with_opt['RMSE']
    print(f"\n🎯 ВИСНОВКИ:")
    if rmse_improvement > 0:
        print(f"✅ Оптимізація покращила RMSE на {rmse_improvement:.2f}")
    else:
        print(f"⚠️ Оптимізація погіршила RMSE на {abs(rmse_improvement):.2f}")

    return results_comparison


if __name__ == "__main__":
    # Основний запуск
    results = main_with_training_analysis()

    if results:
        print(f"\n✅ ПРОГРАМА ЗАВЕРШЕНА УСПІШНО!")
        print(f"🔍 Результати аналізу навчання доступні в змінній 'results'")

        # Додаткові можливості
        print(f"\n💡 ДОДАТКОВІ МОЖЛИВОСТІ:")
        print(f"   1. results['pipeline'].plot_ensemble_training_comparison() - порівняльні графіки")
        print(f"   2. results['pipeline'].print_overfitting_summary() - детальний аналіз")
        print(f"   3. compare_training_modes() - порівняння режимів навчання")

        # Автоматичне відкриття графіків (опціонально)
        try:
            if hasattr(results['pipeline'], 'training_completed') and results['pipeline'].training_completed:
                print(f"\n📊 Створення фінального порівняльного графіка...")
                results['pipeline'].plot_ensemble_training_comparison()
        except Exception as e:
            print(f"⚠️ Не вдалося створити фінальний графік: {e}")

    else:
        print(f"\n❌ ПРОГРАМА ЗАВЕРШИЛАСЯ З ПОМИЛКАМИ")
        print(f"💡 Спробуйте:")
        print(f"   1. Перевірити інтернет-з'єднання")
        print(f"   2. Встановити залежності: pip install matplotlib scikit-learn")
        print(f"   3. Зменшити складність даних у CONFIG")


# Швидкі тести
def quick_overfitting_test():
    """🚀 Швидкий тест для виявлення переобучення"""
    print("🚀 ШВИДКИЙ ТЕСТ: Аналіз переобучення на малих даних")

    # Мінімальні налаштування для швидкого тесту
    original_config = CONFIG.copy()
    CONFIG['data']['period'] = '6mo'  # Менше даних
    CONFIG['models']['optimize_hyperparams'] = False  # Без оптимізації

    try:
        results = main_with_training_analysis()
        return results
    finally:
        # Відновлюємо оригінальний конфіг
        CONFIG.update(original_config)


# Для відладки
def debug_training_tracking():
    """🐛 Відладка системи відстеження навчання"""
    print("🐛 ВІДЛАДКА: Тестування відстеження навчання")

    # Створюємо синтетичні дані
    np.random.seed(42)
    n_samples, n_features = 500, 20

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5

    # Розділяємо
    split = int(0.6 * n_samples)
    val_split = int(0.8 * n_samples)

    X_train = pd.DataFrame(X[:split], columns=[f'f_{i}' for i in range(n_features)])
    y_train = pd.Series(y[:split])
    X_val = pd.DataFrame(X[split:val_split], columns=[f'f_{i}' for i in range(n_features)])
    y_val = pd.Series(y[split:val_split])
    X_test = pd.DataFrame(X[val_split:], columns=[f'f_{i}' for i in range(n_features)])
    y_test = pd.Series(y[val_split:])

    # Тестуємо pipeline
    pipeline = EnhancedCryptoPricePredictionPipeline(n_forecast_periods=1)
    pipeline.fit(X_train, y_train, X_val, y_val,
                 optimize=False, track_training=True,
                 results_dir="debug_test")

    metrics = pipeline.evaluate(X_test, y_test)

    print("✅ Тест відстеження навчання пройшов успішно!")
    return pipeline, metrics