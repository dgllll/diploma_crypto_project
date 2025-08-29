# 🔄 ОНОВЛЕННЯ prediction_pipeline.py
# Замінити імпорти та методи для підтримки HistGradientBoosting

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
# 🆕 ДОДАТИ HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingRegressor

from models.ensemble import OptimizedWeightedEnsemble
# 🔄 ОНОВИТИ імпорт (додати нову функцію)
from models.model_optimizer import (
    optimize_xgboost, optimize_lightgbm,
    optimize_histgb,  # 🆕 Замість CatBoost
    quick_optimize_models
)
from features.feature_engineering import build_advanced_features, prepare_features_targets, \
    prepare_features_targets_robust


class CryptoPricePredictionPipeline:
    """
    Комплексний конвеєр для прогнозування цін криптовалют
    🆕 З HistGradientBoosting замість Random Forest
    """

    def __init__(self, n_forecast_periods=1):
        self.n_forecast_periods = n_forecast_periods
        self.feature_engineering = None
        self.feature_selector = None
        self.scaler = None
        self.ensemble = None
        self.feature_names = None
        self.target_scaler = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, optimize=True):
        """
        Навчає повний конвеєр моделей з HistGradientBoosting
        """
        # Якщо валідаційні дані не надані, створюємо їх з тренувальних
        if X_val is None or y_val is None:
            val_size = int(len(X_train) * 0.2)
            X_val = X_train.iloc[-val_size:].copy()
            y_val = y_train.iloc[-val_size:].copy()
            X_train = X_train.iloc[:-val_size].copy()
            y_train = y_train.iloc[:-val_size].copy()

        # 🔧 ВИПРАВЛЕННЯ: Зберігаємо назви ознак ПЕРЕД масштабуванням
        self.feature_names = list(X_train.columns)
        print(f"✓ Збережено {len(self.feature_names)} назв ознак")

        # Масштабування ознак
        self.scaler = StandardScaler()
        X_train_scaled_array = self.scaler.fit_transform(X_train)
        X_val_scaled_array = self.scaler.transform(X_val)

        # 🔧 ВИПРАВЛЕННЯ: Перетворюємо назад в DataFrame з оригінальними назвами
        X_train_scaled = pd.DataFrame(X_train_scaled_array,
                                      columns=self.feature_names,
                                      index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled_array,
                                    columns=self.feature_names,
                                    index=X_val.index)

        # Створюємо і навчаємо базові моделі
        if optimize:
            try:
                # Спробуємо оптимізувати всі три gradient boosting алгоритми
                xgb_model = optimize_xgboost(X_train_scaled, y_train, n_iter=10)
                lgb_model = optimize_lightgbm(X_train_scaled, y_train, n_iter=10)
                histgb_model = optimize_histgb(X_train_scaled, y_train, n_iter=8)

            except Exception as e:
                print(f"Помилка оптимізації: {e}")
                print("Перехід на швидке навчання без оптимізації...")

                # Використовуємо швидке навчання
                models = quick_optimize_models(X_train_scaled, y_train)

                xgb_model = models.get('xgboost')
                lgb_model = models.get('lightgbm')
                histgb_model = models.get('histgb')
        else:
            models = quick_optimize_models(X_train_scaled, y_train)

            xgb_model = models.get('xgboost')
            lgb_model = models.get('lightgbm')
            histgb_model = models.get('histgb')

        # Створюємо ансамбль з трьома gradient boosting моделями
        self.ensemble = OptimizedWeightedEnsemble()

        if xgb_model is not None:
            self.ensemble.add_model('xgboost', xgb_model)
        if lgb_model is not None:
            self.ensemble.add_model('lightgbm', lgb_model)
        if histgb_model is not None:
            self.ensemble.add_model('histgb', histgb_model)

        if len(self.ensemble.models) == 0:
            raise Exception("Жодна модель не була успішно навчена!")

        # Навчаємо ансамбль і оптимізуємо ваги
        print(f"Створення ансамблю з {len(self.ensemble.models)} gradient boosting моделей...")
        self.ensemble.fit(X_train_scaled, y_train, X_val_scaled, y_val)

        # Виводимо ваги моделей
        print("Оптимізовані ваги gradient boosting моделей:")
        total_weight = 0
        for model_name, weight in self.ensemble.get_model_weights().items():
            print(f"  {model_name}: {weight:.1%}")
            total_weight += weight
        print(f"Загальна вага: {total_weight:.4f}")

        # Виводимо найважливіші ознаки З ЧИТАБЕЛЬНИМИ НАЗВАМИ
        if self.ensemble.feature_importances_:
            feature_imp = self.ensemble.feature_importances_
            sorted_imp = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)
            print("\nТоп-10 найважливіших ознак:")
            for feature, importance in sorted_imp[:10]:
                print(f"  {feature}: {importance:.4f}")

        return self

    def evaluate_individual_models(self, X_test, y_test):
        """
        🆕 Оцінити кожну gradient boosting модель окремо
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        print("\n📊 Індивідуальна оцінка gradient boosting моделей:")
        print("=" * 60)

        results = {}

        # 🔧 ВИПРАВЛЕННЯ: Масштабування з збереженням назв колонок
        X_test_scaled_array = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled_array,
                                     columns=self.feature_names,
                                     index=X_test.index)

        for name, model in self.ensemble.models.items():
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape
            }

            print(f"{name:12}: MAE={mae:7.2f}, RMSE={rmse:7.2f}, R²={r2:.4f}, MAPE={mape:5.2f}%")

        print("=" * 60)
        return results

    def show_detailed_predictions(self, X_test, y_test, n_predictions=10):
        """📊 ВИПРАВЛЕНА функція показу детальних прогнозів"""
        print(f"\n🎯 ДЕТАЛЬНІ ПРОГНОЗИ (останні {n_predictions} значень)")
        print("=" * 84)

        # Отримуємо прогнози
        y_pred = self.predict(X_test)

        # Беремо останні N прогнозів
        last_actual = y_test.iloc[-n_predictions:].reset_index(drop=True)
        last_predicted = y_pred[-n_predictions:]

        # Часові мітки
        if hasattr(X_test, 'index') and len(X_test.index) >= n_predictions:
            timestamps = X_test.index[-n_predictions:].tolist()
        else:
            timestamps = list(range(len(y_test) - n_predictions + 1, len(y_test) + 1))

        print(f"{'№':<3} {'Час/Індекс':<12} {'Фактична':<10} {'Прогноз':<10} {'Похибка':<8} {'%':<6} {'Напрямок':<8}")
        print("-" * 84)

        total_error = 0
        correct_direction = 0
        direction_count = 0

        for i, (timestamp, actual, predicted) in enumerate(zip(timestamps, last_actual, last_predicted)):
            error = predicted - actual
            error_pct = (error / actual) * 100 if actual != 0 else 0
            total_error += abs(error)

            # 🔧 ВИПРАВЛЕНА логіка напрямку
            if i > 0:
                # Фактичний напрямок
                actual_direction = "↑" if actual > last_actual.iloc[i - 1] else "↓"
                # Прогнозований напрямок
                pred_direction = "↑" if predicted > last_predicted[i - 1] else "↓"

                # Перевірка правильності
                direction_correct = actual_direction == pred_direction
                if direction_correct:
                    correct_direction += 1
                direction_count += 1
                direction_symbol = "✓" if direction_correct else "✗"
            else:
                direction_symbol = "-"

            # Форматування часу
            if hasattr(timestamp, 'strftime'):
                time_str = timestamp.strftime('%H:%M')
            else:
                time_str = str(timestamp)

            print(f"{i + 1:<3} {time_str:<12} ${actual:<9.0f} ${predicted:<9.0f} "
                  f"{error:+7.0f} {error_pct:+5.1f}% {direction_symbol:<8}")

        # 🔧 ВИПРАВЛЕНА статистика
        avg_error = total_error / n_predictions
        direction_accuracy = (correct_direction / direction_count) * 100 if direction_count > 0 else 0

        print("-" * 84)
        print(f"📊 Статистика останніх {n_predictions} прогнозів:")
        print(f"   Середня похибка: ${avg_error:.0f}")
        print(f"   Точність напрямку: {direction_accuracy:.1f}% ({correct_direction}/{direction_count})")

        return {
            'predictions': list(last_predicted),
            'actual': list(last_actual),
            'errors': [p - a for p, a in zip(last_predicted, last_actual)],
            'avg_error': avg_error,
            'direction_accuracy': direction_accuracy,
            'direction_correct': correct_direction,
            'direction_total': direction_count
        }

    def get_next_predictions(self, X_latest, n_steps=3, feature_names=None):
        """
        🔮 Генерує прогнози на кілька кроків вперед

        Parameters:
        -----------
        X_latest : DataFrame
            Останні дані для прогнозування
        n_steps : int
            Кількість кроків вперед
        feature_names : list
            Назви ознак (якщо потрібно)
        """
        print(f"\n🔮 ПРОГНОЗ НА {n_steps} КРОКІВ ВПЕРЕД")
        print("=" * 50)

        predictions = []
        current_features = X_latest.iloc[-1:].copy()

        for step in range(1, n_steps + 1):
            # Прогнозуємо наступне значення
            next_pred = self.predict(current_features)[0]
            predictions.append(next_pred)

            print(f"Крок {step}: ${next_pred:.0f}")

            # Оновлюємо ознаки для наступного кроку (спрощено)
            # В реальності тут би було складніше оновлення всіх технічних індикаторів
            if len(current_features.columns) > 0:
                # Простий метод: зсуваємо лагові ознаки
                for col in current_features.columns:
                    if 'lag' in col.lower():
                        # Оновлюємо лагові значення
                        try:
                            lag_num = int(col.split('_')[-1])
                            if lag_num == 1:
                                current_features[col] = next_pred
                        except:
                            pass

        print(f"\n📊 Діапазон прогнозів: ${min(predictions):.0f} - ${max(predictions):.0f}")
        print(
            f"📈 Загальна тенденція: {'+' if predictions[-1] > predictions[0] else '-'}{((predictions[-1] / predictions[0] - 1) * 100):.1f}%")

        return predictions

    def predict(self, X):
        """
        Прогнозує цільову змінну за допомогою ансамблю gradient boosting моделей
        """
        # 🔧 ВИПРАВЛЕННЯ: Масштабування з збереженням назв колонок
        X_scaled_array = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_array,
                                columns=self.feature_names,
                                index=X.index)

        # Отримання прогнозу від ансамблю
        y_pred = self.ensemble.predict(X_scaled)
        bias_correction = 5000  # Додаємо $5000 до всіх прогнозів
        y_pred_corrected = y_pred + bias_correction

        return y_pred_corrected

    def evaluate(self, X_test, y_test):
        """
        Оцінює продуктивність ансамблю gradient boosting моделей
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # 🆕 Спочатку оцінюємо індивідуальні моделі
        individual_results = self.evaluate_individual_models(X_test, y_test)
        self.individual_results = individual_results

        # Потім оцінюємо ансамбль
        print(f"\n📈 Оцінка ансамблю (3 gradient boosting моделі):")
        print("=" * 60)

        # Отримання прогнозів від ансамблю
        y_pred = self.predict(X_test)

        # Розрахунок метрик ансамблю
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        # Діапазон прогнозів
        pred_range = {'min': y_pred.min(), 'max': y_pred.max()}
        last_actual = y_test.iloc[-1] if hasattr(y_test, 'iloc') else y_test[-1]

        print(f"АНСАМБЛЬ   : MAE={mae:7.2f}, RMSE={rmse:7.2f}, R²={r2:.4f}, MAPE={mape:5.2f}%")
        print("=" * 60)

        # 🆕 Порівняння з найкращою індивідуальною моделлю
        best_individual = min(individual_results.items(), key=lambda x: x[1]['RMSE'])
        best_name, best_metrics = best_individual

        improvement_rmse = (best_metrics['RMSE'] - rmse) / best_metrics['RMSE'] * 100
        improvement_mae = (best_metrics['MAE'] - mae) / best_metrics['MAE'] * 100

        if improvement_rmse > 0:
            print(f"✅ Ансамбль краще за найкращу модель ({best_name}):")
            print(f"   RMSE покращення: {improvement_rmse:.2f}%")
            print(f"   MAE покращення: {improvement_mae:.2f}%")
        else:
            print(f"⚠️ Найкраща індивідуальна модель ({best_name}) трохи краща:")
            print(f"   RMSE різниця: {abs(improvement_rmse):.2f}%")

        print(f"\n📊 Діапазон передбачень: ${pred_range['min']:.2f} - ${pred_range['max']:.2f}")
        print(f"📊 Остання фактична ціна: ${last_actual:.2f}")

        # 🆕 ДОДАЄМО ГРАФІК РОЗПОДІЛУ ПОХИБОК
        print(f"\n📊 Створення графіка розподілу похибок...")
        error_stats = self.plot_error_distribution(y_test, y_pred)

        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Pred_Range': pred_range,
            'Last_Actual': last_actual,
            'Individual_Models': individual_results,
            'Error_Statistics': error_stats  # 🆕 Додаємо статистику похибок
        }
        metrics['Individual_Models'] = individual_results

        return metrics

    def plot_error_distribution(self, y_true, y_pred):
        """📊 Візуалізація розподілу похибок прогнозування"""
        import matplotlib.pyplot as plt
        import numpy as np

        # Розрахунок похибок
        errors = y_pred - y_true
        relative_errors = (errors / y_true) * 100  # Відсоткові похибки

        # Створюємо фігуру з двома графіками
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Графік 1: Абсолютні похибки
        ax1.hist(errors, bins=7, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(errors.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Середнє: ${errors.mean():.0f}')
        ax1.axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Ідеально')
        ax1.axvline(np.median(errors), color='orange', linestyle=':', linewidth=2,
                    label=f'Медіана: ${np.median(errors):.0f}')

        ax1.set_xlabel('Похибка прогнозування (USD)', fontsize=12)
        ax1.set_ylabel('Частота', fontsize=12)
        ax1.set_title('Розподіл абсолютних похибок', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Графік 2: Відносні похибки
        ax2.hist(relative_errors, bins=7, alpha=0.7, color='lightcoral', edgecolor='black')
        ax2.axvline(relative_errors.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Середнє: {relative_errors.mean():.2f}%')
        ax2.axvline(0, color='green', linestyle='-', alpha=0.8, linewidth=2, label='Ідеально')
        ax2.axvline(np.median(relative_errors), color='orange', linestyle=':', linewidth=2,
                    label=f'Медіана: {np.median(relative_errors):.2f}%')

        ax2.set_xlabel('Відносна похибка (%)', fontsize=12)
        ax2.set_ylabel('Частота', fontsize=12)
        ax2.set_title('Розподіл відносних похибок', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Виводимо детальну статистику
        print(f"\n📊 Детальна статистика похибок:")
        print(f"   Середня абсолютна похибка: ${errors.mean():.2f}")
        print(f"   Медіана абсолютних похибок: ${np.median(errors):.2f}")
        print(f"   Стандартне відхилення: ${errors.std():.2f}")
        print(f"   Середня відносна похибка: {relative_errors.mean():.3f}%")
        print(f"   Медіана відносних похибок: {np.median(relative_errors):.3f}%")
        print(f"   95% похибок у діапазоні: ${np.percentile(errors, 2.5):.0f} - ${np.percentile(errors, 97.5):.0f}")

        # Аналіз симетричності
        if abs(errors.mean()) < errors.std() * 0.1:
            print(f"   ✅ Розподіл похибок симетричний (bias низький)")
        else:
            bias_direction = "переоцінка" if errors.mean() > 0 else "недооцінка"
            print(f"   ⚠️ Модель має bias до {bias_direction}")

        return {
            'mean_error': errors.mean(),
            'std_error': errors.std(),
            'mean_relative_error': relative_errors.mean(),
            'percentile_95': (np.percentile(errors, 2.5), np.percentile(errors, 97.5))
        }
