# 🔧 ВИПРАВЛЕНИЙ ENSEMBLE.PY - БЕЗ ПЕРЕНАВЧАННЯ
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from sklearn.model_selection import KFold


class OptimizedWeightedEnsemble:
    """
    🔧 ВИПРАВЛЕНИЙ Ансамблева модель БЕЗ перенавчання на валідаційних даних
    """

    def __init__(self, models=None, min_weight=0.1, optimization_method='equal'):
        """
        optimization_method: 'equal', 'cv', 'holdout'
        - 'equal': рівномірні ваги (найбезпечніше)
        - 'cv': оптимізація через cross-validation (безпечно)
        - 'holdout': використання окремого holdout набору (потребує додаткових даних)
        """
        self.models = models or {}
        self.weights = None
        self.feature_importances_ = None
        self.min_weight = min_weight
        self.optimization_method = optimization_method

    def add_model(self, name, model):
        """Додає модель до ансамблю"""
        self.models[name] = model
        return self

    def _optimize_weights_cv(self, X_train, y_train, cv=3):
        """🔧 БЕЗПЕЧНА оптимізація ваг через cross-validation"""
        print(f"🎯 Оптимізація ваг через {cv}-fold CV (БЕЗ перенавчання)")

        # Збираємо прогнози з CV
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        all_predictions = []
        all_targets = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_fold_train = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_fold_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_fold_train = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            y_fold_val = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]

            # Тренуємо моделі на fold
            fold_predictions = {}
            for name, model in self.models.items():
                # Клонуємо модель для цього fold
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)

                pred = fold_model.predict(X_fold_val)
                fold_predictions[name] = pred

            # Зберігаємо прогнози та цілі
            all_predictions.append(fold_predictions)
            all_targets.extend(y_fold_val)

        # Об'єднуємо прогнози з усіх fold
        combined_predictions = {}
        for name in self.models.keys():
            combined_predictions[name] = np.concatenate([
                fold_pred[name] for fold_pred in all_predictions
            ])

        all_targets = np.array(all_targets)

        # Оптимізуємо ваги
        def objective_cv(weights):
            weights = np.maximum(weights, self.min_weight)
            weights = weights / weights.sum()

            final_pred = np.zeros(len(all_targets))
            for i, (name, _) in enumerate(self.models.items()):
                final_pred += weights[i] * combined_predictions[name]

            return np.sqrt(mean_squared_error(all_targets, final_pred))

        n_models = len(self.models)
        initial_weights = np.full(n_models, 1.0 / n_models)

        try:
            result = minimize(
                objective_cv,
                initial_weights,
                method='SLSQP',
                bounds=[(self.min_weight, 1.0) for _ in range(n_models)],
                constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            )

            if result.success:
                self.weights = result.x
                print(f"✅ CV оптимізація успішна, RMSE: {result.fun:.2f}")
            else:
                self.weights = initial_weights
                print("⚠️ CV оптимізація не сходиться, використовуємо рівномірні ваги")

        except Exception as e:
            print(f"❌ Помилка CV оптимізації: {e}")
            self.weights = initial_weights

        # Фінальна нормалізація
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights = self.weights / self.weights.sum()

        return self

    def _set_equal_weights(self):
        """🔧 Встановлює рівномірні ваги (найбезпечніший варіант)"""
        print("🎯 Використання рівномірних ваг (БЕЗ оптимізації)")
        n_models = len(self.models)
        self.weights = np.full(n_models, 1.0 / n_models)
        return self

    def _optimize_weights_performance_based(self, X_train, y_train):
        """🔧 Ваги на основі індивідуальної продуктивності (без додаткової оптимізації)"""
        print("🎯 Ваги на основі індивідуальної продуктивності")

        individual_rmse = {}
        for name, model in self.models.items():
            pred = model.predict(X_train)
            rmse = np.sqrt(mean_squared_error(y_train, pred))
            individual_rmse[name] = rmse
            print(f"  {name}: RMSE = {rmse:.2f}")

        # Інвертуємо RMSE: кращі моделі отримують більші ваги
        inverted_rmse = np.array([1.0 / rmse for rmse in individual_rmse.values()])
        self.weights = inverted_rmse / inverted_rmse.sum()

        # Застосовуємо мінімальні ваги
        self.weights = np.maximum(self.weights, self.min_weight)
        self.weights = self.weights / self.weights.sum()

        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        🔧 ВИПРАВЛЕНИЙ метод fit без перенавчання

        X_val та y_val тепер НЕ використовуються для оптимізації ваг!
        """
        print(f"🔧 Навчання ансамблю методом: {self.optimization_method}")

        # Навчаємо кожну модель (якщо ще не навчена)
        for name, model in self.models.items():
            if not hasattr(model, 'predict'):
                model.fit(X_train, y_train)

        # Вибираємо метод оптимізації ваг
        if self.optimization_method == 'equal':
            self._set_equal_weights()

        elif self.optimization_method == 'cv':
            self._optimize_weights_cv(X_train, y_train, cv=3)

        elif self.optimization_method == 'performance':
            self._optimize_weights_performance_based(X_train, y_train)

        else:
            print(f"⚠️ Невідомий метод {self.optimization_method}, використовуємо рівномірні ваги")
            self._set_equal_weights()

        # Розрахунок важливості ознак
        self._calculate_feature_importance(X_train)

        # Показуємо фінальні ваги
        print("\n📊 Фінальні ваги моделей:")
        for i, (name, weight) in enumerate(zip(self.models.keys(), self.weights)):
            print(f"  {name}: {weight:.1%}")

        return self

    def _calculate_feature_importance(self, X):
        """Обчислює важливість ознак як зважену суму важливостей з окремих моделей"""
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        importance_dict = {feature: 0 for feature in feature_names}

        for i, (name, model) in enumerate(self.models.items()):
            if hasattr(model, 'feature_importances_'):
                print(f"✓ Важливість ознак отримано від {name}")
                for j, importance in enumerate(model.feature_importances_):
                    importance_dict[feature_names[j]] += self.weights[i] * importance
            else:
                print(f"⚠️ {name} не підтримує feature_importances_")

        self.feature_importances_ = importance_dict
        return self

    def predict(self, X):
        """Прогнозує цільову змінну за допомогою зваженого ансамблю"""
        if self.weights is None:
            raise ValueError("Модель не навчена. Викличте метод fit перед predict.")

        final_predictions = np.zeros(X.shape[0])
        for i, (name, model) in enumerate(self.models.items()):
            final_predictions += self.weights[i] * model.predict(X)

        return final_predictions

    def get_model_weights(self):
        """Повертає ваги моделей в ансамблі"""
        if self.weights is None:
            raise ValueError("Модель не навчена. Викличте метод fit перед отриманням ваг.")

        return {name: weight for name, weight in zip(self.models.keys(), self.weights)}


# 🔧 ВИПРАВЛЕНИЙ PREDICTION PIPELINE
class FixedCryptoPricePredictionPipeline:
    """
    🔧 Виправлений pipeline без перенавчання
    """

    def __init__(self, n_forecast_periods=1, ensemble_method='equal'):
        self.n_forecast_periods = n_forecast_periods
        self.ensemble_method = ensemble_method  # 'equal', 'cv', 'performance'
        self.feature_engineering = None
        self.feature_selector = None
        self.scaler = None
        self.ensemble = None
        self.feature_names = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, optimize=True):
        """
        🔧 ВИПРАВЛЕНИЙ метод fit
        """
        print(f"🔧 Використання ensemble методу: {self.ensemble_method}")

        # Імпорт model optimizer
        from models.model_optimizer import bayesian_optimize_models, quick_optimize_models
        from sklearn.preprocessing import StandardScaler

        # Зберігаємо назви ознак
        self.feature_names = list(X_train.columns)

        # Масштабування
        self.scaler = StandardScaler()
        X_train_scaled_array = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=self.feature_names, index=X_train.index)

        # Навчання базових моделей (БЕЗ використання валідаційних даних)
        if optimize:
            try:
                models = bayesian_optimize_models(X_train_scaled, y_train, n_iter=10)
            except Exception as e:
                print(f"Помилка оптимізації: {e}")
                models = quick_optimize_models(X_train_scaled, y_train)
        else:
            models = quick_optimize_models(X_train_scaled, y_train)

        # Створюємо ансамбль з виправленою логікою
        self.ensemble = OptimizedWeightedEnsemble(optimization_method=self.ensemble_method)

        if models.get('xgboost') is not None:
            self.ensemble.add_model('xgboost', models['xgboost'])
        if models.get('lightgbm') is not None:
            self.ensemble.add_model('lightgbm', models['lightgbm'])
        if models.get('histgb') is not None:
            self.ensemble.add_model('histgb', models['histgb'])

        if len(self.ensemble.models) == 0:
            raise Exception("Жодна модель не була успішно навчена!")

        # 🔧 КЛЮЧОВА ЗМІНА: Навчаємо ансамбль БЕЗ валідаційних даних
        print(f"\n🔧 Створення ансамблю з {len(self.ensemble.models)} моделей...")
        print(f"📊 Метод оптимізації ваг: {self.ensemble_method}")

        # Валідаційні дані НЕ передаються для оптимізації ваг!
        self.ensemble.fit(X_train_scaled, y_train)  # ❌ НЕ передаємо X_val, y_val

        # Виводимо ваги моделей
        print("\n📊 Оптимізовані ваги моделей:")
        for model_name, weight in self.ensemble.get_model_weights().items():
            print(f"  {model_name}: {weight:.1%}")

        return self

    def predict(self, X):
        """Прогнозує цільову змінну"""
        X_scaled_array = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_array, columns=self.feature_names, index=X.index)
        return self.ensemble.predict(X_scaled)

    def evaluate(self, X_test, y_test):
        """Оцінює продуктивність ансамблю"""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred = self.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"\n📈 Оцінка ансамблю ({self.ensemble_method} ваги):")
        print("=" * 60)
        mean_price = np.mean(y_test)
        mae_percent = (mae / mean_price) * 100
        rmse_percent = (rmse / mean_price) * 100
        print(f"АНСАМБЛЬ   : MAE={mae_percent:5.2f}%, RMSE={rmse_percent:5.2f}%, R²={r2:.4f}, MAPE={mape:5.2f}%")

        return {
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'Method': self.ensemble_method
        }

    def show_detailed_predictions(self, X_test, y_test, n_predictions=10):
        """
        📊 Показує детальні прогнози (з виправленого оригінального pipeline)
        """
        print(f"\n🎯 ДЕТАЛЬНІ ПРОГНОЗИ (останні {n_predictions} значень)")
        print("=" * 80)

        # Отримуємо прогнози
        y_pred = self.predict(X_test)

        # Беремо останні N прогнозів
        last_actual = y_test.iloc[-n_predictions:] if hasattr(y_test, 'iloc') else y_test[-n_predictions:]
        last_predicted = y_pred[-n_predictions:]

        # Якщо є часові мітки
        if hasattr(X_test, 'index') and len(X_test.index) >= n_predictions:
            timestamps = X_test.index[-n_predictions:]
        else:
            timestamps = range(len(last_actual) - n_predictions + 1, len(last_actual) + 1)

        print(f"{'№':<3} {'Час/Індекс':<12} {'Фактична':<10} {'Прогноз':<10} {'Похибка':<8} {'%':<6} {'Напрямок':<8}")
        print("-" * 80)

        total_error = 0
        correct_direction = 0

        for i, (timestamp, actual, predicted) in enumerate(zip(timestamps, last_actual, last_predicted)):
            error = predicted - actual
            error_pct = (error / actual) * 100 if actual != 0 else 0
            total_error += abs(error)

            # Перевірка напрямку (якщо є попереднє значення)
            if i > 0:
                # Отримуємо попереднє фактичне значення
                prev_actual = last_actual.iloc[i - 1] if hasattr(last_actual, 'iloc') else last_actual[i - 1]

                # Визначаємо напрямки
                actual_direction = "↑" if actual > prev_actual else "↓"
                pred_direction = "↑" if predicted > last_predicted[i - 1] else "↓"

                # Перевіряємо чи напрямки співпадають
                direction_correct = actual_direction == pred_direction
                if direction_correct:
                    correct_direction += 1
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

        # Статистика
        avg_error = total_error / n_predictions
        direction_accuracy = (correct_direction / (n_predictions - 1)) * 100 if n_predictions > 1 else 0

        print("-" * 80)
        print(f"📊 Статистика останніх {n_predictions} прогнозів:")
        print(f"   Середня похибка: ${avg_error:.0f}")
        print(f"   Точність напрямку: {direction_accuracy:.1f}%")

        return {
            'predictions': list(last_predicted),
            'actual': list(last_actual),
            'errors': [p - a for p, a in zip(last_predicted, last_actual)],
            'avg_error': avg_error,
            'direction_accuracy': direction_accuracy
        }

    def get_next_predictions(self, X_latest, n_steps=6, feature_names=None):
        """
        🔮 Генерує прогнози на кілька кроків вперед
        """
        print(f"\n🔮 ПРОГНОЗ НА {n_steps} КРОКІВ ВПЕРЕД")
        print("=" * 50)

        predictions = []
        current_features = X_latest.iloc[-1:].copy() if hasattr(X_latest, 'iloc') else X_latest[-1:].copy()

        for step in range(1, n_steps + 1):
            # Прогнозуємо наступне значення
            next_pred = self.predict(current_features)[0]
            predictions.append(next_pred)

            print(f"Крок {step}: ${next_pred:.0f}")

            # Оновлюємо ознаки для наступного кроку (спрощено)
            if hasattr(current_features, 'columns') and len(current_features.columns) > 0:
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
        if len(predictions) > 1:
            trend_pct = ((predictions[-1] / predictions[0] - 1) * 100)
            trend_symbol = '+' if trend_pct > 0 else ''
            print(f"📈 Загальна тенденція: {trend_symbol}{trend_pct:.1f}%")

        return predictions

    def evaluate_individual_models(self, X_test, y_test):
        """
        🆕 Оцінити кожну модель окремо (для сумісності з оригінальним кодом)
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        print("\n📊 Індивідуальна оцінка моделей:")
        print("=" * 60)

        results = {}

        # Масштабування тестових даних
        X_test_scaled_array = self.scaler.transform(X_test)
        X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=self.feature_names, index=X_test.index)

        # СТАЛО:
        mean_price = np.mean(y_test)  # Розраховуємо один раз для всіх моделей

        for name, model in self.ensemble.models.items():
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            # Конвертуємо у відсотки
            mae_percent = (mae / mean_price) * 100
            rmse_percent = (rmse / mean_price) * 100

            results[name] = {
                'MAE': mae_percent,
                'RMSE': rmse_percent,
                'R2': r2,
                'MAPE': mape
            }

            print(f"{name:12}: MAE={mae_percent:5.2f}%, RMSE={rmse_percent:5.2f}%, R²={r2:.4f}, MAPE={mape:5.2f}%")

        print("=" * 60)
        return results


