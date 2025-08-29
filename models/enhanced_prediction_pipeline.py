# enhanced_prediction_pipeline.py - З відстеженням навчання

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
import matplotlib.pyplot as plt
import os

from models.ensemble import OptimizedWeightedEnsemble
from models.enhanced_model_optimizer import (
    train_models_with_full_tracking,
    TrainingTracker
)


class EnhancedCryptoPricePredictionPipeline:
    """
    Покращений конвеєр з відстеженням train/validation curves та виявленням переобучення
    """

    def __init__(self, n_forecast_periods=1):
        self.n_forecast_periods = n_forecast_periods
        self.scaler = None
        self.ensemble = None
        self.feature_names = None

        # 🆕 Додаємо відстеження навчання
        self.training_trackers = {}
        self.overfitting_analyses = {}
        self.training_completed = False

    def fit(self, X_train, y_train, X_val=None, y_val=None, optimize=True,
            track_training=True, save_plots=True, results_dir="training_analysis"):
        """
        Навчає конвеєр з опціональним відстеженням train/val curves

        Parameters:
        -----------
        track_training : bool
            Чи відстежувати train/validation метрики під час навчання
        save_plots : bool
            Чи зберігати графіки train/val curves
        results_dir : str
            Папка для збереження результатів аналізу
        """
        print("FIT FUNCTION")
        # Створюємо val дані якщо не надані
        if X_val is None or y_val is None:
            val_size = int(len(X_train) * 0.2)
            X_val = X_train.iloc[-val_size:].copy()
            y_val = y_train.iloc[-val_size:].copy()
            X_train = X_train.iloc[:-val_size].copy()
            y_train = y_train.iloc[:-val_size].copy()

        # Зберігаємо назви ознак
        self.feature_names = list(X_train.columns)
        print(f"✓ Збережено {len(self.feature_names)} назв ознак")

        # Масштабування
        self.scaler = StandardScaler()
        X_train_scaled_array = self.scaler.fit_transform(X_train)
        X_val_scaled_array = self.scaler.transform(X_val)

        X_train_scaled = pd.DataFrame(X_train_scaled_array,
                                      columns=self.feature_names,
                                      index=X_train.index)
        X_val_scaled = pd.DataFrame(X_val_scaled_array,
                                    columns=self.feature_names,
                                    index=X_val.index)

        # 🆕 ГОЛОВНА ЗМІНА: Навчання з відстеженням або без
        if track_training and optimize:
            print("🔍 НАВЧАННЯ З ПОВНИМ ВІДСТЕЖЕННЯМ МЕТРИК")
            print("=" * 50)

            # Створюємо папку для результатів
            if save_plots:
                os.makedirs(results_dir, exist_ok=True)

            # Навчаємо моделі з відстеженням
            models, trackers, analyses = train_models_with_full_tracking(
                X_train_scaled, y_train, X_val_scaled, y_val,
                save_plots=save_plots, results_dir=results_dir
            )

            # Зберігаємо результати відстеження
            self.training_trackers = trackers
            self.overfitting_analyses = analyses
            self.training_completed = True

            # Створюємо ансамбль з навчених моделей
            self.ensemble = OptimizedWeightedEnsemble()

            # Додаємо моделі до ансамблю з оберткою для сумісності
            for name, model in models.items():
                if name == 'xgboost':
                    # XGBoost потребує спеціальної обгортки для predict
                    class XGBWrapper:
                        def __init__(self, xgb_model):
                            self.model = xgb_model
                            self.feature_importances_ = xgb_model.get_score(importance_type='weight')

                        def predict(self, X):
                            if isinstance(X, pd.DataFrame):
                                dtest = xgb.DMatrix(X)
                            else:
                                dtest = xgb.DMatrix(X)
                            return self.model.predict(dtest)

                    self.ensemble.add_model(name, XGBWrapper(model))

                elif name == 'lightgbm':
                    # LightGBM обгортка
                    class LGBWrapper:
                        def __init__(self, lgb_model):
                            self.model = lgb_model
                            self.feature_importances_ = lgb_model.feature_importance()

                        def predict(self, X):
                            return self.model.predict(X)

                    self.ensemble.add_model(name, LGBWrapper(model))
                else:
                    # HistGB вже сумісний з sklearn
                    self.ensemble.add_model(name, model)

        else:
            # Стандартне навчання (ваш оригінальний код)
            print("🚀 СТАНДАРТНЕ НАВЧАННЯ БЕЗ ВІДСТЕЖЕННЯ")

            if optimize:
                print("Оптимізація моделей...")
                from models.model_optimizer import bayesian_optimize_models
                try:
                    models = bayesian_optimize_models(X_train_scaled, y_train, n_iter=10)
                except Exception as e:
                    print(f"Помилка оптимізації: {e}")
                    from models.model_optimizer import quick_optimize_models
                    models = quick_optimize_models(X_train_scaled, y_train)
            else:
                print("Швидке навчання без оптимізації...")
                from models.model_optimizer import quick_optimize_models
                models = quick_optimize_models(X_train_scaled, y_train)

            # Створюємо ансамбль
            self.ensemble = OptimizedWeightedEnsemble()
            for name, model in models.items():
                if model is not None:
                    self.ensemble.add_model(name, model)

        # Навчаємо ансамбль і оптимізуємо ваги
        if len(self.ensemble.models) == 0:
            raise Exception("Жодна модель не була успішно навчена!")

        print(f"\nСтворення ансамблю з {len(self.ensemble.models)} моделей...")
        self.ensemble.fit(X_train_scaled, y_train, X_val_scaled, y_val)

        # Виводимо ваги
        print("Оптимізовані ваги моделей:")
        for model_name, weight in self.ensemble.get_model_weights().items():
            print(f"  {model_name}: {weight:.1%}")

        return self

    def get_training_analysis(self):
        """
        Повертає детальний аналіз навчання з графіками та діагностикою переобучення
        """
        if not self.training_completed:
            return {
                'status': 'no_tracking',
                'message': 'Навчання проводилося без відстеження метрик'
            }

        analysis = {
            'status': 'completed',
            'models_analyzed': list(self.training_trackers.keys()),
            'overfitting_analysis': self.overfitting_analyses,
            'trackers': self.training_trackers
        }

        return analysis

    def plot_ensemble_training_comparison(self, save_path=None):
        """
        Створює порівняльні графіки train/val curves для всіх моделей ансамблю
        """
        if not self.training_completed:
            print("⚠️ Графіки доступні тільки після навчання з відстеженням")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        colors = {'xgboost': 'blue', 'lightgbm': 'green', 'histgb': 'red'}

        # Loss comparison
        ax1 = axes[0, 0]
        for model_name, tracker in self.training_trackers.items():
            color = colors.get(model_name, 'black')
            ax1.plot(tracker.iterations, tracker.val_losses,
                     label=f'{model_name} Val', color=color, linestyle='-', alpha=0.8)
            ax1.plot(tracker.iterations, tracker.train_losses,
                     label=f'{model_name} Train', color=color, linestyle='--', alpha=0.6)

        ax1.set_title('🔥 Порівняння Loss Curves всіх моделей', fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # R² comparison
        ax2 = axes[0, 1]
        for model_name, tracker in self.training_trackers.items():
            color = colors.get(model_name, 'black')
            ax2.plot(tracker.iterations, tracker.val_r2s,
                     label=f'{model_name} Val R²', color=color, linestyle='-', alpha=0.8)
            ax2.plot(tracker.iterations, tracker.train_r2s,
                     label=f'{model_name} Train R²', color=color, linestyle='--', alpha=0.6)

        ax2.set_title('📊 Порівняння R² Score', fontweight='bold')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('R² Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Overfitting detection
        ax3 = axes[1, 0]
        for model_name, tracker in self.training_trackers.items():
            color = colors.get(model_name, 'black')
            if len(tracker.iterations) > 5:
                loss_diff = np.array(tracker.train_losses) - np.array(tracker.val_losses)
                ax3.plot(tracker.iterations, loss_diff,
                         label=f'{model_name}', color=color, linewidth=2)

        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax3.set_title('🔍 Детекція переобучення (Train - Val Loss)', fontweight='bold')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Train Loss - Val Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Final comparison bar chart
        ax4 = axes[1, 1]
        model_names = list(self.overfitting_analyses.keys())
        final_val_losses = [self.overfitting_analyses[name]['final_val_loss'] for name in model_names]
        final_r2_scores = [self.overfitting_analyses[name]['val_r2'] for name in model_names]

        x_pos = np.arange(len(model_names))
        bars = ax4.bar(x_pos, final_val_losses, color=[colors.get(name, 'gray') for name in model_names], alpha=0.7)

        # Додаємо R² як текст на барах
        for i, (bar, r2) in enumerate(zip(bars, final_r2_scores)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height + height * 0.02,
                     f'R²: {r2:.3f}', ha='center', va='bottom', fontweight='bold')

        ax4.set_title('🏆 Фінальні результати моделей', fontweight='bold')
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Final Validation RMSE')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_names)
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Порівняльний графік збережено: {save_path}")

        plt.show()

    def print_overfitting_summary(self):
        """
        Виводить детальний текстовий звіт про переобучення
        """
        if not self.training_completed:
            print("⚠️ Аналіз переобучення доступний тільки після навчання з відстеженням")
            return

        print("\n" + "=" * 70)
        print("🔍 ДЕТАЛЬНИЙ АНАЛІЗ ПЕРЕОБУЧЕННЯ МОДЕЛЕЙ")
        print("=" * 70)

        for model_name, analysis in self.overfitting_analyses.items():
            print(f"\n🤖 {model_name.upper()}:")
            print(f"   Статус: {analysis['message']}")
            print(f"   📊 Фінальні метрики:")
            print(f"      Train RMSE: {analysis['final_train_loss']:.2f}")
            print(f"      Val RMSE:   {analysis['final_val_loss']:.2f}")
            print(f"      Loss Gap:   {analysis['loss_gap']:+.2f}")
            print(f"      Val R²:     {analysis['val_r2']:.4f}")

            # Рекомендації
            if analysis['status'] == 'severe_overfitting':
                print(f"   💡 Рекомендації:")
                print(f"      • Збільшити регуляризацію (L1/L2)")
                print(f"      • Зменшити learning_rate")
                print(f"      • Додати dropout/noise")
                print(f"      • Зменшити складність моделі")

            elif analysis['status'] == 'moderate_overfitting':
                print(f"   💡 Рекомендації:")
                print(f"      • Легко збільшити регуляризацію")
                print(f"      • Додати раннє зупинення")
                print(f"      • Перевірити якість валідаційних даних")

            elif analysis['status'] == 'underfitting':
                print(f"   💡 Рекомендації:")
                print(f"      • Збільшити складність моделі")
                print(f"      • Зменшити регуляризацію")
                print(f"      • Збільшити кількість ітерацій")
                print(f"      • Перевірити якість ознак")

            elif analysis['status'] == 'good_fit':
                print(f"   ✅ Модель добре генералізує!")

        # Загальні рекомендації для ансамблю
        print(f"\n🎯 РЕКОМЕНДАЦІЇ ДЛЯ АНСАМБЛЮ:")

        overfitting_models = [name for name, analysis in self.overfitting_analyses.items()
                              if 'overfitting' in analysis['status']]

        if len(overfitting_models) > 1:
            print(f"   ⚠️  Кілька моделей показують переобучення: {', '.join(overfitting_models)}")
            print(f"   💡 Розгляньте зменшення їх ваг в ансамблі")

        elif len(overfitting_models) == 1:
            print(f"   ⚠️  Модель {overfitting_models[0]} показує переобучення")
            print(f"   💡 Розгляньте налаштування її параметрів")

        else:
            print(f"   ✅ Всі моделі показують хорошу генералізацію!")

        # Найкраща модель
        best_model = min(self.overfitting_analyses.items(),
                         key=lambda x: x[1]['final_val_loss'])
        print(f"\n🏆 НАЙКРАЩА МОДЕЛЬ: {best_model[0].upper()}")
        print(f"   Val RMSE: {best_model[1]['final_val_loss']:.2f}")
        print(f"   Val R²: {best_model[1]['val_r2']:.4f}")

    def evaluate(self, X_test, y_test):
        """
        Розширена оцінка з урахуванням аналізу навчання
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Стандартна оцінка
        if hasattr(self, 'ensemble') and self.ensemble:
            y_pred = self.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

            metrics = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
            }

            # Додаємо інформацію про навчання якщо доступна
            if self.training_completed:
                metrics['training_analysis'] = self.get_training_analysis()

                print(f"\n📊 РЕЗУЛЬТАТИ ТЕСТУВАННЯ:")
                print(f"   MAE: {mae:.2f}")
                print(f"   RMSE: {rmse:.2f}")
                print(f"   R²: {r2:.4f}")
                print(f"   MAPE: {mape:.2f}%")

                # Порівняння з validation результатами
                print(f"\n🔍 ПОРІВНЯННЯ З VALIDATION:")
                for model_name, analysis in self.overfitting_analyses.items():
                    val_rmse = analysis['final_val_loss']
                    generalization_gap = rmse - val_rmse
                    print(f"   {model_name}: Val RMSE {val_rmse:.2f} → Test gap {generalization_gap:+.2f}")

            return metrics
        else:
            raise ValueError("Модель не навчена!")

    def predict(self, X):
        """Стандартний predict з перевіркою"""
        if not hasattr(self, 'ensemble') or self.ensemble is None:
            raise ValueError("Модель не навчена!")

        if not hasattr(self, 'scaler') or self.scaler is None:
            raise ValueError("Scaler не ініціалізований!")

        # Масштабування
        X_scaled_array = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled_array,
                                columns=self.feature_names,
                                index=X.index)

        return self.ensemble.predict(X_scaled)

    def save_training_analysis(self, save_dir="training_analysis"):
        """
        Зберігає повний аналіз навчання в файли
        """
        if not self.training_completed:
            print("⚠️ Немає аналізу для збереження")
            return

        import os
        import json
        from datetime import datetime

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Зберігаємо порівняльний графік
        comparison_plot_path = os.path.join(save_dir, f"model_comparison_{timestamp}.png")
        self.plot_ensemble_training_comparison(comparison_plot_path)

        # Зберігаємо аналіз у JSON
        analysis_path = os.path.join(save_dir, f"overfitting_analysis_{timestamp}.json")

        # Підготовка даних для JSON (видаляємо trackers - вони містять numpy arrays)
        json_data = {
            'timestamp': timestamp,
            'models_analyzed': list(self.overfitting_analyses.keys()),
            'overfitting_analysis': {}
        }

        for model_name, analysis in self.overfitting_analyses.items():
            # Конвертуємо numpy типи в звичайні Python типи
            clean_analysis = {}
            for key, value in analysis.items():
                if isinstance(value, (np.integer, np.floating)):
                    clean_analysis[key] = float(value)
                else:
                    clean_analysis[key] = value
            json_data['overfitting_analysis'][model_name] = clean_analysis

        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # Зберігаємо текстовий звіт
        report_path = os.path.join(save_dir, f"training_report_{timestamp}.txt")

        import sys
        from io import StringIO

        # Перехоплюємо print output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        self.print_overfitting_summary()

        sys.stdout = old_stdout
        report_text = captured_output.getvalue()

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"ЗВІТ ПРО НАВЧАННЯ МОДЕЛЕЙ\n")
            f.write(f"Створено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")
            f.write(report_text)

        print(f"\n💾 АНАЛІЗ ЗБЕРЕЖЕНО:")
        print(f"   📊 Графіки: {comparison_plot_path}")
        print(f"   📄 JSON аналіз: {analysis_path}")
        print(f"   📝 Текстовий звіт: {report_path}")

        return {
            'comparison_plot': comparison_plot_path,
            'analysis_json': analysis_path,
            'text_report': report_path
        }