# enhanced_model_optimizer.py - Розширена версія з відстеженням втрат

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform


# Зберігаємо оригінальний код виправлення numpy
def fix_numpy_compatibility():
    """Відновлює застарілі numpy алиаси для сумісності з scikit-optimize"""
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'complex'):
        np.complex = complex


fix_numpy_compatibility()

# Придушуємо deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.int.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.float.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.int.*")


# Клас для відстеження метрик під час навчання
class TrainingTracker:
    """Клас для відстеження train/validation метрик під час навчання"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.val_losses = []
        self.train_maes = []
        self.val_maes = []
        self.train_r2s = []
        self.val_r2s = []
        self.iterations = []

    def add_metrics(self, iteration, train_loss, val_loss, train_mae, val_mae, train_r2, val_r2):
        self.iterations.append(iteration)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_maes.append(train_mae)
        self.val_maes.append(val_mae)
        self.train_r2s.append(train_r2)
        self.val_r2s.append(val_r2)

    def plot_training_curves(self, model_name="Model", save_path=None):
        """Створює графіки train/validation curves"""
        if not self.iterations:
            print(f"⚠️ Немає даних для побудови графіків для {model_name}")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Loss curves
        ax1.plot(self.iterations, self.train_losses, 'b-', label='Train Loss', linewidth=2)
        ax1.plot(self.iterations, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title(f'{model_name}: Loss Curves')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('RMSE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE curves
        ax2.plot(self.iterations, self.train_maes, 'b-', label='Train MAE', linewidth=2)
        ax2.plot(self.iterations, self.val_maes, 'r-', label='Validation MAE', linewidth=2)
        ax2.set_title(f'{model_name}: MAE Curves')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # R² curves
        ax3.plot(self.iterations, self.train_r2s, 'b-', label='Train R²', linewidth=2)
        ax3.plot(self.iterations, self.val_r2s, 'r-', label='Validation R²', linewidth=2)
        ax3.set_title(f'{model_name}: R² Curves')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('R² Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Overfitting detection
        if len(self.iterations) > 10:
            # Розраховуємо різницю між train та validation
            loss_diff = np.array(self.train_losses) - np.array(self.val_losses)
            ax4.plot(self.iterations, loss_diff, 'purple', linewidth=2, label='Train - Val Loss')
            ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax4.set_title(f'{model_name}: Overfitting Detection')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Train Loss - Val Loss')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            # Додаємо текст з діагностикою
            final_diff = loss_diff[-1]
            if final_diff < -100:  # Validation loss значно більший
                overfitting_status = "🔴 СИЛЬНЕ ПЕРЕОБУЧЕННЯ"
                color = 'red'
            elif final_diff < -50:
                overfitting_status = "🟡 ПОМІРНЕ ПЕРЕОБУЧЕННЯ"
                color = 'orange'
            elif final_diff < 0:
                overfitting_status = "🟢 МІНІМАЛЬНЕ ПЕРЕОБУЧЕННЯ"
                color = 'green'
            else:
                overfitting_status = "🔵 НЕДООБУЧЕННЯ"
                color = 'blue'

            ax4.text(0.02, 0.98, overfitting_status, transform=ax4.transAxes,
                     verticalalignment='top', fontweight='bold', color=color,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Графік збережено: {save_path}")

        plt.show()

    def get_overfitting_analysis(self):
        """Повертає аналіз переобучення"""
        if len(self.iterations) < 5:
            return {"status": "insufficient_data", "message": "Недостатньо даних для аналізу"}

        # Аналізуємо останні 5 ітерацій
        recent_train = self.train_losses[-5:]
        recent_val = self.val_losses[-5:]

        # Тренди
        train_trend = np.polyfit(range(5), recent_train, 1)[0]  # slope
        val_trend = np.polyfit(range(5), recent_val, 1)[0]

        # Різниця
        final_gap = self.val_losses[-1] - self.train_losses[-1]
        avg_gap = np.mean(np.array(self.val_losses[-10:]) - np.array(self.train_losses[-10:]))

        analysis = {
            "final_train_loss": self.train_losses[-1],
            "final_val_loss": self.val_losses[-1],
            "loss_gap": final_gap,
            "avg_loss_gap": avg_gap,
            "train_trend": train_trend,  # negative = decreasing (good)
            "val_trend": val_trend,
            "train_r2": self.train_r2s[-1],
            "val_r2": self.val_r2s[-1],
        }

        # Діагностика
        if final_gap > 200 and val_trend > 0:
            analysis["status"] = "severe_overfitting"
            analysis["message"] = "🔴 Сильне переобучення: validation loss зростає"
        elif final_gap > 100:
            analysis["status"] = "moderate_overfitting"
            analysis["message"] = "🟡 Помірне переобучення: значний розрив"
        elif final_gap > 0 and abs(train_trend) < abs(val_trend):
            analysis["status"] = "mild_overfitting"
            analysis["message"] = "🟢 Легке переобучення: validation менш стабільний"
        elif final_gap < -50:
            analysis["status"] = "underfitting"
            analysis["message"] = "🔵 Недообучення: train loss більший за validation"
        else:
            analysis["status"] = "good_fit"
            analysis["message"] = "✅ Хороша генералізація"

        return analysis


def train_xgboost_with_tracking(X_train, y_train, X_val, y_val, params=None, n_estimators=200):
    """XGBoost з відстеженням метрик (ВИПРАВЛЕНО для XGBoost 1.6+)"""
    tracker = TrainingTracker()

    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.04,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 1.5,
            'reg_lambda': 4,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'tree_method': 'hist'
        }

    print("🚀 Навчання XGBoost з відстеженням метрик (нова версія)...")

    # 🔧 ВИПРАВЛЕННЯ: Новий стиль callback для XGBoost 1.6+
    class TrackingCallback(xgb.callback.TrainingCallback):
        def __init__(self, X_train, y_train, X_val, y_val, tracker):
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            self.tracker = tracker

        def after_iteration(self, model, epoch, evals_log):
            # Отримуємо прогнози
            dtrain = xgb.DMatrix(self.X_train)
            dval = xgb.DMatrix(self.X_val)

            train_pred = model.predict(dtrain)
            val_pred = model.predict(dval)

            # Розраховуємо метрики
            train_loss = np.sqrt(mean_squared_error(self.y_train, train_pred))
            val_loss = np.sqrt(mean_squared_error(self.y_val, val_pred))
            train_mae = mean_absolute_error(self.y_train, train_pred)
            val_mae = mean_absolute_error(self.y_val, val_pred)
            train_r2 = r2_score(self.y_train, train_pred)
            val_r2 = r2_score(self.y_val, val_pred)

            self.tracker.add_metrics(epoch, train_loss, val_loss, train_mae, val_mae, train_r2, val_r2)

            # Виводимо прогрес кожні 50 ітерацій
            if epoch % 50 == 0:
                print(f"  Iter {epoch:3d}: Train RMSE={train_loss:.2f}, Val RMSE={val_loss:.2f}, Val R²={val_r2:.4f}")

            return False  # Продовжуємо навчання

    # Навчання з новим callback
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Створюємо callback
    tracking_cb = TrackingCallback(X_train, y_train, X_val, y_val, tracker)

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        callbacks=[tracking_cb],
        verbose_eval=False
    )

    return model, tracker


def train_lightgbm_with_tracking(X_train, y_train, X_val, y_val, params=None, n_estimators=200):
    """LightGBM з відстеженням метрик"""
    tracker = TrainingTracker()

    if params is None:
        params = {
            'n_estimators': n_estimators,
            'num_leaves': 25,
            'learning_rate': 0.025,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'min_child_samples': 20,
            'reg_alpha': 3.0,
            'reg_lambda': 3.0,
            'random_state': 42,
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'verbose': -1
        }

    print("🚀 Навчання LightGBM з відстеженням метрик...")

    # Створюємо datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Callback для відстеження
    def tracking_callback(env):
        if env.iteration % 20 == 0:  # Кожні 20 ітерацій
            # Отримуємо прогнози
            train_pred = env.model.predict(X_train, num_iteration=env.iteration)
            val_pred = env.model.predict(X_val, num_iteration=env.iteration)

            # Розраховуємо метрики
            train_loss = np.sqrt(mean_squared_error(y_train, train_pred))
            val_loss = np.sqrt(mean_squared_error(y_val, val_pred))
            train_mae = mean_absolute_error(y_train, train_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)

            tracker.add_metrics(env.iteration, train_loss, val_loss, train_mae, val_mae, train_r2, val_r2)

            if env.iteration % 80 == 0:
                print(
                    f"  Iter {env.iteration:3d}: Train RMSE={train_loss:.2f}, Val RMSE={val_loss:.2f}, Val R²={val_r2:.4f}")

    # Навчання
    model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'eval'],
        callbacks=[tracking_callback],
    )

    return model, tracker


def train_histgb_with_tracking(X_train, y_train, X_val, y_val, params=None):
    """HistGradientBoosting з відстеженням метрик (поетапне навчання)"""
    tracker = TrainingTracker()

    if params is None:
        params = {
            'max_iter': 200,
            'max_depth': 6,
            'learning_rate': 0.04,
            'max_leaf_nodes': 35,
            'min_samples_leaf': 15,
            'l2_regularization': 3.0,
            'max_bins': 200,
            'random_state': 42,
        }

    print("🚀 Навчання HistGradientBoosting з відстеженням метрик...")

    # Для HistGB потрібно симулювати поетапне навчання
    max_iter = params.pop('max_iter', 200)
    step_size = 20

    model = HistGradientBoostingRegressor(max_iter=step_size, **params)

    for i in range(0, max_iter, step_size):
        current_iter = min(i + step_size, max_iter)

        if i == 0:
            # Перше навчання
            model.fit(X_train, y_train)
        else:
            # Продовжуємо навчання (warm_start симуляція)
            model.max_iter = current_iter
            model.fit(X_train, y_train)

        # Отримуємо прогнози
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        # Розраховуємо метрики
        train_loss = np.sqrt(mean_squared_error(y_train, train_pred))
        val_loss = np.sqrt(mean_squared_error(y_val, val_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        tracker.add_metrics(current_iter, train_loss, val_loss, train_mae, val_mae, train_r2, val_r2)

        if current_iter % 40 == 0:
            print(
                f"  Iter {current_iter:3d}: Train RMSE={train_loss:.2f}, Val RMSE={val_loss:.2f}, Val R²={val_r2:.4f}")

    return model, tracker


def train_models_with_full_tracking(X_train, y_train, X_val, y_val, save_plots=True, results_dir="results"):
    """Навчає всі три моделі з повним відстеженням"""
    print("🔥 НАВЧАННЯ МОДЕЛЕЙ З ПОВНИМ ВІДСТЕЖЕННЯМ МЕТРИК")
    print("=" * 60)

    models = {}
    trackers = {}
    analyses = {}

    # Створюємо папку для збереження
    if save_plots:
        import os
        os.makedirs(results_dir, exist_ok=True)

    # 1. XGBoost
    print("\n1️⃣ XGBoost:")
    model_xgb, tracker_xgb = train_xgboost_with_tracking(X_train, y_train, X_val, y_val)
    models['xgboost'] = model_xgb
    trackers['xgboost'] = tracker_xgb
    analyses['xgboost'] = tracker_xgb.get_overfitting_analysis()

    if save_plots:
        tracker_xgb.plot_training_curves("XGBoost", f"{results_dir}/xgboost_training_curves.png")

    # 2. LightGBM
    print("\n2️⃣ LightGBM:")
    model_lgb, tracker_lgb = train_lightgbm_with_tracking(X_train, y_train, X_val, y_val)
    models['lightgbm'] = model_lgb
    trackers['lightgbm'] = tracker_lgb
    analyses['lightgbm'] = tracker_lgb.get_overfitting_analysis()

    if save_plots:
        tracker_lgb.plot_training_curves("LightGBM", f"{results_dir}/lightgbm_training_curves.png")

    # 3. HistGradientBoosting
    print("\n3️⃣ HistGradientBoosting:")
    model_histgb, tracker_histgb = train_histgb_with_tracking(X_train, y_train, X_val, y_val)
    models['histgb'] = model_histgb
    trackers['histgb'] = tracker_histgb
    analyses['histgb'] = tracker_histgb.get_overfitting_analysis()

    if save_plots:
        tracker_histgb.plot_training_curves("HistGradientBoosting", f"{results_dir}/histgb_training_curves.png")

    # Підсумковий аналіз
    print("\n" + "=" * 60)
    print("🔍 АНАЛІЗ ПЕРЕОБУЧЕННЯ:")
    print("=" * 60)

    for model_name, analysis in analyses.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Статус: {analysis['message']}")
        print(f"  Final Train RMSE: {analysis['final_train_loss']:.2f}")
        print(f"  Final Val RMSE: {analysis['final_val_loss']:.2f}")
        print(f"  Loss Gap: {analysis['loss_gap']:.2f}")
        print(f"  Val R²: {analysis['val_r2']:.4f}")

    return models, trackers, analyses


# Приклад використання:
if __name__ == "__main__":
    # Приклад з синтетичними даними
    print("🧪 Тестування системи відстеження з синтетичними даними...")

    # Створюємо тестові дані
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5

    # Розділяємо на train/val
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Конвертуємо в DataFrame для сумісності
    X_train = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(n_features)])
    X_val = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(n_features)])

    # Запускаємо навчання з відстеженням
    models, trackers, analyses = train_models_with_full_tracking(
        X_train, y_train, X_val, y_val,
        save_plots=True,
        results_dir="test_results"
    )