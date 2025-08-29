import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats

def plot_price_prediction(y_true, y_pred, title='Bitcoin Price Prediction'):
    """
    Візуалізує фактичні та прогнозовані ціни

    Parameters:
    -----------
    y_true : array-like
        Фактичні значення
    y_pred : array-like
        Прогнозовані значення
    title : str
        Заголовок графіка
    """
    try:
        # Перетворюємо на numpy масиви для уникнення проблем з індексами
        if hasattr(y_true, 'values'):
            y_true_arr = y_true.values
        else:
            y_true_arr = np.array(y_true)

        if hasattr(y_pred, 'values'):
            y_pred_arr = y_pred.values
        else:
            y_pred_arr = np.array(y_pred)

        # Перевіряємо, чи маємо достатньо даних
        if len(y_true_arr) == 0 or len(y_pred_arr) == 0:
            print("Недостатньо даних для візуалізації")
            return

        # Обрізаємо до однакової довжини
        min_len = min(len(y_true_arr), len(y_pred_arr))
        y_true_arr = y_true_arr[:min_len]
        y_pred_arr = y_pred_arr[:min_len]

        plt.figure(figsize=(14, 8))

        # Створюємо індекси для осі X
        x_indices = np.arange(len(y_true_arr))

        # Візуалізація фактичних та прогнозованих значень
        plt.plot(x_indices, y_true_arr, label='Фактичні значення', color='blue', linewidth=2)
        plt.plot(x_indices, y_pred_arr, label='Прогнозовані значення', color='red', linestyle='--', linewidth=2,
                 alpha=0.8)

        # Налаштування графіка
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Часовий індекс', fontsize=12)
        plt.ylabel('Ціна (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Додавання підписів відносної похибки (тільки якщо достатньо даних)
        if len(y_true_arr) >= 5:
            n_annotations = min(5, len(y_true_arr))
            step = len(y_true_arr) // n_annotations

            for i in range(n_annotations):
                idx = i * step
                if idx < len(y_true_arr) and y_true_arr[idx] != 0:
                    error_pct = ((y_pred_arr[idx] - y_true_arr[idx]) / y_true_arr[idx] * 100)

                    plt.annotate(f'{error_pct:.1f}%',
                                 xy=(idx, y_pred_arr[idx]),
                                 xytext=(0, 15),
                                 textcoords='offset points',
                                 ha='center',
                                 va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                                 fontsize=10)

        # Додаємо статистику на графік
        # СТАЛО:
        mae = np.mean(np.abs(y_true_arr - y_pred_arr))
        rmse = np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))
        mean_price = np.mean(y_true_arr)
        mae_percent = (mae / mean_price) * 100
        rmse_percent = (rmse / mean_price) * 100
        stats_text = f'MAE: {mae_percent:.2f}%\nRMSE: {rmse_percent:.2f}%'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                 fontsize=11)

        plt.tight_layout()
        plt.show()

        print(f"✓ Візуалізацію створено для {len(y_true_arr)} точок даних")

    except Exception as e:
        print(f"Помилка візуалізації: {e}")
        print("Спрощена візуалізація...")

        # Спрощена версія без анотацій
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_true_arr, label='Фактичні', color='blue')
            plt.plot(y_pred_arr, label='Прогнозовані', color='red', linestyle='--')
            plt.title(title)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception as e2:
            print(f"Не вдалося створити візуалізацію: {e2}")


def plot_feature_importance(feature_names, importances, title='Важливість ознак', n_features=20):
    """
    Візуалізує важливість ознак з відносними частотами (у відсотках)

    Parameters:
    -----------
    feature_names : array-like
        Назви ознак
    importances : array-like
        Значення важливості
    title : str
        Заголовок графіка
    n_features : int
        Кількість ознак для відображення
    """
    try:
        if len(feature_names) == 0 or len(importances) == 0:
            print("Недостатньо даних для візуалізації важливості ознак")
            return

        # Перетворення на numpy масив для зручності обчислень
        importances = np.array(importances)

        # 🔧 НОВИЙ РОЗРАХУНОК: Конвертуємо у відносні частоти (відсотки)
        total_importance = np.sum(importances)
        if total_importance == 0:
            print("Сума важливостей дорівнює нулю")
            return

        # Розраховуємо відносні частоти (частка від загальної суми)
        relative_importances = (importances / total_importance) * 100

        # Перетворення на список кортежів (ознака, відносна важливість)
        features = list(zip(feature_names, relative_importances))

        # Сортування за важливістю
        features.sort(key=lambda x: x[1], reverse=True)

        # Відбір top-n ознак
        top_features = features[:n_features]

        if len(top_features) == 0:
            print("Немає ознак для відображення")
            return

        # Розпакування назад у списки
        top_names, top_relative_importances = zip(*top_features)

        plt.figure(figsize=(12, 8))

        # Створюємо горизонтальний bar chart з кольоровим градієнтом
        y_pos = np.arange(len(top_names))

        # 🔧 НОВИЙ ГРАДІЄНТ: Кольори залежно від важливості
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_relative_importances)))
        bars = plt.barh(y_pos, top_relative_importances, align='center', alpha=0.8,
                        color=colors, edgecolor='navy', linewidth=0.5)

        plt.yticks(y_pos, top_names)
        plt.xlabel('Відносна важливість (%)', fontsize=12)
        plt.title(f'{title} (у відсотках від загальної важливості)', fontsize=14, fontweight='bold')

        # 🔧 ОНОВЛЕНЕ ДОДАВАННЯ ЗНАЧЕНЬ: Показуємо відсотки
        for i, (bar, importance) in enumerate(zip(bars, top_relative_importances)):
            plt.text(bar.get_width() + max(top_relative_importances) * 0.01,
                     bar.get_y() + bar.get_height() / 2,
                     f'{importance:.1f}%', va='center', fontsize=10, fontweight='bold')

        # Додаємо сітку для кращого читання
        plt.grid(axis='x', alpha=0.3)

        # 🔧 НОВА СТАТИСТИКА: Показуємо загальну інформацію
        total_shown = sum(top_relative_importances)
        plt.text(0.02, 0.98,
                 f'Топ-{len(top_features)} ознак\nПокрито: {total_shown:.1f}%\nВсього ознак: {len(feature_names)}',
                 transform=plt.gca().transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=10)

        plt.tight_layout()
        plt.show()

        # 🔧 ОНОВЛЕНЕ ПОВІДОМЛЕННЯ: Показуємо статистику
        print(f"✓ Візуалізацію важливості створено для {len(top_features)} ознак")
        print(f"📊 Топ-{len(top_features)} ознак покривають {total_shown:.1f}% загальної важливості")
        print(f"📊 Найважливіша ознака: '{top_names[0]}' ({top_relative_importances[0]:.1f}%)")

        # Показуємо топ-5 ознак у консолі
        print("\n🏆 Топ-5 найважливіших ознак:")
        for i, (name, importance) in enumerate(top_features[:5], 1):
            print(f"  {i}. {name}: {importance:.1f}%")

    except Exception as e:
        print(f"Помилка візуалізації важливості ознак: {e}")
        import traceback
        print(traceback.format_exc())


def plot_learning_curves(train_losses, val_losses, train_accuracies=None, val_accuracies=None, title='Learning Curves'):
    """
    Візуалізує криві навчання: train/validation loss та accuracy

    Parameters:
    -----------
    train_losses : array-like
        Значення loss на тренувальних даних по епохах
    val_losses : array-like
        Значення loss на валідаційних даних по епохах
    train_accuracies : array-like, optional
        Accuracy на тренувальних даних
    val_accuracies : array-like, optional
        Accuracy на валідаційних даних
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        epochs = range(1, len(train_losses) + 1)

        # Графік Loss
        axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Графік Accuracy (якщо є)
        if train_accuracies is not None and val_accuracies is not None:
            axes[1].plot(epochs, train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
            axes[1].plot(epochs, val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
            axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            # Якщо немає accuracy, показуємо розподіл похибок
            errors = np.array(train_losses) - np.array(val_losses)
            axes[1].plot(epochs, errors, 'g-', label='Train-Val Loss Difference', linewidth=2)
            axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.7)
            axes[1].set_title('Train-Validation Loss Difference', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss Difference')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        print("✅ Криві навчання створено")

    except Exception as e:
        print(f"❌ Помилка створення кривих навчання: {e}")


def plot_model_comparison_metrics(model_metrics_dict, title='Model Performance Comparison'):
    """
    Порівняння метрик різних моделей

    Parameters:
    -----------
    model_metrics_dict : dict
        Словник {model_name: {'MAE': value, 'RMSE': value, 'R2': value, ...}}
    """
    try:
        if not model_metrics_dict:
            print("Немає даних для порівняння моделей")
            return

        models = list(model_metrics_dict.keys())
        metrics = list(next(iter(model_metrics_dict.values())).keys())

        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))

        if n_metrics == 1:
            axes = [axes]

        for i, metric in enumerate(metrics):
            values = [model_metrics_dict[model].get(metric, 0) for model in models]

            bars = axes[i].bar(models, values, alpha=0.8,
                               color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(models)])
            axes[i].set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)

            # Додаємо значення на стовпчики
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., height,
                             f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("✅ Порівняння моделей створено")

    except Exception as e:
        print(f"❌ Помилка порівняння моделей: {e}")


def plot_residuals_analysis(y_true, y_pred, title='Residuals Analysis'):
    """
    Аналіз залишків моделі
    """
    try:
        residuals = y_pred - y_true

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Residuals vs Fitted Values
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted Values')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Histogram of Residuals
        axes[0, 1].hist(residuals, bins=7, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Q-Q Plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Residuals vs Order
        axes[1, 1].plot(range(len(residuals)), residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_xlabel('Observation Order')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Order')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("✅ Аналіз залишків створено")

    except Exception as e:
        print(f"❌ Помилка аналізу залишків: {e}")


def plot_prediction_confidence_intervals(y_true, y_pred, confidence_intervals=None, title='Prediction Confidence'):
    """
    Візуалізує прогнози з довірчими інтервалами

    Parameters:
    -----------
    y_true : array-like
        Фактичні значення
    y_pred : array-like
        Прогнозовані значення
    confidence_intervals : tuple of arrays, optional
        (lower_bound, upper_bound) для довірчих інтервалів
    """
    try:
        plt.figure(figsize=(14, 8))

        x_indices = np.arange(len(y_true))

        # Основні лінії
        plt.plot(x_indices, y_true, label='Фактичні значення', color='blue', linewidth=2)
        plt.plot(x_indices, y_pred, label='Прогнозовані значення', color='red', linewidth=2, alpha=0.8)

        # Довірчі інтервали (якщо є)
        if confidence_intervals is not None:
            lower_bound, upper_bound = confidence_intervals
            plt.fill_between(x_indices, lower_bound, upper_bound,
                             alpha=0.3, color='gray', label='Довірчий інтервал')

        # Підрахунок точок в межах інтервалу
        if confidence_intervals is not None:
            lower_bound, upper_bound = confidence_intervals
            within_interval = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
            coverage = within_interval / len(y_true) * 100

            plt.text(0.02, 0.98, f'Coverage: {coverage:.1f}%',
                     transform=plt.gca().transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Час (індекс)')
        plt.ylabel('Ціна (USD)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        print("✅ Довірчі інтервали створено")

    except Exception as e:
        print(f"❌ Помилка створення довірчих інтервалів: {e}")


def plot_feature_correlation_heatmap(X, feature_names=None, title='Feature Correlation Heatmap'):
    """
    Теплова карта кореляцій між ознаками
    """
    try:
        if hasattr(X, 'corr'):
            correlation_matrix = X.corr()
        else:
            correlation_matrix = pd.DataFrame(X).corr()

        plt.figure(figsize=(12, 10))

        # Використовуємо seaborn для красивої теплової карти
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

        sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                    square=True, fmt='.2f', cbar_kws={"shrink": .8})

        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("✅ Теплову карту кореляцій створено")

    except Exception as e:
        print(f"❌ Помилка створення теплової карти: {e}")

def create_summary_plots(y_true, y_pred, feature_importance_dict=None):
    """
    Створює комплексну візуалізацію результатів
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Перетворюємо на numpy масиви
        if hasattr(y_true, 'values'):
            y_true_arr = y_true.values
        else:
            y_true_arr = np.array(y_true)

        if hasattr(y_pred, 'values'):
            y_pred_arr = y_pred.values
        else:
            y_pred_arr = np.array(y_pred)

        # Графік 1: Часовий ряд
        axes[0, 0].plot(y_true_arr, label='Фактичні', color='blue')
        axes[0, 0].plot(y_pred_arr, label='Прогнозовані', color='red', linestyle='--')
        axes[0, 0].set_title('Прогнозування в часі')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Графік 2: Scatter plot
        axes[0, 1].scatter(y_true_arr, y_pred_arr, alpha=0.6)
        min_val, max_val = min(y_true_arr.min(), y_pred_arr.min()), max(y_true_arr.max(), y_pred_arr.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 1].set_xlabel('Фактичні значення')
        axes[0, 1].set_ylabel('Прогнозовані значення')
        axes[0, 1].set_title('Співвідношення прогнозів')
        axes[0, 1].grid(True, alpha=0.3)

        # Графік 3: Розподіл помилок
        errors = y_pred_arr - y_true_arr
        axes[1, 0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Помилка прогнозування')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].set_title('Розподіл помилок')
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        axes[1, 0].grid(True, alpha=0.3)

        # Графік 4: Важливість ознак (якщо доступна)
        if feature_importance_dict and len(feature_importance_dict) > 0:
            # Топ-10 ознак
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)

            y_pos = np.arange(len(features))
            axes[1, 1].barh(y_pos, importances)
            axes[1, 1].set_yticks(y_pos)
            axes[1, 1].set_yticklabels(features)
            axes[1, 1].set_xlabel('Важливість')
            axes[1, 1].set_title('Топ-10 важливих ознак')
        else:
            axes[1, 1].text(0.5, 0.5, 'Важливість ознак\nнедоступна',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Важливість ознак')

        plt.tight_layout()
        plt.show()

        print("✓ Комплексну візуалізацію створено")

    except Exception as e:
        print(f"Помилка створення комплексної візуалізації: {e}")

def plot_ensemble_weights_visualization(ensemble_weights, title='Ensemble Model Weights'):
    """
    Візуалізація ваг в ансамблевій моделі
    """
    try:
        if not ensemble_weights:
            print("Немає даних про ваги ансамблю")
            return

        models = list(ensemble_weights.keys())
        weights = list(ensemble_weights.values())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar chart
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink'][:len(models)]
        bars = ax1.bar(models, weights, color=colors, alpha=0.8, edgecolor='black')
        ax1.set_title('Model Weights in Ensemble', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Weight')
        ax1.tick_params(axis='x', rotation=45)

        # Додаємо відсотки на стовпчики
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{weight:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Pie chart
        ax2.pie(weights, labels=models, autopct='%1.1f%%', startangle=90, colors=colors)
        ax2.set_title('Model Weight Distribution', fontsize=14, fontweight='bold')

        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("✅ Візуалізацію ваг ансамблю створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації ваг ансамблю: {e}")


def plot_cross_validation_scores(cv_scores, model_names=None, title='Cross-Validation Scores'):
    """
    Візуалізація результатів крос-валідації
    """
    try:
        if model_names is None:
            model_names = [f'Model_{i + 1}' for i in range(len(cv_scores))]

        plt.figure(figsize=(12, 8))

        # Box plot для кожної моделі
        box_data = []
        positions = []
        labels = []

        for i, (model_name, scores) in enumerate(zip(model_names, cv_scores)):
            if isinstance(scores, (list, np.ndarray)) and len(scores) > 1:
                box_data.append(scores)
                positions.append(i + 1)
                labels.append(model_name)

        if box_data:
            bp = plt.boxplot(box_data, positions=positions, labels=labels, patch_artist=True)

            colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink']
            for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Додаємо середні значення
            for i, scores in enumerate(box_data):
                mean_score = np.mean(scores)
                plt.text(positions[i], mean_score, f'{mean_score:.3f}',
                         ha='center', va='bottom', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('Score')
        plt.xlabel('Model')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        print("✅ Результати крос-валідації створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації крос-валідації: {e}")


def create_comprehensive_model_dashboard(y_true, y_pred, model_metrics, feature_importance=None,
                                         ensemble_weights=None, residuals=None):
    """
    🆕 Комплексний дашборд для оцінки моделі
    """
    try:
        fig = plt.figure(figsize=(20, 15))

        # 1. Actual vs Predicted
        ax1 = plt.subplot(3, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')

        # 2. Time Series Plot
        ax2 = plt.subplot(3, 3, 2)
        plt.plot(y_true, label='Actual', linewidth=2)
        plt.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8)
        plt.legend()
        plt.title('Time Series Comparison')

        # 3. Residuals
        ax3 = plt.subplot(3, 3, 3)
        if residuals is None:
            residuals = y_pred - y_true
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')

        # 4. Metrics Bar Chart
        ax4 = plt.subplot(3, 3, 4)
        if model_metrics:
            metrics_names = list(model_metrics.keys())[:6]  # Перші 6 метрик
            metrics_values = [model_metrics[name] for name in metrics_names]
            plt.bar(metrics_names, metrics_values, alpha=0.8)
            plt.title('Model Metrics')
            plt.xticks(rotation=45)

        # 5. Feature Importance (якщо є)
        ax5 = plt.subplot(3, 3, 5)
        if feature_importance:
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            features, importances = zip(*sorted_features)
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.title('Top 10 Feature Importance')

        # 6. Ensemble Weights (якщо є)
        ax6 = plt.subplot(3, 3, 6)
        if ensemble_weights:
            models = list(ensemble_weights.keys())
            weights = list(ensemble_weights.values())
            plt.pie(weights, labels=models, autopct='%1.1f%%')
            plt.title('Ensemble Weights')

        # 7. Error Distribution
        ax7 = plt.subplot(3, 3, 7)
        errors = np.abs(y_pred - y_true)
        plt.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution')

        # 8. Percentage Error
        ax8 = plt.subplot(3, 3, 8)
        pct_errors = (y_pred - y_true) / y_true * 100
        plt.hist(pct_errors, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Percentage Error (%)')
        plt.ylabel('Frequency')
        plt.title('Percentage Error Distribution')

        # 9. Cumulative Error
        ax9 = plt.subplot(3, 3, 9)
        cumulative_error = np.cumsum(np.abs(y_pred - y_true))
        plt.plot(cumulative_error)
        plt.xlabel('Sample Index')
        plt.ylabel('Cumulative Absolute Error')
        plt.title('Cumulative Error Growth')

        plt.suptitle('Comprehensive Model Performance Dashboard', fontsize=20, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("✅ Комплексний дашборд створено")

    except Exception as e:
        print(f"❌ Помилка створення дашборду: {e}")