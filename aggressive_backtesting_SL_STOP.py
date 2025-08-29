# aggressive_backtesting_SL.py - Повний оновлений модуль бектестингу

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import traceback

# Імпорт CONFIG з обробкою помилок
try:
    from config import CONFIG
except ImportError:
    print("УВАГА: Не вдалося імпортувати CONFIG. Використовується базовий конфіг.")
    CONFIG = {
        'trading': {'initial_balance': 100000},
        'saving': {'results_dir': 'results'}
    }

# ДОДАЙТЕ ЦЮ НОВУ ФУНКЦІЮ в aggressive_backtesting_SL_STOP.py

import re
from datetime import datetime
try:
    from scipy import stats
    from scipy.stats import jarque_bera, normaltest, shapiro
    SCIPY_AVAILABLE = True
    print("✅ SciPy доступний для статистичного аналізу")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy недоступний, статистичні тести будуть пропущені")
# Глобальна змінна для збереження PnL з логів
TRADING_LOG_PNL = []


def capture_pnl_from_log(message):
    """📝 Перехоплює PnL з повідомлень логу"""
    global TRADING_LOG_PNL

    try:
        # Парсимо різні типи PnL повідомлень з логів
        patterns = [
            # ШТУЧНИЙ ШОРТ PnL: $+222 (+3.94%)
            r'ШТУЧНИЙ ШОРТ PnL:\s*\$([+-]?\d+(?:\.\d+)?)',
            # PnL: $+487 (+3.89%)
            r'PnL:\s*\$([+-]?\d+(?:\.\d+)?)',
            # Для інших форматів
            r'прибуток:\s*\$([+-]?\d+(?:\.\d+)?)',
            r'збиток:\s*\$([+-]?\d+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                pnl_value = float(match.group(1))
                timestamp = datetime.now()

                # Визначаємо тип трейду з повідомлення
                trade_type = "UNKNOWN"
                if "SHORT" in message:
                    if "TAKE_PROFIT" in message or "TP:" in message:
                        trade_type = "SHORT_TP"
                    elif "STOP_LOSS" in message or "SL:" in message:
                        trade_type = "SHORT_SL"
                    elif "TIMEOUT" in message:
                        trade_type = "SHORT_TIMEOUT"
                elif "LONG" in message or "PnL:" in message:
                    if "TAKE_PROFIT" in message or "TP:" in message:
                        trade_type = "LONG_TP"
                    elif "STOP_LOSS" in message or "SL:" in message:
                        trade_type = "LONG_SL"
                    elif "TIMEOUT" in message:
                        trade_type = "LONG_TIMEOUT"

                TRADING_LOG_PNL.append({
                    'timestamp': timestamp,
                    'pnl': pnl_value,
                    'type': trade_type,
                    'message': message.strip()
                })

                print(f"📝 Захоплено PnL: ${pnl_value:+.0f} ({trade_type})")
                return True

    except Exception as e:
        pass  # Просто ігноруємо помилки парсингу

    return False


def get_cumulative_pnl_from_logs():
    """📊 Повертає накопичену дохідність з захоплених логів"""
    global TRADING_LOG_PNL

    if not TRADING_LOG_PNL:
        print("⚠️ Немає захоплених PnL з логів")
        return []

    print(f"📊 Використовуємо {len(TRADING_LOG_PNL)} PnL записів з логів:")

    cumulative_pnl = []
    running_total = 0.0

    for i, log_entry in enumerate(TRADING_LOG_PNL):
        running_total += log_entry['pnl']
        cumulative_pnl.append(running_total)

        print(f"  {i + 1}. {log_entry['type']}: ${log_entry['pnl']:+.0f} → Накопичено: ${running_total:+.0f}")

    print(f"📊 Фінальна дохідність з логів: ${running_total:+.0f}")
    return cumulative_pnl


def reset_trading_log():
    """🔄 Скидає лог PnL (викликати перед новим бектестом)"""
    global TRADING_LOG_PNL
    TRADING_LOG_PNL = []
    print("🔄 Лог PnL скинуто")


def calculate_advanced_trading_statistics_with_kurtosis(backtest_results):
    """📊 РОЗШИРЕНА статистика з коефіцієнтом ексцесу та детальним аналізом розподілу"""

    if backtest_results.empty:
        print("❌ Порожній DataFrame для аналізу")
        return {}

    print("\n" + "📊" * 50)
    print("РОЗШИРЕНИЙ СТАТИСТИЧНИЙ АНАЛІЗ З КОЕФІЦІЄНТОМ ЕКСЦЕСУ")
    print("📊" * 50)

    # Шукаємо дані з PnL
    executed_trades = backtest_results[backtest_results['executed'] == True]

    all_pnl_records = backtest_results[
        (backtest_results['pnl'].notna()) &
        (backtest_results['pnl'] != 0)
        ]

    if len(all_pnl_records) < 5:
        print("❌ Недостатньо PnL записів для статистичного аналізу")
        return {}

    pnl_array = all_pnl_records['pnl'].values
    print(f"📈 Аналізуємо {len(pnl_array)} PnL записів")

    # Основні статистики
    mean_pnl = np.mean(pnl_array)
    std_pnl = np.std(pnl_array)
    median_pnl = np.median(pnl_array)

    # ===== НОВИЙ РОЗДІЛ: КОЕФІЦІЄНТ ЕКСЦЕСУ =====
    print(f"\n🔬 АНАЛІЗ ФОРМИ РОЗПОДІЛУ PnL:")

    # Розрахунок ексцесу
    try:
        from scipy.stats import kurtosis, skew

        # Ексцес (excess kurtosis = kurtosis - 3)
        excess_kurtosis = kurtosis(pnl_array, fisher=True)  # fisher=True дає excess kurtosis
        raw_kurtosis = kurtosis(pnl_array, fisher=False)  # fisher=False дає raw kurtosis

        # Асиметрія
        skewness = skew(pnl_array)

        print(f"   📊 Ексцес (excess kurtosis): {excess_kurtosis:.4f}")
        print(f"   📊 Сирий ексцес (raw kurtosis): {raw_kurtosis:.4f}")
        print(f"   📊 Асиметрія (skewness): {skewness:.4f}")

        # Інтерпретація ексцесу
        print(f"\n🧠 ІНТЕРПРЕТАЦІЯ ФОРМИ РОЗПОДІЛУ:")

        if excess_kurtosis > 0.5:
            kurtosis_type = "Лептокуртичний (важкі хвости)"
            kurtosis_meaning = "Висока ймовірність екстремальних подій"
            risk_assessment = "ПІДВИЩЕНИЙ ризик великих прибутків і збитків"
            strategy_stability = "НЕСТАБІЛЬНА (схильна до екстремумів)"
            color_indicator = "🔴"
        elif excess_kurtosis < -0.5:
            kurtosis_type = "Платикуртичний (легкі хвости)"
            kurtosis_meaning = "Низька ймовірність екстремальних подій"
            risk_assessment = "ЗНИЖЕНИЙ ризик великих коливань"
            strategy_stability = "СТАБІЛЬНА (менше екстремумів)"
            color_indicator = "🟢"
        else:
            kurtosis_type = "Мезокуртичний (нормальний)"
            kurtosis_meaning = "Помірна ймовірність екстремальних подій"
            risk_assessment = "ПОМІРНИЙ ризик"
            strategy_stability = "ЗБАЛАНСОВАНА"
            color_indicator = "🟡"

        print(f"   {color_indicator} Тип розподілу: {kurtosis_type}")
        print(f"   {color_indicator} Значення: {kurtosis_meaning}")
        print(f"   {color_indicator} Ризик-оцінка: {risk_assessment}")
        print(f"   {color_indicator} Стабільність стратегії: {strategy_stability}")

        # Аналіз асиметрії
        print(f"\n📐 АНАЛІЗ АСИМЕТРІЇ:")
        if skewness > 0.5:
            skew_interpretation = "Позитивна асиметрія - більше малих збитків, рідкісні великі прибутки"
            skew_color = "🟢"
        elif skewness < -0.5:
            skew_interpretation = "Негативна асиметрія - більше малих прибутків, рідкісні великі збитки"
            skew_color = "🔴"
        else:
            skew_interpretation = "Симетричний розподіл - збалансовані прибутки та збитки"
            skew_color = "🟡"

        print(f"   {skew_color} {skew_interpretation}")

        # Комбінований аналіз ексцесу та асиметрії
        print(f"\n🎯 КОМБІНОВАНИЙ АНАЛІЗ РИЗИКУ:")

        if excess_kurtosis > 0 and skewness < -0.3:
            combined_risk = "🔴 ВИСОКИЙ РИЗИК: Важкі хвости + негативна асиметрія = ризик катастрофічних збитків"
        elif excess_kurtosis > 0 and skewness > 0.3:
            combined_risk = "🟡 СПЕКУЛЯТИВНИЙ: Важкі хвости + позитивна асиметрія = можливість великих прибутків з ризиком"
        elif excess_kurtosis < 0 and abs(skewness) < 0.3:
            combined_risk = "🟢 НИЗЬКИЙ РИЗИК: Легкі хвости + симетрія = стабільна стратегія"
        elif excess_kurtosis < 0 and skewness > 0:
            combined_risk = "🟢 ОПТИМАЛЬНИЙ: Легкі хвости + позитивна асиметрія = стабільні прибутки"
        else:
            combined_risk = "🟡 ПОМІРНИЙ РИЗИК: Збалансовані характеристики розподілу"

        print(f"   {combined_risk}")

    except ImportError:
        print("⚠️ SciPy недоступний - пропускаємо аналіз ексцесу")
        excess_kurtosis = None
        skewness = None

    # Продовжуємо з основною статистикою...
    positive_pnl = pnl_array[pnl_array > 0]
    negative_pnl = pnl_array[pnl_array < 0]

    win_rate = len(positive_pnl) / len(pnl_array) if len(pnl_array) > 0 else 0
    avg_win = np.mean(positive_pnl) if len(positive_pnl) > 0 else 0
    avg_loss = np.mean(negative_pnl) if len(negative_pnl) > 0 else 0

    profit_factor = abs(np.sum(positive_pnl) / np.sum(negative_pnl)) if np.sum(negative_pnl) != 0 else np.inf

    # Розрахунок максимальної просадки
    cumulative_pnl = np.cumsum(pnl_array)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown = np.min(drawdown)

    # Створюємо словник результатів
    stats_dict = {
        'total_trades': len(pnl_array),
        'mean_pnl': mean_pnl,
        'median_pnl': median_pnl,
        'std_pnl': std_pnl,
        'total_pnl': np.sum(pnl_array),
        'win_rate': win_rate * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'excess_kurtosis': excess_kurtosis,
        'skewness': skewness,
        'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,
    }

    # ===== НОВИЙ РОЗДІЛ: ОЦІНКА СТРАТЕГІЇ НА ОСНОВІ ЕКСЦЕСУ =====
    print(f"\n🎯 ОЦІНКА СТРАТЕГІЇ НА ОСНОВІ ФОРМИ РОЗПОДІЛУ:")

    strategy_score = 0

    # Базова оцінка
    if win_rate > 0.6:
        strategy_score += 2
    elif win_rate > 0.5:
        strategy_score += 1

    if profit_factor > 1.5:
        strategy_score += 2
    elif profit_factor > 1.0:
        strategy_score += 1

    # Бонуси/штрафи за форму розподілу
    if excess_kurtosis is not None:
        if excess_kurtosis < -0.3:  # Платикуртичний
            strategy_score += 1
            print(f"   ✅ БОНУС: Стабільний розподіл (excess kurtosis < 0)")
        elif excess_kurtosis > 1.0:  # Дуже лептокуртичний
            strategy_score -= 1
            print(f"   ⚠️ ШТРАФ: Нестабільний розподіл (excess kurtosis > 1)")

        if skewness > 0.3:  # Позитивна асиметрія
            strategy_score += 1
            print(f"   ✅ БОНУС: Позитивна асиметрія (більше схильність до прибутків)")
        elif skewness < -0.5:  # Негативна асиметрія
            strategy_score -= 1
            print(f"   ⚠️ ШТРАФ: Негативна асиметрія (схильність до великих збитків)")

    # Фінальна оцінка
    if strategy_score >= 5:
        final_verdict = "🏆 ВІДМІННА СТРАТЕГІЯ (стабільна і прибуткова)"
    elif strategy_score >= 3:
        final_verdict = "🟢 ХОРОША СТРАТЕГІЯ (рекомендована)"
    elif strategy_score >= 1:
        final_verdict = "🟡 ЗАДОВІЛЬНА СТРАТЕГІЯ (потребує оптимізації)"
    else:
        final_verdict = "🔴 НЕЗАДОВІЛЬНА СТРАТЕГІЯ (не рекомендована)"

    print(f"\n🏅 ФІНАЛЬНА ОЦІНКА: {final_verdict}")
    print(f"   📊 Загальний бал: {strategy_score}/6")

    # Додаємо рекомендації
    print(f"\n💡 РЕКОМЕНДАЦІЇ НА ОСНОВІ АНАЛІЗУ РОЗПОДІЛУ:")

    if excess_kurtosis is not None and excess_kurtosis > 0.5:
        print(f"   🔧 Розгляньте додавання стоп-лосів для захисту від екстремальних збитків")
        print(f"   🔧 Зменшіть розмір позицій в умовах високої волатільності")

    if excess_kurtosis is not None and excess_kurtosis < -0.3:
        print(f"   🚀 Можна розглянути збільшення розміру позицій (низький tail risk)")
        print(f"   🚀 Стратегія підходить для консервативних інвесторів")

    if skewness is not None and skewness < -0.3:
        print(f"   ⚖️ Розгляньте хеджування проти великих збитків")
        print(f"   ⚖️ Аналізуйте умови, що призводять до негативних викидів")

    # ===== НОВИЙ РОЗДІЛ: ВІЗУАЛІЗАЦІЯ РОЗПОДІЛУ =====
    print(f"\n📈 СТВОРЕННЯ ГРАФІКА РОЗПОДІЛУ PnL...")

    try:
        import matplotlib.pyplot as plt
        from scipy.stats import norm

        # Створюємо фігуру з субплотами
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # График 1: Гістограма розподілу PnL з кривою нормального розподілу
        ax1.hist(pnl_array, bins=min(30, len(pnl_array) // 3), alpha=0.7, color='skyblue',
                 edgecolor='black', density=True, label='Фактичний розподіл PnL')

        # Додаємо криву нормального розподілу для порівняння
        x_norm = np.linspace(pnl_array.min(), pnl_array.max(), 100)
        y_norm = norm.pdf(x_norm, mean_pnl, std_pnl)
        ax1.plot(x_norm, y_norm, 'r-', linewidth=2, label='Нормальний розподіл')

        # Вертикальні лінії для ключових статистик
        ax1.axvline(mean_pnl, color='green', linestyle='--', linewidth=2, label=f'Середнє: ${mean_pnl:.0f}')
        ax1.axvline(median_pnl, color='orange', linestyle='--', linewidth=2, label=f'Медіана: ${median_pnl:.0f}')
        ax1.axvline(0, color='black', linestyle='-', alpha=0.5, label='Беззбитковість')

        ax1.set_title('Розподіл PnL vs Нормальний розподіл', fontsize=14, fontweight='bold')
        ax1.set_xlabel('PnL (USD)', fontsize=12)
        ax1.set_ylabel('Щільність', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Додаємо текст з основними статистиками
        stats_text = f'Ексцес: {excess_kurtosis:.3f}\nАсиметрія: {skewness:.3f}\n'
        stats_text += f'Std: ${std_pnl:.0f}\nN: {len(pnl_array)}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8), fontsize=10)

        # График 2: Q-Q Plot для перевірки нормальності
        from scipy import stats as scipy_stats
        scipy_stats.probplot(pnl_array, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Перевірка нормальності)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Додаємо лінію ідеальної нормальності
        ax2.get_lines()[0].set_markerfacecolor('skyblue')
        ax2.get_lines()[0].set_markeredgecolor('navy')
        ax2.get_lines()[0].set_markersize(6)
        ax2.get_lines()[1].set_color('red')
        ax2.get_lines()[1].set_linewidth(2)

        # График 3: Box Plot з додатковими статистиками
        bp = ax3.boxplot(pnl_array, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(color='black', linewidth=1.5),
                         capprops=dict(color='black', linewidth=1.5),
                         flierprops=dict(marker='o', markerfacecolor='red', markersize=8, alpha=0.7))

        ax3.set_title('Box Plot: Викиди та квартилі', fontsize=14, fontweight='bold')
        ax3.set_ylabel('PnL (USD)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)

        # Додаємо статистики квартилів
        q1, q3 = np.percentile(pnl_array, [25, 75])
        iqr = q3 - q1
        whisker_low = q1 - 1.5 * iqr
        whisker_high = q3 + 1.5 * iqr
        outliers_count = len(pnl_array[(pnl_array < whisker_low) | (pnl_array > whisker_high)])

        quartile_text = f'Q1: ${q1:.0f}\nQ3: ${q3:.0f}\nIQR: ${iqr:.0f}\nВикиди: {outliers_count}'
        ax3.text(0.02, 0.98, quartile_text, transform=ax3.transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8), fontsize=10)

        # График 4: Кумулятивна крива PnL з зонами ризику
        cumulative_pnl = np.cumsum(pnl_array)
        trade_numbers = np.arange(1, len(pnl_array) + 1)

        ax4.plot(trade_numbers, cumulative_pnl, linewidth=2, color='darkblue', label='Кумулятивний PnL')
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5, label='Беззбитковість')

        # Додаємо зони ризику на основі стандартних відхилень
        running_mean = np.cumsum(pnl_array) / trade_numbers
        running_std = np.array([np.std(pnl_array[:i + 1]) * np.sqrt(i + 1) for i in range(len(pnl_array))])

        ax4.fill_between(trade_numbers, running_mean - running_std, running_mean + running_std,
                         alpha=0.2, color='gray', label='±1σ зона')
        ax4.fill_between(trade_numbers, running_mean - 2 * running_std, running_mean + 2 * running_std,
                         alpha=0.1, color='red', label='±2σ зона')

        # Відмічаємо найбільшу просадку
        max_dd_idx = np.argmin(drawdown)
        ax4.scatter(max_dd_idx + 1, cumulative_pnl[max_dd_idx], color='red', s=100,
                    zorder=5, label=f'Max DD: ${max_drawdown:.0f}')

        ax4.set_title('Кумулятивний PnL з зонами ризику', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Номер трейду', fontsize=12)
        ax4.set_ylabel('Кумулятивний PnL (USD)', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)

        # Загальний заголовок з ключовими метриками
        title_text = f'Аналіз розподілу PnL: {kurtosis_type if "kurtosis_type" in locals() else "Unknown"}'
        title_text += f'\nЗагальний PnL: ${np.sum(pnl_array):+.0f} | '
        title_text += f'Win Rate: {win_rate * 100:.1f}% | '
        title_text += f'Profit Factor: {profit_factor:.2f}'

        plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98)

        # Колірне кодування заголовку на основі якості стратегії
        if strategy_score >= 5:
            title_color = 'green'
        elif strategy_score >= 3:
            title_color = 'orange'
        else:
            title_color = 'red'

        plt.suptitle(title_text, fontsize=16, fontweight='bold', y=0.98, color=title_color)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

        print(f"✅ Графік розподілу PnL створено успішно")

        # Додаткова інтерпретація графіків
        print(f"\n📊 ІНТЕРПРЕТАЦІЯ ГРАФІКІВ:")
        print(f"   📈 Гістограма: Показує форму розподілу порівняно з нормальним")
        print(f"   📉 Q-Q Plot: Точки на лінії = нормальний розподіл, відхилення = аномалії")
        print(f"   📦 Box Plot: Медіана, квартилі та викиди ({outliers_count} викидів)")
        print(f"   📊 Кумулятивний: Еволюція стратегії в часі з зонами ризику")

        # Додаткові висновки на основі візуального аналізу
        outlier_percentage = (outliers_count / len(pnl_array)) * 100
        if outlier_percentage > 10:
            print(f"\n⚠️ УВАГА: Високий відсоток викидів ({outlier_percentage:.1f}%) - нестабільна стратегія")
        elif outlier_percentage < 2:
            print(f"\n✅ ДОБРЕ: Низький відсоток викидів ({outlier_percentage:.1f}%) - стабільна стратегія")

    except ImportError as e:
        print(f"⚠️ Не вдалося створити графіки: {e}")
        print("   Переконайтеся, що встановлені matplotlib та scipy")
    except Exception as e:
        print(f"❌ Помилка створення графіків: {e}")

    stats_dict.update({
        'strategy_score': strategy_score,
        'final_verdict': final_verdict,
        'kurtosis_type': kurtosis_type if 'kurtosis_type' in locals() else 'Unknown',
        'risk_assessment': risk_assessment if 'risk_assessment' in locals() else 'Unknown',
        'outliers_count': outliers_count if 'outliers_count' in locals() else 0,
        'outlier_percentage': outlier_percentage if 'outlier_percentage' in locals() else 0
    })

    return stats_dict
def calculate_advanced_trading_statistics(backtest_results):
    """📊 ВИПРАВЛЕНА розширена статистика з математичним аналізом"""

    if backtest_results.empty:
        print("❌ Порожній DataFrame для аналізу")
        return {}

    print("\n" + "📊" * 50)
    print("РОЗШИРЕНИЙ СТАТИСТИЧНИЙ АНАЛІЗ СТРАТЕГІЇ")
    print("📊" * 50)

    # 1. ДІАГНОСТИКА ДАНИХ
    print(f"🔍 ДІАГНОСТИКА ДАНИХ:")
    print(f"   Загальна кількість записів: {len(backtest_results)}")

    executed_trades = backtest_results[backtest_results['executed'] == True]
    print(f"   Виконаних трейдів: {len(executed_trades)}")

    # Перевіряємо наявність колонки PnL
    if 'pnl' not in backtest_results.columns:
        print("❌ Колонка 'pnl' відсутня в даних")
        print(f"📊 Доступні колонки: {list(backtest_results.columns)}")
        return {}

    # Шукаємо всі записи з PnL (не тільки executed)
    all_pnl_records = backtest_results[
        (backtest_results['pnl'].notna()) &
        (backtest_results['pnl'] != 0)
        ]
    print(f"   Записів з PnL: {len(all_pnl_records)}")

    # Спробуємо різні варіанти фільтрації
    executed_with_pnl = executed_trades[
        (executed_trades['pnl'].notna()) &
        (executed_trades['pnl'] != 0)
        ]
    print(f"   Виконаних трейдів з PnL: {len(executed_with_pnl)}")

    # Шукаємо записи з close_reason (закриті позиції)
    if 'close_reason' in backtest_results.columns:
        closed_positions = backtest_results[
            (backtest_results['close_reason'].notna()) &
            (backtest_results['pnl'].notna()) &
            (backtest_results['pnl'] != 0)
            ]
        print(f"   Закритих позицій з PnL: {len(closed_positions)}")
    else:
        closed_positions = pd.DataFrame()
        print("   Колонка 'close_reason' відсутня")

    # Шукаємо записи з signal_type що містить 'close'
    if 'signal_type' in backtest_results.columns:
        close_signal_types = backtest_results[
            (backtest_results['signal_type'].str.contains('close', na=False)) &
            (backtest_results['pnl'].notna()) &
            (backtest_results['pnl'] != 0)
            ]
        print(f"   Сигналів закриття з PnL: {len(close_signal_types)}")
    else:
        close_signal_types = pd.DataFrame()
        print("   Колонка 'signal_type' відсутня")

    # Вибираємо найкращий набір даних для аналізу
    datasets_to_try = [
        ("Закриті позиції", closed_positions),
        ("Сигнали закриття", close_signal_types),
        ("Виконані трейди з PnL", executed_with_pnl),
        ("Всі записи з PnL", all_pnl_records)
    ]

    analysis_data = None
    data_source = ""

    for name, dataset in datasets_to_try:
        if len(dataset) >= 3:  # Мінімум 3 записи для статистики
            analysis_data = dataset
            data_source = name
            print(f"✅ Використовуємо для аналізу: {name} ({len(dataset)} записів)")
            break
        else:
            print(f"⚠️ {name}: недостатньо даних ({len(dataset)} записів)")

    if analysis_data is None or len(analysis_data) == 0:
        print("❌ Немає достатніх даних для розширеного аналізу")
        print("\n🔧 РЕКОМЕНДАЦІЇ:")
        print("   1. Перевірте, чи правильно записується PnL при закритті позицій")
        print("   2. Переконайтеся, що функція capture_pnl_from_log() викликається")
        print("   3. Перевірте логіку закриття позицій в trading system")
        return {}

    # 2. ОСНОВНА СТАТИСТИКА
    pnl_array = analysis_data['pnl'].values
    print(f"\n📈 БАЗОВИЙ АНАЛІЗ ({data_source}):")
    print(f"   Записів для аналізу: {len(pnl_array)}")

    # Перевіряємо розподіл PnL
    positive_pnl = pnl_array[pnl_array > 0]
    negative_pnl = pnl_array[pnl_array < 0]
    zero_pnl = pnl_array[pnl_array == 0]

    print(f"   Позитивних PnL: {len(positive_pnl)}")
    print(f"   Негативних PnL: {len(negative_pnl)}")
    print(f"   Нульових PnL: {len(zero_pnl)}")

    if len(positive_pnl) > 0:
        print(f"   Діапазон прибутків: ${np.min(positive_pnl):.2f} - ${np.max(positive_pnl):.2f}")
    if len(negative_pnl) > 0:
        print(f"   Діапазон збитків: ${np.min(negative_pnl):.2f} - ${np.max(negative_pnl):.2f}")

    # 3. РОЗРАХУНОК ОСНОВНИХ МЕТРИК
    stats_dict = {
        'data_source': data_source,
        'total_trades': len(pnl_array),
        'mean_pnl': np.mean(pnl_array),
        'median_pnl': np.median(pnl_array),
        'std_pnl': np.std(pnl_array),
        'min_pnl': np.min(pnl_array),
        'max_pnl': np.max(pnl_array),
        'total_pnl': np.sum(pnl_array),
    }

    # 4. ПОКАЗНИКИ РИЗИКУ
    win_rate = len(positive_pnl) / len(pnl_array) if len(pnl_array) > 0 else 0
    avg_win = np.mean(positive_pnl) if len(positive_pnl) > 0 else 0
    avg_loss = np.mean(negative_pnl) if len(negative_pnl) > 0 else 0

    profit_factor = abs(np.sum(positive_pnl) / np.sum(negative_pnl)) if np.sum(negative_pnl) != 0 else np.inf

    stats_dict.update({
        'win_rate': win_rate * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else np.inf,
        'positive_trades': len(positive_pnl),
        'negative_trades': len(negative_pnl),
    })

    # 5. PORTFOLIO ANALYSIS (якщо доступно)
    portfolio_values = backtest_results['portfolio_value'].values
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        returns = returns[~np.isnan(returns)]  # Видаляємо NaN

        if len(returns) > 1 and np.std(returns) > 0:
            risk_free_rate = 0.02 / 252  # 2% річних, денна ставка
            excess_returns = returns - risk_free_rate

            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252)

            # Sortino Ratio
            negative_returns = returns[returns < 0]
            downside_std = np.std(negative_returns) if len(negative_returns) > 0 else np.std(returns)
            sortino_ratio = np.mean(excess_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

            stats_dict.update({
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'volatility_annual': np.std(returns) * np.sqrt(252) * 100,
                'portfolio_returns_count': len(returns)
            })

    # 6. МАКСИМАЛЬНА ПРОСАДКА
    cumulative_pnl = np.cumsum(pnl_array)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown = np.min(drawdown)

    initial_capital = 100000  # Базовий капітал
    max_drawdown_pct = (max_drawdown / initial_capital) * 100

    stats_dict.update({
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
    })

    # 7. ТЕСТИ НА НОРМАЛЬНІСТЬ (тільки якщо SciPy доступний)
    if SCIPY_AVAILABLE and len(pnl_array) >= 3:
        print(f"\n🔬 ТЕСТИ РОЗПОДІЛУ PnL:")

        try:
            if len(pnl_array) <= 5000:  # Shapiro працює до 5000 зразків
                shapiro_stat, shapiro_p = shapiro(pnl_array)
                stats_dict['shapiro_test_p'] = shapiro_p
                shapiro_normal = shapiro_p > 0.05
                print(f"   Шапіро-Вілк: p={shapiro_p:.4f} ({'нормальний' if shapiro_normal else 'не нормальний'})")
        except Exception as e:
            print(f"   Помилка тесту Шапіро-Вілк: {e}")
            stats_dict['shapiro_test_p'] = None

        try:
            if len(pnl_array) >= 8:  # Jarque-Bera потребує мінімум 8 зразків
                jb_stat, jb_p = jarque_bera(pnl_array)
                stats_dict['jarque_bera_p'] = jb_p
                jb_normal = jb_p > 0.05
                print(f"   Жарк-Бера: p={jb_p:.4f} ({'нормальний' if jb_normal else 'не нормальний'})")
        except Exception as e:
            print(f"   Помилка тесту Жарк-Бера: {e}")
            stats_dict['jarque_bera_p'] = None

    # 8. ВИВЕДЕННЯ РЕЗУЛЬТАТІВ
    print(f"\n📊 ОСНОВНІ СТАТИСТИКИ:")
    print(f"   Джерело даних: {data_source}")
    print(f"   Загалом трейдів: {stats_dict['total_trades']}")
    print(f"   Успішність: {stats_dict['win_rate']:.1f}%")
    print(f"   Середній PnL: ${stats_dict['mean_pnl']:+.2f}")
    print(f"   Медіанний PnL: ${stats_dict['median_pnl']:+.2f}")
    print(f"   Загальний PnL: ${stats_dict['total_pnl']:+.2f}")
    print(f"   Співвідношення прибуток/ризик: {stats_dict['risk_reward_ratio']:.2f}")
    print(f"   Фактор прибутку: {stats_dict['profit_factor']:.2f}")

    if 'sharpe_ratio' in stats_dict:
        print(f"\n📈 РИЗИК-СКОРИГОВАНІ МЕТРИКИ:")
        print(f"   Коефіцієнт Шарпа: {stats_dict['sharpe_ratio']:.3f}")
        print(f"   Коефіцієнт Сортіно: {stats_dict['sortino_ratio']:.3f}")
        print(f"   Річна волатільність: {stats_dict['volatility_annual']:.1f}%")
        print(f"   Максимальна просадка: ${stats_dict['max_drawdown']:+.2f} ({stats_dict['max_drawdown_pct']:+.2f}%)")

    # 9. МАТЕМАТИЧНИЙ ВИСНОВОК
    print(f"\n🎯 МАТЕМАТИЧНИЙ ВИСНОВОК:")

    strategy_score = 0
    conclusions = []

    # Оцінка успішності
    if stats_dict['win_rate'] > 60:
        strategy_score += 2
        conclusions.append("✅ Висока успішність")
    elif stats_dict['win_rate'] > 50:
        strategy_score += 1
        conclusions.append("🟡 Помірна успішність")
    else:
        conclusions.append("❌ Низька успішність")

    # Оцінка фактора прибутку
    if stats_dict['profit_factor'] > 1.5:
        strategy_score += 2
        conclusions.append("✅ Хороший фактор прибутку")
    elif stats_dict['profit_factor'] > 1.0:
        strategy_score += 1
        conclusions.append("🟡 Помірний фактор прибутку")
    else:
        conclusions.append("❌ Поганий фактор прибутку")

    # Оцінка загального PnL
    if stats_dict['total_pnl'] > 1000:
        strategy_score += 2
        conclusions.append("✅ Значний прибуток")
    elif stats_dict['total_pnl'] > 0:
        strategy_score += 1
        conclusions.append("🟡 Помірний прибуток")
    else:
        conclusions.append("❌ Збиток")

    # Оцінка Sharpe ratio (якщо доступний)
    if 'sharpe_ratio' in stats_dict:
        if stats_dict['sharpe_ratio'] > 1.0:
            strategy_score += 2
            conclusions.append("✅ Відмінний Sharpe ratio")
        elif stats_dict['sharpe_ratio'] > 0.5:
            strategy_score += 1
            conclusions.append("🟡 Прийнятний Sharpe ratio")
        else:
            conclusions.append("❌ Поганий Sharpe ratio")

    # Фінальна оцінка
    if strategy_score >= 6:
        final_verdict = "🏆 СТРАТЕГІЯ МАТЕМАТИЧНО ПРИБУТКОВА"
    elif strategy_score >= 4:
        final_verdict = "🟡 СТРАТЕГІЯ ПОТРЕБУЄ ОПТИМІЗАЦІЇ"
    else:
        final_verdict = "❌ СТРАТЕГІЯ МАТЕМАТИЧНО ЗБИТКОВА"

    print(f"   {final_verdict}")
    for conclusion in conclusions:
        print(f"   {conclusion}")

    stats_dict['strategy_score'] = strategy_score
    stats_dict['final_verdict'] = final_verdict
    stats_dict['conclusions'] = conclusions

    return stats_dict
def plot_backtest_results_simple_pnl(backtest_results_df, system_name="Торгова система"):
    """📊 ПРОСТА але ЕФЕКТИВНА візуалізація з PnL прямо з даних"""
    try:
        if backtest_results_df.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 8))

        # График 1: ЦІНА І ТРЕЙДИ
        ax1.plot(backtest_results_df['timestamp'], backtest_results_df['price'],
                 label='Ціна BTC', color='blue', linewidth=2, alpha=0.8)

        executed_trades = backtest_results_df[backtest_results_df['executed'] == True]

        if not executed_trades.empty:
            # Розділяємо по типах сигналів
            buy_signals = executed_trades[executed_trades['signal'] == 'BUY']
            short_signals = executed_trades[executed_trades['signal'] == 'SHORT']
            sell_signals = executed_trades[executed_trades['signal'].isin(['SELL', 'COVER'])]

            # Розділяємо sell_signals по прибутковості (якщо є PnL)
            if 'pnl' in sell_signals.columns:
                profitable_sells = sell_signals[sell_signals['pnl'] > 0]
                losing_sells = sell_signals[sell_signals['pnl'] <= 0]
            else:
                profitable_sells = sell_signals
                losing_sells = pd.DataFrame()

            # Показуємо трейди
            if not buy_signals.empty:
                ax1.scatter(buy_signals['timestamp'], buy_signals['price'],
                            marker='^', color='green', s=100, label=f'🟢 BUY ({len(buy_signals)})',
                            zorder=5, alpha=0.9)

            if not short_signals.empty:
                ax1.scatter(short_signals['timestamp'], short_signals['price'],
                            marker='s', color='purple', s=120, label=f'🟣 SHORT ({len(short_signals)})',
                            zorder=6, alpha=0.9)

            if not profitable_sells.empty:
                ax1.scatter(profitable_sells['timestamp'], profitable_sells['price'],
                            marker='v', color='darkgreen', s=100, label=f'✅ PROFIT ({len(profitable_sells)})',
                            zorder=7, alpha=0.9)

            if not losing_sells.empty:
                ax1.scatter(losing_sells['timestamp'], losing_sells['price'],
                            marker='X', color='red', s=100, label=f'❌ LOSS ({len(losing_sells)})',
                            zorder=7, alpha=0.9)

        ax1.set_title(f'{system_name}: Ціна та трейди', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)

        # График 2: КРИВА PnL (ПРОСТА ЛОГІКА)
        pnl_curve = []
        running_pnl = 0.0

        print("📊 Розрахунок простої кривої PnL...")

        for i, row in backtest_results_df.iterrows():
            # Якщо трейд виконано і є PnL - додаємо до накопичення
            if row.get('executed', False) and pd.notna(row.get('pnl', 0)):
                pnl_value = row['pnl']
                running_pnl += pnl_value
                if abs(pnl_value) > 1:  # Тільки суттєві зміни
                    print(f"   {row['signal']}: ${pnl_value:+.0f} → Накопичено: ${running_pnl:+.0f}")

            pnl_curve.append(running_pnl)

        # Альтернативний розрахунок якщо PnL дані відсутні
        if all(pnl == 0 for pnl in pnl_curve):
            print("⚠️ Немає PnL даних, використовуємо зміни портфеля...")
            running_value = backtest_results_df['portfolio_value'].iloc[0] if not backtest_results_df.empty else 100000

            for i, row in backtest_results_df.iterrows():
                if i == 0:
                    pnl_curve[i] = 0
                else:
                    portfolio_change = row['portfolio_value'] - backtest_results_df['portfolio_value'].iloc[0]
                    pnl_curve[i] = portfolio_change

        # Малюємо криву PnL
        ax2.plot(backtest_results_df['timestamp'], pnl_curve,
                 label='Накопичена дохідність', color='green', linewidth=3)

        # Базова лінія
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Базова лінія')

        final_pnl = pnl_curve[-1] if pnl_curve else 0

        # Визначаємо колір і статус
        if final_pnl > 0:
            title_color = 'green'
            status = '📈'
            fill_color = 'green'
            fill_alpha = 0.2
        else:
            title_color = 'red'
            status = '📉'
            fill_color = 'red'
            fill_alpha = 0.2

        # Заповнення області
        ax2.fill_between(backtest_results_df['timestamp'], 0, pnl_curve,
                         alpha=fill_alpha, color=fill_color)

        # Анотації для піків і провалів
        if len(pnl_curve) > 0:
            max_pnl = max(pnl_curve)
            min_pnl = min(pnl_curve)

            if max_pnl > 100:
                max_idx = pnl_curve.index(max_pnl)
                max_time = backtest_results_df.iloc[max_idx]['timestamp']
                ax2.annotate(f'Пік: ${max_pnl:,.0f}',
                             xy=(max_time, max_pnl), xytext=(10, 10),
                             textcoords='offset points',
                             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                             fontsize=9)

            if min_pnl < -100:
                min_idx = pnl_curve.index(min_pnl)
                min_time = backtest_results_df.iloc[min_idx]['timestamp']
                ax2.annotate(f'Дно: ${min_pnl:,.0f}',
                             xy=(min_time, min_pnl), xytext=(10, -20),
                             textcoords='offset points',
                             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                             fontsize=9)

        ax2.set_title(f'{status} Дохідність: ${final_pnl:+,.0f}',
                      fontsize=14, fontweight='bold', color=title_color)
        ax2.set_ylabel('Накопичена дохідність (USD)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='upper left')

        # Статистика на графіку
        total_trades = len(executed_trades) if not executed_trades.empty else 0

        if 'pnl' in executed_trades.columns and not executed_trades.empty:
            profitable_trades = len(executed_trades[executed_trades['pnl'] > 0])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        else:
            profitable_trades = 0
            win_rate = 0

        stats_text = f'Трейдів: {total_trades}\nПрибуткових: {profitable_trades}\nУспішність: {win_rate:.1f}%'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=10)

        plt.tight_layout()
        plt.show()

        print(f"✅ Проста візуалізація створена")
        print(f"📊 Фінальний PnL: ${final_pnl:+.0f}")
        print(f"📊 Трейдів: {total_trades}, Прибуткових: {profitable_trades}")

    except Exception as e:
        print(f"❌ Помилка простої візуалізації: {e}")
        import traceback
        print(traceback.format_exc())
# ПОКРАЩЕНА ФУНКЦІЯ ВІЗУАЛІЗАЦІЇ З ВИКОРИСТАННЯМ ЛОГІВ
def plot_backtest_results_with_log_pnl(backtest_results_df, system_name="Торгова система"):
    """🔧 Візуалізація з PnL з логів замість даних"""
    try:
        if backtest_results_df.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 8))

        # График 1: ТРЕЙДИ (існуючий код)
        ax1.plot(backtest_results_df['timestamp'], backtest_results_df['price'],
                 label='Ціна BTC', color='blue', linewidth=2, alpha=0.8)

        executed_trades = backtest_results_df[backtest_results_df['executed'] == True]
        long_open = executed_trades[executed_trades['signal'] == 'BUY']
        short_open = executed_trades[executed_trades['signal'] == 'SHORT']

        # Оцінюємо закриття
        long_close = pd.DataFrame()
        short_close = pd.DataFrame()

        for _, buy_trade in long_open.iterrows():
            next_trades = executed_trades[
                (executed_trades['timestamp'] > buy_trade['timestamp']) &
                (executed_trades.get('portfolio_change', 0) < 0)
                ]
            if not next_trades.empty:
                close_trade = next_trades.iloc[0]
                long_close = pd.concat([long_close, close_trade.to_frame().T])

        for _, short_trade in short_open.iterrows():
            next_trades = executed_trades[
                (executed_trades['timestamp'] > short_trade['timestamp']) &
                (executed_trades.get('portfolio_change', 0) > 0)
                ]
            if not next_trades.empty:
                close_trade = next_trades.iloc[0]
                short_close = pd.concat([short_close, close_trade.to_frame().T])

        long_close = long_close.drop_duplicates()
        short_close = short_close.drop_duplicates()

        # Показуємо трейди
        if not long_open.empty:
            ax1.scatter(long_open['timestamp'], long_open['price'],
                        marker='^', color='green', s=120, label=f'🟢 LONG OPEN ({len(long_open)})',
                        zorder=5, alpha=0.9, edgecolors='white', linewidth=2)

        if not long_close.empty:
            ax1.scatter(long_close['timestamp'], long_close['price'],
                        marker='v', color='red', s=120, label=f'🔴 LONG CLOSE ({len(long_close)})',
                        zorder=5, alpha=0.9, edgecolors='white', linewidth=2)

        if not short_open.empty:
            ax1.scatter(short_open['timestamp'], short_open['price'],
                        marker='s', color='purple', s=120, label=f'🟣 SHORT OPEN ({len(short_open)})',
                        zorder=6, alpha=0.9, edgecolors='white', linewidth=2)

        if not short_close.empty:
            ax1.scatter(short_close['timestamp'], short_close['price'],
                        marker='X', color='orange', s=140, label=f'🟠 SHORT CLOSE ({len(short_close)})',
                        zorder=6, alpha=0.9, edgecolors='white', linewidth=2)

        ax1.set_title(f'{system_name}: Ціна та ТРЕЙДИ (OPEN/CLOSE)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)

        # 🆕 График 2: КРИВА З ЛОГІВ
        cumulative_pnl_from_logs = get_cumulative_pnl_from_logs()

        if cumulative_pnl_from_logs and len(cumulative_pnl_from_logs) > 0:
            # Створюємо часові мітки для PnL з логів
            if len(cumulative_pnl_from_logs) <= len(backtest_results_df):
                # Розподіляємо PnL по часу
                time_indices = np.linspace(0, len(backtest_results_df) - 1, len(cumulative_pnl_from_logs), dtype=int)
                pnl_timestamps = [backtest_results_df.iloc[i]['timestamp'] for i in time_indices]
            else:
                # Якщо PnL записів більше, обрізаємо
                cumulative_pnl_from_logs = cumulative_pnl_from_logs[:len(backtest_results_df)]
                pnl_timestamps = backtest_results_df['timestamp'].tolist()

            # Розширюємо для повного графіка
            full_pnl_curve = []
            pnl_idx = 0

            for i, timestamp in enumerate(backtest_results_df['timestamp']):
                if pnl_idx < len(pnl_timestamps) and timestamp >= pnl_timestamps[pnl_idx]:
                    current_pnl = cumulative_pnl_from_logs[pnl_idx]
                    if pnl_idx < len(cumulative_pnl_from_logs) - 1:
                        pnl_idx += 1
                else:
                    current_pnl = cumulative_pnl_from_logs[pnl_idx - 1] if pnl_idx > 0 else 0

                full_pnl_curve.append(current_pnl)

            ax2.plot(backtest_results_df['timestamp'], full_pnl_curve,
                     label='Накопичена дохідність (з логів)', color='green', linewidth=3)

            # Початкова лінія (0)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Початок (0)')

            final_pnl = full_pnl_curve[-1] if full_pnl_curve else 0

            # Кольорове заповнення
            if final_pnl > 0:
                ax2.fill_between(backtest_results_df['timestamp'], 0, full_pnl_curve,
                                 alpha=0.2, color='green', label='Накопичений прибуток')
                title_color = 'green'
                status = '📈'
            else:
                ax2.fill_between(backtest_results_df['timestamp'], 0, full_pnl_curve,
                                 alpha=0.2, color='red', label='Накопичений збиток')
                title_color = 'red'
                status = '📉'

            ax2.set_title(f'{status} Крива дохідності (з логів): ${final_pnl:+,.0f}',
                          fontsize=14, fontweight='bold', color=title_color)

        else:
            ax2.text(0.5, 0.5, 'Немає PnL даних з логів\nПеревірте функцію capture_pnl_from_log',
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            ax2.set_title('❌ Крива дохідності: немає логів', fontsize=14, fontweight='bold', color='red')

        ax2.set_ylabel('Накопичена дохідність (USD)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='upper left')

        plt.tight_layout()
        plt.show()

        print("✅ Візуалізацію з PnL з логів створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        import traceback
        print(traceback.format_exc())


# МОДИФІКАЦІЇ ДЛЯ ІНТЕГРАЦІЇ В ІСНУЮЧУ СИСТЕМУ
def integrate_log_capture_into_trading_system():
    """🔧 Інструкції для інтеграції захоплення логів"""
    print("""
    🔧 ДЛЯ ІНТЕГРАЦІЇ ДОДАЙТЕ ЦЕ В aggressive_trading_system_SL_STOP.py:

    1. В початок файлу:
    from aggressive_backtesting_SL_STOP import capture_pnl_from_log

    2. В функції _close_position, після кожного print з PnL:

    print(f"🔒 ЗАКРИТТЯ LONG позиції ({reason.upper()}): {amount_to_close:.6f} BTC @ ${effective_price:.0f}")
    print(f"   PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%) | Balance: ${self.balance:.0f}")
    capture_pnl_from_log(f"PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%)")  # ← ДОДАТИ ЦЕ

    print(f"🔒 ЗАКРИТТЯ SHORT позиції ({reason.upper()}): {amount_to_close:.6f} BTC @ ${effective_price:.0f}")
    print(f"   🔧 ШТУЧНИЙ ШОРТ PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%)")
    capture_pnl_from_log(f"ШТУЧНИЙ ШОРТ PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%)")  # ← ДОДАТИ ЦЕ

    3. В main_stoploss.py перед бектестом:
    reset_trading_log()  # Скидаємо попередні дані

    4. Замінити виклик візуалізації:
    plot_backtest_results_with_log_pnl(backtest_results_df, "Система з логів")
    """)
def ensure_dir(directory_path):
    """Створює директорію, якщо вона не існує"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def timestamp_to_string():
    """Повертає поточну дату і час у форматі для назв файлів"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def safe_get_risk_params(trading_system):
    """Безпечно отримує параметри ризику з торгової системи"""
    if hasattr(trading_system, 'risk_params') and trading_system.risk_params:
        return trading_system.risk_params
    else:
        # Значення за замовчуванням
        print("⚠️ risk_params не знайдено, використовуємо значення за замовчуванням")
        return {
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'trailing_stop_enabled': False,
            'trailing_stop_pct': 0.03,
            'max_open_positions': 2,
            'position_timeout_hours': 100
        }


# ВИПРАВЛЕННЯ В ФАЙЛІ aggressive_backtesting_SL_STOP.py
# Додайте цю змінену частину в функцію run_aggressive_backtest:

def run_aggressive_backtest(trading_system, historical_data, feature_names_for_model):
    """
    🔧 ПОКРАЩЕНИЙ бектест з детальним логуванням та ПРАВИЛЬНИМ записуванням PnL
    """
    print("⚖️ Запуск покращеного бектесту з шорт позиціями...")

    # Безпечно отримуємо параметри ризику
    risk_params = safe_get_risk_params(trading_system)

    print(f"🛡️ Стоп-лос: {risk_params['stop_loss_pct'] * 100:.1f}%")
    print(f"🎯 Тейк-профіт: {risk_params['take_profit_pct'] * 100:.1f}%")

    # Скидаємо стан системи
    if hasattr(trading_system, 'reset_state'):
        trading_system.reset_state()
    else:
        print("⚠️ Метод reset_state не знайдено")

    results_log = []
    n_points_to_process = min(200, len(historical_data))
    step = max(1, len(historical_data) // n_points_to_process)

    # Отримуємо початковий баланс безпечно
    if hasattr(trading_system, 'initial_balance'):
        initial_balance_display = trading_system.initial_balance
    elif hasattr(trading_system, 'balance'):
        initial_balance_display = trading_system.balance
    else:
        initial_balance_display = 10000
        print("⚠️ Не вдалося знайти початковий баланс, використовується $10,000")

    signals_generated = {'BUY': 0, 'SELL': 0, 'SHORT': 0, 'HOLD': 0}

    # Перевірка наявності необхідних колонок
    required_cols = ['close', 'timestamp'] + feature_names_for_model
    missing_cols = [col for col in required_cols if col not in historical_data.columns]
    if missing_cols:
        print(f"⚠️ Відсутні колонки: {missing_cols}")
        for col in missing_cols:
            if col == 'close':
                historical_data[col] = 50000
            elif col == 'timestamp':
                historical_data[col] = pd.date_range('2023-01-01', periods=len(historical_data), freq='H')
            else:
                historical_data[col] = 0

    print(f"📊 ПОЧАТКОВИЙ СТАН ПОРТФЕЛЯ:")
    print(f"   💰 Balance: ${initial_balance_display:,.2f}")
    print(f"📊 Обробляємо {n_points_to_process} точок з {len(historical_data)} (крок: {step})")

    for i in range(0, len(historical_data), step):
        try:
            row = historical_data.iloc[i]
            current_price = row['close']
            timestamp = row['timestamp']

            # Встановлюємо high/low для точного стоп-лосу/тейк-профіту
            trading_system._current_high = row.get('high', current_price)
            trading_system._current_low = row.get('low', current_price)

            # 🆕 СПОЧАТКУ ПЕРЕВІРЯЄМО СТОП-ЛОС/ТЕЙК-ПРОФІТ (може створити PnL записи)
            closed_positions_count = 0
            if hasattr(trading_system, 'check_stop_loss_take_profit'):
                closed_positions_count = trading_system.check_stop_loss_take_profit(
                    current_price, timestamp, trading_system._current_high, trading_system._current_low
                )

            # 🆕 ДОДАЄМО ЗАПИСИ ПРО ЗАКРИТІ ПОЗИЦІЇ ДО РЕЗУЛЬТАТІВ
            if closed_positions_count > 0 and hasattr(trading_system, 'trade_history'):
                # Беремо останні записи з trade_history (закриті позиції)
                recent_closes = trading_system.trade_history[-closed_positions_count:]
                for close_record in recent_closes:
                    # Створюємо запис для результатів бектесту з PnL
                    close_log_entry = {
                        'timestamp': close_record['timestamp'],
                        'price': close_record['price'],
                        'predicted_price': close_record.get('predicted_price', close_record['price']),
                        'price_change_pct': 0,
                        'signal': close_record['signal'],
                        'signal_type': close_record['signal_type'],
                        'confidence': close_record.get('confidence', 1.0),
                        'portfolio_value': close_record['portfolio_value'],
                        'balance': close_record['balance_after'],
                        'btc_holdings': close_record['btc_after'],
                        'borrowed_btc': 0,
                        'executed': True,
                        'total_cost': close_record['total_cost'],
                        'fee': close_record['fee'],
                        'slippage': close_record['slippage'],
                        'effective_price': close_record['effective_price'],
                        'reason_not_executed': None,
                        'portfolio_change': close_record.get('portfolio_change', close_record.get('pnl', 0)),
                        'open_positions': len(trading_system.positions),
                        'long_positions': len([p for p in trading_system.positions if p.get('type') == 'BUY']),
                        'short_positions': len([p for p in trading_system.positions if p.get('type') == 'SHORT']),
                        'stop_loss_price': None,
                        'take_profit_price': None,
                        'rsi': row.get('rsi', 50),
                        'macd': row.get('macd_12_26', row.get('macd', 0)),
                        'macd_signal': row.get('macd_signal_12_26', row.get('macd_signal', 0)),
                        # 🆕 ВАЖЛИВО: Додаємо PnL дані
                        'pnl': close_record.get('pnl', 0),
                        'pnl_pct': close_record.get('pnl_pct', 0),
                        'close_reason': close_record.get('close_reason'),
                        'position_type': close_record.get('position_type'),
                        'entry_price': close_record.get('entry_price'),
                    }
                    results_log.append(close_log_entry)
                    print(f"📝 Додано запис про закриття з PnL: ${close_record.get('pnl', 0):+.0f}")

            # Підготовка ознак для моделі
            model_input_features_series = row[feature_names_for_model]
            model_input_features_for_prediction = pd.DataFrame([model_input_features_series],
                                                               columns=feature_names_for_model)

            # Технічні індикатори з безпечним доступом
            technical_indicators_dict = {
                'rsi': row.get('rsi', 50),
                'macd': row.get('macd_12_26', row.get('macd', 0)),
                'macd_signal': row.get('macd_signal_12_26', row.get('macd_signal', 0)),
                'adx': row.get('adx', 20),
                'volume_ratio_20': row.get('volume_ratio_20', 1.0)
            }

            # Отримуємо баланс портфеля безпечно
            if hasattr(trading_system, 'balance') and hasattr(trading_system, 'btc_holdings'):
                portfolio_before_signal = trading_system.balance + (trading_system.btc_holdings * current_price)
            else:
                portfolio_before_signal = initial_balance_display

            # Генерація сигналу
            try:
                if hasattr(trading_system, 'generate_aggressive_signal'):
                    signal_result = trading_system.generate_aggressive_signal(
                        features=model_input_features_for_prediction,
                        current_price=current_price,
                        technical_indicators=technical_indicators_dict
                    )

                    if len(signal_result) >= 5:
                        signal, predicted_price, price_change_pct, confidence, signal_type = signal_result
                    else:
                        signal, predicted_price, price_change_pct, confidence = signal_result[:4]
                        signal_type = 'regular'
                else:
                    signal = 'HOLD'
                    predicted_price = current_price
                    price_change_pct = 0
                    confidence = 0
                    signal_type = 'regular'

            except Exception as e:
                print(f"⚠️ Помилка генерації сигналу: {e}")
                signal = 'HOLD'
                predicted_price = current_price
                price_change_pct = 0
                confidence = 0
                signal_type = 'regular'

            # Підрахунок сигналів
            signals_generated[signal] = signals_generated.get(signal, 0) + 1

            # ВИКОНАННЯ ТОРГІВЛІ (тільки якщо це не HOLD)
            if signal != 'HOLD':
                try:
                    if hasattr(trading_system, 'execute_trade_with_costs'):
                        trade_info = trading_system.execute_trade_with_costs(
                            signal=signal,
                            current_price=current_price,
                            timestamp=timestamp,
                            predicted_price=predicted_price,
                            confidence=confidence,
                            signal_type=signal_type
                        )
                    else:
                        trade_info = {
                            'executed': False,
                            'portfolio_value': portfolio_before_signal,
                            'balance_after': getattr(trading_system, 'balance', initial_balance_display),
                            'btc_after': getattr(trading_system, 'btc_holdings', 0),
                            'total_cost': 0,
                            'fee': 0,
                            'slippage': 0,
                            'effective_price': current_price,
                            'reason': 'Method not found'
                        }
                except Exception as e:
                    print(f"⚠️ Помилка виконання торгівлі: {e}")
                    trade_info = {
                        'executed': False,
                        'portfolio_value': portfolio_before_signal,
                        'balance_after': initial_balance_display,
                        'btc_after': 0,
                        'borrowed_btc_after': 0,
                        'total_cost': 0,
                        'fee': 0,
                        'slippage': 0,
                        'effective_price': current_price,
                        'reason': str(e)
                    }

                # Стан портфеля ПІСЛЯ торгівлі
                portfolio_after_trade = trade_info.get('portfolio_value', portfolio_before_signal)
                portfolio_change = portfolio_after_trade - portfolio_before_signal

                # Отримуємо кількість відкритих позицій безпечно
                open_positions = 0
                long_positions = 0
                short_positions = 0

                if hasattr(trading_system, 'positions'):
                    open_positions = len(trading_system.positions)
                    long_positions = len([p for p in trading_system.positions if p.get('type') == 'BUY'])
                    short_positions = len([p for p in trading_system.positions if p.get('type') == 'SHORT'])

                # Логування результату (тільки для не-HOLD сигналів)
                log_entry = {
                    'timestamp': timestamp,
                    'price': current_price,
                    'predicted_price': predicted_price,
                    'price_change_pct': price_change_pct,
                    'signal': signal,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'portfolio_value': portfolio_after_trade,
                    'balance': trade_info.get('balance_after', initial_balance_display),
                    'btc_holdings': trade_info.get('btc_after', 0),
                    'borrowed_btc': trade_info.get('borrowed_btc_after', 0),
                    'executed': trade_info.get('executed', False),
                    'total_cost': trade_info.get('total_cost', 0),
                    'fee': trade_info.get('fee', 0),
                    'slippage': trade_info.get('slippage', 0),
                    'effective_price': trade_info.get('effective_price', current_price),
                    'reason_not_executed': trade_info.get('reason', None),
                    'portfolio_change': portfolio_change,
                    'open_positions': open_positions,
                    'long_positions': long_positions,
                    'short_positions': short_positions,
                    'stop_loss_price': trade_info.get('stop_loss_price', None),
                    'take_profit_price': trade_info.get('take_profit_price', None),
                    'rsi': technical_indicators_dict['rsi'],
                    'macd': technical_indicators_dict['macd'],
                    'macd_signal': technical_indicators_dict['macd_signal'],
                    # PnL буде 0 для відкриття позицій, не-0 для закриття
                    'pnl': trade_info.get('pnl', 0),
                    'pnl_pct': trade_info.get('pnl_pct', 0),
                    'close_reason': trade_info.get('close_reason', None),
                    'position_type': trade_info.get('position_type', None),
                    'entry_price': trade_info.get('entry_price', None),
                }

                results_log.append(log_entry)

        except Exception as e:
            print(f"⚠️ Помилка на ітерації {i}: {e}")
            continue

        # Прогрес та діагностика
        if i > 0 and (i // step) % 50 == 0:
            progress_pct = (i + step) / len(historical_data) * 100
            executed_trades = sum(1 for r in results_log if r.get('executed', False))
            pnl_records = sum(1 for r in results_log if abs(r.get('pnl', 0)) > 0)
            print(
                f"🔄 Прогрес: {progress_pct:.1f}% | Сигналів: {signals_generated} | Виконано: {executed_trades} | PnL записів: {pnl_records}")

            if results_log:
                current_portfolio = results_log[-1]['portfolio_value']
                long_positions = len([p for p in trading_system.positions if p.get('type') == 'BUY'])
                short_positions = len([p for p in trading_system.positions if p.get('type') == 'SHORT'])
                print(f"   💰 Портфель: ${current_portfolio:,.0f} | Позицій: L{long_positions}/S{short_positions}")
    print(f"\n🔚 ФІНАЛЬНЕ ЗАКРИТТЯ ПОЗИЦІЙ")
    print("=" * 50)

    final_price = historical_data['close'].iloc[-1]
    final_timestamp = historical_data['timestamp'].iloc[-1]

    # Додаємо метод до торгової системи якщо його немає
    if not hasattr(trading_system, 'force_close_all_positions'):
        # Inline реалізація
        positions_count = len(trading_system.positions)
        if positions_count > 0:
            print(f"📊 Закриваємо {positions_count} відкритих позицій за фінальною ціною ${final_price:.0f}")

            positions_to_close = []
            for i, position in enumerate(trading_system.positions):
                positions_to_close.append({
                    'index': i,
                    'position': position,
                    'reason': 'final_close',
                    'current_price': final_price,
                    'timestamp': final_timestamp
                })

            # Закриваємо всі позиції
            for close_info in reversed(positions_to_close):
                trading_system._close_position(close_info)

            # 🆕 ДОДАЄМО ЗАПИСИ ПРО ФІНАЛЬНІ ЗАКРИТТЯ ДО РЕЗУЛЬТАТІВ
            if hasattr(trading_system, 'trade_history'):
                recent_closes = trading_system.trade_history[-positions_count:]
                for close_record in recent_closes:
                    close_log_entry = {
                        'timestamp': close_record['timestamp'],
                        'price': close_record['price'],
                        'predicted_price': close_record.get('predicted_price', close_record['price']),
                        'price_change_pct': 0,
                        'signal': close_record['signal'],
                        'signal_type': close_record['signal_type'],
                        'confidence': close_record.get('confidence', 1.0),
                        'portfolio_value': close_record['portfolio_value'],
                        'balance': close_record['balance_after'],
                        'btc_holdings': close_record['btc_after'],
                        'borrowed_btc': 0,
                        'executed': True,
                        'total_cost': close_record['total_cost'],
                        'fee': close_record['fee'],
                        'slippage': close_record['slippage'],
                        'effective_price': close_record['effective_price'],
                        'reason_not_executed': None,
                        'portfolio_change': close_record.get('portfolio_change', close_record.get('pnl', 0)),
                        'open_positions': 0,  # Всі позиції закриті
                        'long_positions': 0,
                        'short_positions': 0,
                        'stop_loss_price': None,
                        'take_profit_price': None,
                        'rsi': historical_data['rsi'].iloc[-1] if 'rsi' in historical_data.columns else 50,
                        'macd': historical_data.get('macd_12_26', historical_data.get('macd', [0])).iloc[
                            -1] if 'macd_12_26' in historical_data.columns else 0,
                        'macd_signal':
                            historical_data.get('macd_signal_12_26', historical_data.get('macd_signal', [0])).iloc[
                                -1] if 'macd_signal_12_26' in historical_data.columns else 0,
                        # 🆕 ВАЖЛИВО: PnL з фінального закриття
                        'pnl': close_record.get('pnl', 0),
                        'pnl_pct': close_record.get('pnl_pct', 0),
                        'close_reason': close_record.get('close_reason'),
                        'position_type': close_record.get('position_type'),
                        'entry_price': close_record.get('entry_price'),
                    }
                    results_log.append(close_log_entry)
                    print(f"📝 Додано фінальне закриття з PnL: ${close_record.get('pnl', 0):+.0f}")

            print(f"✅ Фінальне закриття: {positions_count} позицій")
        else:
            print("📊 Немає відкритих позицій для закриття")
    else:
        # Використовуємо існуючий метод
        final_closes = trading_system.force_close_all_positions(final_price, final_timestamp)

    # Створення результатів
    results_df = pd.DataFrame(results_log)

    # 🆕 ПОКРАЩЕНА ДІАГНОСТИКА РЕЗУЛЬТАТІВ
    if not results_df.empty:
        all_pnl_records = results_df[results_df['pnl'] != 0]
        positive_pnl = len(all_pnl_records[all_pnl_records['pnl'] > 0])
        negative_pnl = len(all_pnl_records[all_pnl_records['pnl'] < 0])
        total_pnl = results_df['pnl'].sum()

        print(f"\n📊 ФІНАЛЬНА ДІАГНОСТИКА РЕЗУЛЬТАТІВ БЕКТЕСТУ:")
        print(f"   Загальна кількість записів: {len(results_df)}")
        print(f"   Записів з PnL: {len(all_pnl_records)}")
        print(f"   Прибуткових записів: {positive_pnl}")
        print(f"   Збиткових записів: {negative_pnl}")
        print(f"   Загальний PnL: ${total_pnl:+.2f}")

        if len(all_pnl_records) > 0:
            print(f"   Середній PnL на операцію: ${total_pnl / len(all_pnl_records):+.2f}")
            print(f"   Успішність: {positive_pnl / len(all_pnl_records) * 100:.1f}%")

            # Показуємо кілька прикладів PnL
            print(f"\n📋 Приклади PnL записів:")
            sample_pnl = all_pnl_records.head(5)
            for idx, row in sample_pnl.iterrows():
                print(f"   {row['signal']} {row['signal_type']}: ${row['pnl']:+.0f} ({row.get('close_reason', 'N/A')})")
        else:
            print(f"   ⚠️ УВАГА: Все ще немає PnL записів!")
    # Створення результатів
    results_df = pd.DataFrame(results_log)

    # 🆕 ДІАГНОСТИКА РЕЗУЛЬТАТІВ
    if not results_df.empty:
        pnl_records = len(results_df[results_df['pnl'] != 0])
        total_pnl = results_df['pnl'].sum()
        print(f"\n📊 ДІАГНОСТИКА РЕЗУЛЬТАТІВ БЕКТЕСТУ:")
        print(f"   Загальна кількість записів: {len(results_df)}")
        print(f"   Записів з PnL: {pnl_records}")
        print(f"   Загальний PnL: ${total_pnl:+.2f}")

    # Решта функції залишається без змін...
    try:
        if hasattr(trading_system, 'get_trading_statistics'):
            final_trading_stats = trading_system.get_trading_statistics()
        else:
            final_trading_stats = {
                'total_trades': len([r for r in results_log if r.get('executed', False)]),
                'stop_loss_triggered': 0,
                'take_profit_triggered': 0,
                'total_fees_paid': sum([r.get('fee', 0) for r in results_log])
            }
    except Exception as e:
        print(f"⚠️ Помилка отримання статистики: {e}")
        final_trading_stats = {'total_trades': 0}

    return results_df


# ЗАМІНІТЬ ІСНУЮЧУ ФУНКЦІЮ plot_backtest_results у файлі aggressive_backtesting_SL_STOP.py

# ЗАМІНІТЬ ІСНУЮЧУ ФУНКЦІЮ plot_backtest_results у файлі aggressive_backtesting_SL_STOP.py

# ЗАМІНІТЬ ІСНУЮЧУ ФУНКЦІЮ plot_backtest_results у файлі aggressive_backtesting_SL_STOP.py

# Замініть функцію plot_backtest_results в aggressive_backtesting_SL_STOP.py

def plot_backtest_results(backtest_results_df, system_name="Торгова система"):
    """📊 ВИПРАВЛЕНА візуалізація з правильною кривою дохідності"""
    try:
        if backtest_results_df.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 8))

        # График 1: ЦІНА І ТРЕЙДИ
        ax1.plot(backtest_results_df['timestamp'], backtest_results_df['price'],
                 label='Ціна BTC', color='blue', linewidth=2, alpha=0.8)

        executed_trades = backtest_results_df[backtest_results_df['executed'] == True]

        if not executed_trades.empty:
            # Розділяємо по типах сигналів
            buy_signals = executed_trades[executed_trades['signal'] == 'BUY']
            short_signals = executed_trades[executed_trades['signal'] == 'SHORT']

            # Всі закриття позицій (SELL, COVER або автоматичні закриття)
            close_signals = executed_trades[
                (executed_trades['signal'].isin(['SELL', 'COVER'])) |
                (executed_trades['signal_type'].str.contains('close_', na=False))
                ]

            # Розділяємо закриття по прибутковості
            if 'pnl' in close_signals.columns and not close_signals['pnl'].isna().all():
                profitable_closes = close_signals[close_signals['pnl'] > 0]
                losing_closes = close_signals[close_signals['pnl'] <= 0]
            else:
                profitable_closes = close_signals
                losing_closes = pd.DataFrame()

            # Показуємо трейди
            if not buy_signals.empty:
                ax1.scatter(buy_signals['timestamp'], buy_signals['price'],
                            marker='^', color='green', s=100,
                            label=f'🟢 BUY ({len(buy_signals)})',
                            zorder=5, alpha=0.9, edgecolors='white', linewidth=1)

            if not short_signals.empty:
                ax1.scatter(short_signals['timestamp'], short_signals['price'],
                            marker='s', color='purple', s=120,
                            label=f'🟣 SHORT ({len(short_signals)})',
                            zorder=6, alpha=0.9, edgecolors='white', linewidth=1)

            if not profitable_closes.empty:
                ax1.scatter(profitable_closes['timestamp'], profitable_closes['price'],
                            marker='v', color='darkgreen', s=100,
                            label=f'✅ PROFIT ({len(profitable_closes)})',
                            zorder=7, alpha=0.9, edgecolors='white', linewidth=1)

            if not losing_closes.empty:
                ax1.scatter(losing_closes['timestamp'], losing_closes['price'],
                            marker='X', color='red', s=100,
                            label=f'❌ LOSS ({len(losing_closes)})',
                            zorder=7, alpha=0.9, edgecolors='white', linewidth=1)

        ax1.set_title(f'{system_name}: Ціна та трейди', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)

        # График 2: ВИПРАВЛЕНА КРИВА ДОХІДНОСТІ
        pnl_curve = calculate_correct_pnl_curve(backtest_results_df)

        if any(pnl != 0 for pnl in pnl_curve):
            # Є реальні PnL дані
            ax2.plot(backtest_results_df['timestamp'], pnl_curve,
                     label='Накопичена дохідність', color='green', linewidth=3)

            final_pnl = pnl_curve[-1] if pnl_curve else 0

            # Колір залежно від результату
            if final_pnl > 0:
                title_color = 'green'
                status = '📈'
                fill_color = 'green'
            else:
                title_color = 'red'
                status = '📉'
                fill_color = 'red'

            # Заповнення області
            ax2.fill_between(backtest_results_df['timestamp'], 0, pnl_curve,
                             alpha=0.2, color=fill_color)

            ax2.set_title(f'{status} Дохідність: ${final_pnl:+,.0f}',
                          fontsize=14, fontweight='bold', color=title_color)

        else:
            # Fallback: використовуємо зміни портфеля
            initial_value = backtest_results_df['portfolio_value'].iloc[0]
            portfolio_curve = backtest_results_df['portfolio_value'] - initial_value

            ax2.plot(backtest_results_df['timestamp'], portfolio_curve,
                     label='Зміна портфеля', color='blue', linewidth=3)

            final_change = portfolio_curve.iloc[-1]

            if final_change > 0:
                title_color = 'green'
                status = '📈'
                fill_color = 'green'
            else:
                title_color = 'red'
                status = '📉'
                fill_color = 'red'

            ax2.fill_between(backtest_results_df['timestamp'], 0, portfolio_curve,
                             alpha=0.2, color=fill_color)

            ax2.set_title(f'{status} Зміна портфеля: ${final_change:+,.0f}',
                          fontsize=14, fontweight='bold', color=title_color)

        # Базова лінія
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Початок')

        ax2.set_ylabel('Накопичена дохідність (USD)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='upper left')

        # Статистика
        total_trades = len(executed_trades) if not executed_trades.empty else 0
        profitable_count = len(profitable_closes) if not profitable_closes.empty else 0
        win_rate = (profitable_count / len(close_signals) * 100) if len(close_signals) > 0 else 0

        stats_text = f'Трейдів: {total_trades}\nПрибуткових: {profitable_count}\nУспішність: {win_rate:.1f}%'
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=10)

        plt.tight_layout()
        plt.show()

        print("✅ Виправлену візуалізацію створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        import traceback
        print(traceback.format_exc())


def calculate_correct_pnl_curve(backtest_results_df):
    """📊 ПРАВИЛЬНИЙ розрахунок кривої накопиченої дохідності"""
    pnl_curve = []
    running_pnl = 0.0

    for _, row in backtest_results_df.iterrows():
        pnl_this_step = 0

        # Пріоритет 1: Якщо трейд виконано і є PnL
        if row.get('executed', False) and pd.notna(row.get('pnl', 0)):
            pnl_value = row.get('pnl', 0)
            if abs(pnl_value) > 0.01:  # Ігноруємо мінімальні зміни
                pnl_this_step = pnl_value

        # Пріоритет 2: Використовуємо portfolio_change
        elif row.get('executed', False) and pd.notna(row.get('portfolio_change', 0)):
            portfolio_change = row.get('portfolio_change', 0)
            if abs(portfolio_change) > 1:  # Ігноруємо мінімальні зміни
                pnl_this_step = portfolio_change

        running_pnl += pnl_this_step
        pnl_curve.append(running_pnl)

    return pnl_curve

def calculate_trading_metrics(backtest_results):
    """Спрощений розрахунок торгових метрик"""
    try:
        if backtest_results.empty:
            return {'Total Return': 0.0, 'Number of Trades': 0}

        portfolio_values = backtest_results['portfolio_value'].values
        if len(portfolio_values) < 2:
            return {'Total Return': 0.0, 'Number of Trades': len(backtest_results)}

        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value

        trades = len(backtest_results[backtest_results['executed'] == True])

        # Додаткові метрики
        long_trades = len(backtest_results[
                              (backtest_results['signal'] == 'BUY') & (backtest_results['executed'] == True)
                              ])
        short_trades = len(backtest_results[
                               (backtest_results['signal'] == 'SHORT') & (backtest_results['executed'] == True)
                               ])

        # PnL аналіз
        total_pnl = 0
        long_pnl = 0
        short_pnl = 0

        if 'pnl' in backtest_results.columns:
            long_pnl_trades = backtest_results[
                (backtest_results['position_type'] == 'BUY') &
                (backtest_results['pnl'].notna())
                ]
            short_pnl_trades = backtest_results[
                (backtest_results['position_type'] == 'SHORT') &
                (backtest_results['pnl'].notna())
                ]

            long_pnl = long_pnl_trades['pnl'].sum()
            short_pnl = short_pnl_trades['pnl'].sum()
            total_pnl = long_pnl + short_pnl

        return {
            'Total Return': total_return,
            'Initial Value': initial_value,
            'Final Value': final_value,
            'Number of Trades': trades,
            'Long Trades': long_trades,
            'Short Trades': short_trades,
            'Total PnL': total_pnl,
            'Long PnL': long_pnl,
            'Short PnL': short_pnl,
            'total_trades': trades,  # для сумісності
            'short_trades': short_trades,  # для сумісності
            'buy_trades': long_trades,  # для сумісності
            'short_pnl': short_pnl,  # для сумісності
            'long_pnl': long_pnl,  # для сумісності
        }

    except Exception as e:
        print(f"Помилка розрахунку метрик: {e}")
        return {'Total Return': 0.0, 'Number of Trades': 0}


def save_backtest_results(backtest_results_df, trading_stats, filename_prefix="backtest"):
    """Зберігає результати бектесту"""
    try:
        results_dir = CONFIG.get('saving', {}).get('results_dir', 'results')
        ensure_dir(results_dir)

        timestamp = timestamp_to_string()
        results_filename = f"{filename_prefix}_results_{timestamp}.csv"
        results_path = os.path.join(results_dir, results_filename)

        backtest_results_df.to_csv(results_path, index=False)
        print(f"✓ Результати збережено в {results_path}")

        # Збереження статистики окремо
        stats_filename = f"{filename_prefix}_stats_{timestamp}.txt"
        stats_path = os.path.join(results_dir, stats_filename)

        try:
            with open(stats_path, 'w', encoding='utf-8') as f:
                f.write("=== СТАТИСТИКА ТОРГОВОЇ СИСТЕМИ ===\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                for key, value in trading_stats.items():
                    f.write(f"{key}: {value}\n")

                # Додаткова аналітика
                if not backtest_results_df.empty:
                    f.write(f"\n=== АНАЛІТИКА РЕЗУЛЬТАТІВ ===\n")
                    f.write(f"Загальна кількість записів: {len(backtest_results_df)}\n")
                    f.write(f"Виконаних трейдів: {len(backtest_results_df[backtest_results_df['executed'] == True])}\n")

                    initial_portfolio = backtest_results_df['portfolio_value'].iloc[0]
                    final_portfolio = backtest_results_df['portfolio_value'].iloc[-1]
                    total_return_pct = ((final_portfolio - initial_portfolio) / initial_portfolio) * 100

                    f.write(f"Початкова вартість портфеля: ${initial_portfolio:,.2f}\n")
                    f.write(f"Кінцева вартість портфеля: ${final_portfolio:,.2f}\n")
                    f.write(f"Загальна прибутковість: {total_return_pct:+.2f}%\n")

                    # Аналіз сигналів
                    signal_counts = backtest_results_df['signal'].value_counts()
                    f.write(f"\n=== РОЗПОДІЛ СИГНАЛІВ ===\n")
                    for signal, count in signal_counts.items():
                        percentage = (count / len(backtest_results_df)) * 100
                        f.write(f"{signal}: {count} ({percentage:.1f}%)\n")

                    # Аналіз успішності
                    executed_trades = backtest_results_df[backtest_results_df['executed'] == True]
                    if not executed_trades.empty and 'pnl' in executed_trades.columns:
                        profitable_trades = executed_trades[executed_trades['pnl'] > 0]
                        success_rate = (len(profitable_trades) / len(executed_trades)) * 100
                        f.write(f"\n=== УСПІШНІСТЬ ===\n")
                        f.write(f"Прибуткових трейдів: {len(profitable_trades)} з {len(executed_trades)}\n")
                        f.write(f"Відсоток успішності: {success_rate:.1f}%\n")

                        avg_profit = profitable_trades['pnl'].mean() if len(profitable_trades) > 0 else 0
                        avg_loss = executed_trades[executed_trades['pnl'] < 0]['pnl'].mean()
                        f.write(f"Середній прибуток: ${avg_profit:.2f}\n")
                        f.write(f"Середній збиток: ${avg_loss:.2f}\n")

            print(f"✓ Статистику збережено в {stats_path}")

        except Exception as e:
            print(f"⚠️ Помилка збереження статистики: {e}")

        return results_path, stats_path

    except Exception as e:
        print(f"❌ Помилка збереження даних: {e}")
        return None, None


def create_detailed_trade_analysis(backtest_results_df):
    """
    📊 НОВА ФУНКЦІЯ: Створює детальний аналіз торгових результатів
    """
    if backtest_results_df.empty:
        print("❌ Немає даних для аналізу")
        return {}

    print("\n" + "📊" * 30)
    print("ДЕТАЛЬНИЙ АНАЛІЗ ТОРГОВИХ РЕЗУЛЬТАТІВ")
    print("📊" * 30)

    # Базові метрики
    total_records = len(backtest_results_df)
    executed_trades = backtest_results_df[backtest_results_df['executed'] == True]
    total_executed = len(executed_trades)

    print(f"📈 Загальна статистика:")
    print(f"  Загальна кількість записів: {total_records}")
    print(f"  Виконаних трейдів: {total_executed}")
    print(f"  Відсоток виконання: {(total_executed / total_records) * 100:.1f}%")

    # Аналіз по типах сигналів
    signal_analysis = {}
    for signal_type in ['BUY', 'SELL', 'SHORT', 'COVER', 'HOLD']:
        signal_count = len(backtest_results_df[backtest_results_df['signal'] == signal_type])
        executed_count = len(executed_trades[executed_trades['signal'] == signal_type])
        signal_analysis[signal_type] = {
            'total': signal_count,
            'executed': executed_count,
            'execution_rate': (executed_count / signal_count) * 100 if signal_count > 0 else 0
        }

    print(f"\n📊 Аналіз по типах сигналів:")
    for signal, data in signal_analysis.items():
        if data['total'] > 0:
            print(f"  {signal}: {data['total']} сигналів, {data['executed']} виконано ({data['execution_rate']:.1f}%)")

    # Фінансовий аналіз
    if not backtest_results_df.empty:
        initial_value = backtest_results_df['portfolio_value'].iloc[0]
        final_value = backtest_results_df['portfolio_value'].iloc[-1]
        max_value = backtest_results_df['portfolio_value'].max()
        min_value = backtest_results_df['portfolio_value'].min()

        total_return = ((final_value - initial_value) / initial_value) * 100
        max_gain = ((max_value - initial_value) / initial_value) * 100
        max_drawdown = ((min_value - max_value) / max_value) * 100

        print(f"\n💰 Фінансовий аналіз:")
        print(f"  Початкова вартість: ${initial_value:,.2f}")
        print(f"  Кінцева вартість: ${final_value:,.2f}")
        print(f"  Загальна прибутковість: {total_return:+.2f}%")
        print(f"  Максимальний приріст: {max_gain:+.2f}%")
        print(f"  Максимальна просадка: {max_drawdown:.2f}%")

    # Аналіз PnL (якщо доступний)
    if 'pnl' in executed_trades.columns and not executed_trades['pnl'].isna().all():
        profitable_trades = executed_trades[executed_trades['pnl'] > 0]
        losing_trades = executed_trades[executed_trades['pnl'] < 0]

        print(f"\n📈 Аналіз прибутковості трейдів:")
        print(f"  Прибуткових: {len(profitable_trades)}")
        print(f"  Збиткових: {len(losing_trades)}")
        print(f"  Відсоток успіху: {(len(profitable_trades) / len(executed_trades)) * 100:.1f}%")

        if len(profitable_trades) > 0:
            avg_profit = profitable_trades['pnl'].mean()
            max_profit = profitable_trades['pnl'].max()
            print(f"  Середній прибуток: ${avg_profit:.2f}")
            print(f"  Максимальний прибуток: ${max_profit:.2f}")

        if len(losing_trades) > 0:
            avg_loss = losing_trades['pnl'].mean()
            max_loss = losing_trades['pnl'].min()
            print(f"  Середній збиток: ${avg_loss:.2f}")
            print(f"  Максимальний збиток: ${max_loss:.2f}")

    # Аналіз позицій (лонг vs шорт)
    long_trades = executed_trades[executed_trades['signal'] == 'BUY']
    short_trades = executed_trades[executed_trades['signal'] == 'SHORT']

    if len(long_trades) > 0 or len(short_trades) > 0:
        print(f"\n🎯 Аналіз типів позицій:")
        print(f"  Лонг позицій відкрито: {len(long_trades)}")
        print(f"  Шорт позицій відкрито: {len(short_trades)}")

        total_positions = len(long_trades) + len(short_trades)
        if total_positions > 0:
            long_percentage = (len(long_trades) / total_positions) * 100
            short_percentage = (len(short_trades) / total_positions) * 100
            print(f"  Розподіл: {long_percentage:.1f}% лонг, {short_percentage:.1f}% шорт")

    # Технічний аналіз умов торгівлі
    if 'rsi' in executed_trades.columns:
        avg_rsi_buy = executed_trades[executed_trades['signal'] == 'BUY']['rsi'].mean()
        avg_rsi_short = executed_trades[executed_trades['signal'] == 'SHORT']['rsi'].mean()

        print(f"\n🔧 Технічний аналіз:")
        if not pd.isna(avg_rsi_buy):
            print(f"  Середній RSI при BUY: {avg_rsi_buy:.1f}")
        if not pd.isna(avg_rsi_short):
            print(f"  Середній RSI при SHORT: {avg_rsi_short:.1f}")

    # Аналіз впевненості
    if 'confidence' in executed_trades.columns:
        avg_confidence = executed_trades['confidence'].mean()
        max_confidence = executed_trades['confidence'].max()
        min_confidence = executed_trades['confidence'].min()

        print(f"\n🎯 Аналіз впевненості сигналів:")
        print(f"  Середня впевненість: {avg_confidence:.3f}")
        print(f"  Максимальна впевненість: {max_confidence:.3f}")
        print(f"  Мінімальна впевненість: {min_confidence:.3f}")

    print("📊" * 30)

    return {
        'total_records': total_records,
        'total_executed': total_executed,
        'execution_rate': (total_executed / total_records) * 100,
        'signal_analysis': signal_analysis,
        'financial_metrics': {
            'initial_value': initial_value if 'initial_value' in locals() else 0,
            'final_value': final_value if 'final_value' in locals() else 0,
            'total_return': total_return if 'total_return' in locals() else 0,
            'max_drawdown': max_drawdown if 'max_drawdown' in locals() else 0,
        }
    }


# Приклад використання та тестування
if __name__ == '__main__':
    print("🔧 Тестування оновленого модуля бектестингу...")


    # Простий тест
    class MockTradingSystem:
        def __init__(self):
            self.balance = 10000
            self.btc_holdings = 0
            self.initial_balance = 10000
            self.positions = []
            self.risk_params = {
                'stop_loss_pct': 0.05,
                'take_profit_pct': 0.10,
                'trailing_stop_enabled': False,
                'max_open_positions': 1
            }

        def reset_state(self):
            self.balance = self.initial_balance
            self.btc_holdings = 0
            self.positions = []

        def generate_aggressive_signal(self, features, current_price, technical_indicators):
            # Симулюємо різні типи сигналів
            import random
            signals = ['HOLD', 'BUY', 'SHORT', 'SELL']
            signal = random.choice(signals)
            return signal, current_price, 0, 0.5, 'regular'

        def execute_trade_with_costs(self, signal, current_price, timestamp, predicted_price, confidence, signal_type):
            executed = signal != 'HOLD' and random.random() > 0.3  # 70% шанс виконання
            return {
                'executed': executed,
                'portfolio_value': self.balance,
                'balance_after': self.balance,
                'btc_after': self.btc_holdings,
                'borrowed_btc_after': 0,
                'total_cost': 0,
                'fee': 0,
                'slippage': 0,
                'effective_price': current_price,
                'pnl': random.uniform(-100, 200) if executed else 0,
                'position_type': 'BUY' if signal in ['BUY', 'SELL'] else 'SHORT' if signal == 'SHORT' else None
            }

        def get_trading_statistics(self):
            return {'total_trades': 0, 'total_fees_paid': 0}


    # Тестові дані
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
        'close': [50000 + i * 10 + np.random.randint(-500, 500) for i in range(100)],
        'high': [50000 + i * 10 + np.random.randint(0, 1000) for i in range(100)],
        'low': [50000 + i * 10 - np.random.randint(0, 1000) for i in range(100)],
        'rsi': [np.random.randint(20, 80) for _ in range(100)],
        'macd': [np.random.uniform(-50, 50) for _ in range(100)],
        'volume_ratio_20': [np.random.uniform(0.5, 2.0) for _ in range(100)],
        'close_lag_1': [50000 + i * 10 for i in range(100)],
        'sma_20': [50000 + i * 10 for i in range(100)]
    })

    # Тест
    mock_system = MockTradingSystem()
    feature_names = ['close_lag_1', 'sma_20', 'rsi']

    print("🔄 Запуск тестового бектесту...")
    results = run_aggressive_backtest(mock_system, test_data, feature_names)

    if not results.empty:
        print("✅ Тест пройшов успішно!")
        print(f"📊 Оброблено {len(results)} точок")

        # Тест візуалізації
        print("🎨 Тестування візуалізації...")
        plot_backtest_results(results, "Тестова система")

        # Тест детального аналізу
        print("📊 Тестування детального аналізу...")
        analysis = create_detailed_trade_analysis(results)

        # Тест збереження
        print("💾 Тестування збереження...")
        save_path, stats_path = save_backtest_results(results, {'test': 'data'}, "test_backtest")

        print("🎉 Всі тести пройшли успішно!")
    else:
        print("❌ Тест не пройшов - порожні результати")