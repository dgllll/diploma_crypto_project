# test_pnl_visualization.py - Тестовий файл для перевірки нової візуалізації кривої дохідності

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import traceback

# ВИПРАВЛЕНІ ІМПОРТИ - використовуємо функції безпосередньо в цьому файлі
# або переконайтеся, що вони додані до aggressive_backtesting_SL_STOP.py

def calculate_cumulative_pnl_curve(backtest_results_df):
    """📊 Розраховує криву накопиченої дохідності від виконаних трейдів"""
    cumulative_pnl = []
    running_total = 0.0

    for _, row in backtest_results_df.iterrows():
        # Додаємо PnL тільки від виконаних трейдів з реальним PnL
        if (row.get('executed', False) and
                pd.notna(row.get('pnl', 0)) and
                row.get('pnl', 0) != 0):
            running_total += row['pnl']
            print(f"🔄 Трейд: {row.get('signal', 'N/A')} PnL: ${row['pnl']:+.0f} | Накопичено: ${running_total:+.0f}")

        cumulative_pnl.append(running_total)

    print(f"📊 Побудовано криву з {len([p for p in cumulative_pnl if p != 0])} точками змін")
    return cumulative_pnl


def plot_backtest_results_with_pnl_curve(backtest_results_df, system_name="Торгова система"):
    """🔧 ВИПРАВЛЕНА візуалізація з кривою накопиченої дохідності"""
    try:
        if backtest_results_df.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(18, 8))

        # График 1: ТІЛЬКИ ТРЕЙДИ (без всіх сигналів)
        ax1.plot(backtest_results_df['timestamp'], backtest_results_df['price'],
                 label='Ціна BTC', color='blue', linewidth=2, alpha=0.8)

        # 🔧 ВИПРАВЛЕНО: Показуємо ТІЛЬКИ виконані трейди
        executed_trades = backtest_results_df[backtest_results_df['executed'] == True]

        # Розділяємо по типах виконаних трейдів
        buy_trades = executed_trades[executed_trades['signal'] == 'BUY']
        sell_trades = executed_trades[executed_trades['signal'] == 'SELL']
        short_open = executed_trades[executed_trades['signal'] == 'SHORT']
        short_close = executed_trades[
            (executed_trades['signal'] == 'COVER') |
            (executed_trades['signal_type'].str.contains('close_', na=False) &
             executed_trades['position_type'].eq('SHORT'))
            ]

        # Додаємо тільки виконані трейди
        if not buy_trades.empty:
            ax1.scatter(buy_trades['timestamp'], buy_trades['price'],
                        marker='^', color='green', s=100, label=f'🟢 BUY ({len(buy_trades)})',
                        zorder=5, alpha=0.9, edgecolors='white', linewidth=1)

        if not sell_trades.empty:
            ax1.scatter(sell_trades['timestamp'], sell_trades['price'],
                        marker='v', color='red', s=100, label=f'🔴 SELL ({len(sell_trades)})',
                        zorder=5, alpha=0.9, edgecolors='white', linewidth=1)

        if not short_open.empty:
            ax1.scatter(short_open['timestamp'], short_open['price'],
                        marker='s', color='purple', s=120, label=f'🟣 SHORT OPEN ({len(short_open)})',
                        zorder=6, alpha=0.9, edgecolors='white', linewidth=2)

        if not short_close.empty:
            ax1.scatter(short_close['timestamp'], short_close['price'],
                        marker='X', color='orange', s=140, label=f'🟠 SHORT CLOSE ({len(short_close)})',
                        zorder=6, alpha=0.9, edgecolors='white', linewidth=2)

        ax1.set_title(f'{system_name}: Ціна та ВИКОНАНІ трейди', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize=10)

        # Статистика на графіку 1
        total_executed = len(executed_trades)
        long_trades = len(buy_trades)
        short_trades = len(short_open)

        stats_text = f'Виконано: {total_executed}\nЛонг: {long_trades}\nШорт: {short_trades}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                 fontsize=10)

        # 🆕 График 2: КРИВА НАКОПИЧЕНОЇ ДОХІДНОСТІ ВІД ТРЕЙДІВ
        cumulative_pnl = calculate_cumulative_pnl_curve(backtest_results_df)

        if len(cumulative_pnl) > 0:
            ax2.plot(backtest_results_df['timestamp'], cumulative_pnl,
                     label='Накопичена дохідність', color='green', linewidth=3)

            # 🆕 ДОДАЄМО МАРКЕРИ для великих PnL подій
            trades_with_pnl = backtest_results_df[
                (backtest_results_df['executed'] == True) &
                (backtest_results_df['pnl'].notna()) &
                (backtest_results_df['pnl'] != 0)
                ]

            # Великі прибутки (>$500)
            big_profits = trades_with_pnl[trades_with_pnl['pnl'] > 500]
            if not big_profits.empty:
                profit_pnl = []
                for _, trade in big_profits.iterrows():
                    trade_idx = backtest_results_df[backtest_results_df['timestamp'] <= trade['timestamp']].index
                    if len(trade_idx) > 0:
                        profit_pnl.append(cumulative_pnl[trade_idx[-1]])

                if profit_pnl:
                    ax2.scatter(big_profits['timestamp'], profit_pnl,
                                marker='^', color='darkgreen', s=120, alpha=0.8,
                                label=f'📈 Великі прибутки (${big_profits["pnl"].sum():.0f})', zorder=5)

            # Великі збитки (<-$300)
            big_losses = trades_with_pnl[trades_with_pnl['pnl'] < -300]
            if not big_losses.empty:
                loss_pnl = []
                for _, trade in big_losses.iterrows():
                    trade_idx = backtest_results_df[backtest_results_df['timestamp'] <= trade['timestamp']].index
                    if len(trade_idx) > 0:
                        loss_pnl.append(cumulative_pnl[trade_idx[-1]])

                if loss_pnl:
                    ax2.scatter(big_losses['timestamp'], loss_pnl,
                                marker='v', color='darkred', s=120, alpha=0.8,
                                label=f'📉 Великі збитки (${big_losses["pnl"].sum():.0f})', zorder=5)

            # Початкова лінія (0)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Початок (0)')

            # Фінальне значення та прибутковість
            final_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0

            # Кольорове заповнення для прибутку/збитку
            if final_pnl > 0:
                ax2.fill_between(backtest_results_df['timestamp'], 0, cumulative_pnl,
                                 alpha=0.2, color='green', label='Накопичений прибуток')
                title_color = 'green'
                status = '📈'
            else:
                ax2.fill_between(backtest_results_df['timestamp'], 0, cumulative_pnl,
                                 alpha=0.2, color='red', label='Накопичений збиток')
                title_color = 'red'
                status = '📉'

            ax2.set_title(f'{status} Крива дохідності: ${final_pnl:+,.0f}',
                          fontsize=14, fontweight='bold', color=title_color)

            # 🆕 АНОТАЦІЇ для максимального та мінімального PnL
            max_pnl = max(cumulative_pnl)
            min_pnl = min(cumulative_pnl)

            if max_pnl > 0:
                max_idx = cumulative_pnl.index(max_pnl)
                max_time = backtest_results_df.iloc[max_idx]['timestamp']
                ax2.annotate(f'Пік\n${max_pnl:,.0f}',
                             xy=(max_time, max_pnl), xytext=(10, 15),
                             textcoords='offset points', ha='left',
                             bbox=dict(boxstyle='round,pad=0.3', fc='lightgreen', alpha=0.8),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                             fontsize=9)

            if min_pnl < -100:  # Показуємо тільки суттєві просадки
                min_idx = cumulative_pnl.index(min_pnl)
                min_time = backtest_results_df.iloc[min_idx]['timestamp']
                ax2.annotate(f'Просадка\n${min_pnl:,.0f}',
                             xy=(min_time, min_pnl), xytext=(10, -25),
                             textcoords='offset points', ha='left',
                             bbox=dict(boxstyle='round,pad=0.3', fc='lightcoral', alpha=0.8),
                             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                             fontsize=9)

        else:
            ax2.text(0.5, 0.5, 'Немає даних PnL\nдля побудови кривої',
                     ha='center', va='center', transform=ax2.transAxes,
                     fontsize=14, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax2.set_ylabel('Накопичена дохідність (USD)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='upper left')

        plt.tight_layout()
        plt.show()

        print("✅ Виправлену візуалізацію з кривою накопиченої дохідності створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        print(traceback.format_exc())


def analyze_pnl_curve_performance(backtest_results_df):
    """📊 Детальний аналіз продуктивності на основі кривої PnL"""
    print("\n" + "📊" * 30)
    print("АНАЛІЗ КРИВОЇ ДОХІДНОСТІ")
    print("📊" * 30)

    # Отримуємо всі трейди з PnL
    trades_with_pnl = backtest_results_df[
        (backtest_results_df['executed'] == True) &
        (backtest_results_df['pnl'].notna()) &
        (backtest_results_df['pnl'] != 0)
        ].copy()

    if trades_with_pnl.empty:
        print("❌ Немає трейдів з PnL для аналізу")
        return

    total_trades = len(trades_with_pnl)
    winning_trades = trades_with_pnl[trades_with_pnl['pnl'] > 0]
    losing_trades = trades_with_pnl[trades_with_pnl['pnl'] < 0]

    total_pnl = trades_with_pnl['pnl'].sum()
    win_rate = len(winning_trades) / total_trades * 100

    print(f"📈 Загальна статистика:")
    print(f"   Всього трейдів: {total_trades}")
    print(f"   Прибуткових: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"   Збиткових: {len(losing_trades)} ({100 - win_rate:.1f}%)")
    print(f"   Загальний PnL: ${total_pnl:+,.0f}")

    if len(winning_trades) > 0:
        avg_win = winning_trades['pnl'].mean()
        max_win = winning_trades['pnl'].max()
        total_wins = winning_trades['pnl'].sum()
        print(f"\n💰 Прибуткові трейди:")
        print(f"   Середній прибуток: ${avg_win:+.0f}")
        print(f"   Максимальний прибуток: ${max_win:+.0f}")
        print(f"   Загальний прибуток: ${total_wins:+.0f}")

    if len(losing_trades) > 0:
        avg_loss = losing_trades['pnl'].mean()
        max_loss = losing_trades['pnl'].min()
        total_losses = losing_trades['pnl'].sum()
        print(f"\n📉 Збиткові трейди:")
        print(f"   Середній збиток: ${avg_loss:.0f}")
        print(f"   Максимальний збиток: ${max_loss:.0f}")
        print(f"   Загальні збитки: ${total_losses:.0f}")

        # Коефіцієнт прибуток/збиток
        if len(winning_trades) > 0:
            profit_factor = abs(total_wins / total_losses) if total_losses != 0 else float('inf')
            print(f"\n⚖️ Коефіцієнт прибуток/збиток: {profit_factor:.2f}")

            if profit_factor > 2.0:
                print("   ✅ Відмінний коефіцієнт!")
            elif profit_factor > 1.5:
                print("   ✅ Хороший коефіцієнт")
            elif profit_factor > 1.0:
                print("   ⚠️ Помірний коефіцієнт")
            else:
                print("   ❌ Низький коефіцієнт")

    # Аналіз по типах трейдів
    print(f"\n🔍 Аналіз по типах позицій:")

    long_pnl = trades_with_pnl[
        trades_with_pnl['position_type'].eq('BUY') |
        trades_with_pnl['signal'].eq('SELL')
        ]['pnl'].sum()

    short_pnl = trades_with_pnl[
        trades_with_pnl['position_type'].eq('SHORT') |
        trades_with_pnl['signal'].eq('COVER')
        ]['pnl'].sum()

    print(f"   🟢 Лонг позиції PnL: ${long_pnl:+.0f}")
    print(f"   🔴 Шорт позиції PnL: ${short_pnl:+.0f}")

    # Найкращі та найгірші трейди
    print(f"\n🏆 ТОП-3 НАЙКРАЩІ ТРЕЙДИ:")
    top_trades = winning_trades.nlargest(3, 'pnl')[['timestamp', 'signal', 'price', 'pnl']]
    for i, (_, trade) in enumerate(top_trades.iterrows(), 1):
        time_str = trade['timestamp'].strftime('%m-%d %H:%M') if hasattr(trade['timestamp'], 'strftime') else str(
            trade['timestamp'])
        print(f"   {i}. {time_str}: {trade['signal']} @ ${trade['price']:.0f} → PnL: ${trade['pnl']:+.0f}")

    print(f"\n💸 ТОП-3 НАЙГІРШІ ТРЕЙДИ:")
    worst_trades = losing_trades.nsmallest(3, 'pnl')[['timestamp', 'signal', 'price', 'pnl']]
    for i, (_, trade) in enumerate(worst_trades.iterrows(), 1):
        time_str = trade['timestamp'].strftime('%m-%d %H:%M') if hasattr(trade['timestamp'], 'strftime') else str(
            trade['timestamp'])
        print(f"   {i}. {time_str}: {trade['signal']} @ ${trade['price']:.0f} → PnL: ${trade['pnl']:+.0f}")

    print("📊" * 30)

    return {
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'avg_win': avg_win if len(winning_trades) > 0 else 0,
        'avg_loss': avg_loss if len(losing_trades) > 0 else 0,
        'profit_factor': profit_factor if 'profit_factor' in locals() else 0,
        'long_pnl': long_pnl,
        'short_pnl': short_pnl
    }


# Імпорти НЕ потрібні, оскільки функції вже визначені вище


def create_test_backtest_data():
    """Створює тестові дані для перевірки візуалізації кривої PnL"""
    print("🔧 Створення тестових даних бектесту з реалістичними PnL...")

    # Створюємо базові дані
    n_points = 100
    start_date = datetime.now() - timedelta(hours=n_points)

    # Генеруємо ціни з трендом
    base_price = 50000
    prices = []
    for i in range(n_points):
        # Додаємо тренд + шум
        trend = i * 5  # Зростаючий тренд
        noise = random.uniform(-200, 200)
        price = base_price + trend + noise
        prices.append(max(price, 45000))  # Мінімум $45k

    timestamps = [start_date + timedelta(hours=i) for i in range(n_points)]

    # Створюємо базовий DataFrame
    test_data = pd.DataFrame({
        'timestamp': timestamps,
        'price': prices,
        'executed': [False] * n_points,
        'signal': ['HOLD'] * n_points,
        'signal_type': ['regular'] * n_points,
        'pnl': [0.0] * n_points,
        'position_type': [None] * n_points,
        'portfolio_value': [100000] * n_points,  # Початковий баланс
    })

    # 🔥 ДОДАЄМО РЕАЛІСТИЧНІ ТРЕЙДИ З PnL
    trade_indices = [10, 15, 25, 30, 40, 45, 55, 65, 70, 80, 85, 95]  # Індекси трейдів

    trade_scenarios = [
        # (signal, position_type, pnl, description)
        ('BUY', 'BUY', 0, 'Відкриття лонг'),
        ('SELL', 'BUY', 1200, 'Закриття лонг з прибутком'),
        ('SHORT', 'SHORT', 0, 'Відкриття шорт'),
        ('COVER', 'SHORT', -800, 'Закриття шорт зі збитком'),
        ('BUY', 'BUY', 0, 'Відкриття лонг'),
        ('SELL', 'BUY', 2100, 'Великий прибуток лонг'),
        ('SHORT', 'SHORT', 0, 'Відкриття шорт'),
        ('COVER', 'SHORT', 1500, 'Прибуток від шорт'),
        ('BUY', 'BUY', 0, 'Відкриття лонг'),
        ('SELL', 'BUY', -1100, 'Збиток лонг (стоп-лос)'),
        ('SHORT', 'SHORT', 0, 'Відкриття шорт'),
        ('COVER', 'SHORT', 800, 'Невеликий прибуток шорт'),
    ]

    for i, trade_idx in enumerate(trade_indices):
        if i < len(trade_scenarios):
            signal, pos_type, pnl, desc = trade_scenarios[i]

            test_data.loc[trade_idx, 'executed'] = True
            test_data.loc[trade_idx, 'signal'] = signal
            test_data.loc[trade_idx, 'position_type'] = pos_type
            test_data.loc[trade_idx, 'pnl'] = pnl

            # Додаємо тип сигналу
            if 'стоп-лос' in desc:
                test_data.loc[trade_idx, 'signal_type'] = 'close_stop_loss'
            elif 'прибуток' in desc and 'Великий' in desc:
                test_data.loc[trade_idx, 'signal_type'] = 'close_take_profit'
            elif signal in ['BUY', 'SHORT']:
                test_data.loc[trade_idx, 'signal_type'] = 'strong'
            else:
                test_data.loc[trade_idx, 'signal_type'] = 'close_long'

            print(f"  Трейд {i + 1}: {desc} - PnL: ${pnl:+}")

    print(f"✅ Створено {len(test_data)} записів з {len(trade_indices)} трейдами")

    # Підраховуємо загальну статистику
    total_pnl = test_data[test_data['executed'] == True]['pnl'].sum()
    executed_count = len(test_data[test_data['executed'] == True])
    print(f"📊 Загальний PnL: ${total_pnl:+}")
    print(f"📊 Виконано трейдів: {executed_count}")

    return test_data


def test_pnl_curve_visualization():
    """Тестує нову візуалізацію кривої PnL"""
    print("\n" + "🧪" * 30)
    print("ТЕСТ НОВОЇ ВІЗУАЛІЗАЦІЇ КРИВОЇ PnL")
    print("🧪" * 30)

    # Створюємо тестові дані
    test_data = create_test_backtest_data()

    print("\n📊 Тестування нової функції візуалізації...")

    # Тестуємо нову візуалізацію
    try:
        plot_backtest_results_with_pnl_curve(test_data, "🧪 ТЕСТОВА система з PnL кривою")
        print("✅ Візуалізація успішно створена!")
    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        import traceback
        print(traceback.format_exc())

    print("\n📊 Тестування аналізу продуктивності...")

    # Тестуємо аналіз продуктивності
    try:
        performance_stats = analyze_pnl_curve_performance(test_data)
        print("✅ Аналіз продуктивності завершено!")

        if performance_stats:
            print(f"\n🎯 Результати аналізу:")
            print(f"   Загальний PnL: ${performance_stats.get('total_pnl', 0):+,.0f}")
            print(f"   Відсоток перемог: {performance_stats.get('win_rate', 0):.1f}%")
            print(f"   Коефіцієнт прибуток/збиток: {performance_stats.get('profit_factor', 0):.2f}")

    except Exception as e:
        print(f"❌ Помилка аналізу: {e}")
        import traceback
        print(traceback.format_exc())

    print("\n🧪" * 30)
    print("ТЕСТ ЗАВЕРШЕНО")
    print("🧪" * 30)


def compare_old_vs_new_visualization():
    """Порівнює стару та нову візуалізацію"""
    print("\n" + "⚖️" * 25)
    print("ПОРІВНЯННЯ СТАРОЇ ТА НОВОЇ ВІЗУАЛІЗАЦІЇ")
    print("⚖️" * 25)

    test_data = create_test_backtest_data()

    # Додаємо portfolio_value як в старій візуалізації
    portfolio_values = []
    current_balance = 100000

    for _, row in test_data.iterrows():
        if row['executed'] and pd.notna(row['pnl']):
            current_balance += row['pnl']
        portfolio_values.append(current_balance)

    test_data['portfolio_value'] = portfolio_values

    print("📊 Створюємо СТАРУ візуалізацію (загальна вартість портфеля)...")

    # Стара візуалізація - загальна вартість портфеля
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Стара версія - вартість портфеля
        ax1.plot(test_data['timestamp'], test_data['portfolio_value'],
                 label='Стара: Вартість портфеля', color='blue', linewidth=2)
        ax1.set_title('СТАРА ВІЗУАЛІЗАЦІЯ\n(Загальна вартість портфеля)', fontweight='bold')
        ax1.set_ylabel('Вартість портфеля (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Нова версія - накопичена дохідність
        from aggressive_backtesting_SL_STOP import calculate_cumulative_pnl_curve
        cumulative_pnl = calculate_cumulative_pnl_curve(test_data)

        ax2.plot(test_data['timestamp'], cumulative_pnl,
                 label='Нова: Накопичена дохідність', color='green', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7, label='Початок (0)')
        ax2.set_title('НОВА ВІЗУАЛІЗАЦІЯ\n(Накопичена дохідність від трейдів)', fontweight='bold')
        ax2.set_ylabel('Накопичена дохідність (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Заповнення для наочності
        final_pnl = cumulative_pnl[-1] if cumulative_pnl else 0
        if final_pnl > 0:
            ax2.fill_between(test_data['timestamp'], 0, cumulative_pnl, alpha=0.2, color='green')
        else:
            ax2.fill_between(test_data['timestamp'], 0, cumulative_pnl, alpha=0.2, color='red')

        plt.tight_layout()
        plt.show()

        print("✅ Порівняння створено!")
        print(f"\n📊 РІЗНИЦЯ:")
        print(f"   Стара крива (фінальне значення): ${test_data['portfolio_value'].iloc[-1]:,.0f}")
        print(f"   Нова крива (фінальне значення): ${final_pnl:+,.0f}")
        print(f"   Різниця: ${test_data['portfolio_value'].iloc[-1] - 100000 - final_pnl:,.0f}")
        print(f"\n💡 ПОЯСНЕННЯ:")
        print(f"   Стара крива показує загальну вартість портфеля (баланс + активи)")
        print(f"   Нова крива показує ТІЛЬКИ дохідність від трейдів (сума PnL)")
        print(f"   Нова крива краще відображає ефективність торгової стратегії!")

    except Exception as e:
        print(f"❌ Помилка порівняння: {e}")
        import traceback
        print(traceback.format_exc())

    print("⚖️" * 25)


if __name__ == "__main__":
    print("🚀 ЗАПУСК ТЕСТУВАННЯ НОВОЇ ВІЗУАЛІЗАЦІЇ")
    print("=" * 50)

    # Основний тест нової візуалізації
    test_pnl_curve_visualization()

    # Порівняння з старою візуалізацією
    compare_old_vs_new_visualization()

    print(f"\n✅ ВСІ ТЕСТИ ЗАВЕРШЕНО!")
    print(f"💡 Тепер ви можете використовувати нові функції:")
    print(f"   - plot_backtest_results_with_pnl_curve()")
    print(f"   - analyze_pnl_curve_performance()")
    print(f"   - calculate_cumulative_pnl_curve()")