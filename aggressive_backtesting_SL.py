# aggressive_backtesting_fixed.py - Виправлений модуль бектестингу

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
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


def run_aggressive_backtest(trading_system, historical_data, feature_names_for_model):
    """
    🔧 ВИПРАВЛЕНИЙ бектест з безпечним доступом до атрибутів
    """
    print("⚖️ Запуск бектесту з перевіркою атрибутів...")

    # Безпечно отримуємо параметри ризику
    risk_params = safe_get_risk_params(trading_system)

    print(f"🛡️ Стоп-лос: {risk_params['stop_loss_pct'] * 100:.1f}%")
    print(f"🎯 Тейк-профіт: {risk_params['take_profit_pct'] * 100:.1f}%")
    print(f"📈 Трейлінг стоп: {risk_params.get('trailing_stop_pct', 0.03) * 100:.1f}%")

    # Скидаємо стан системи
    if hasattr(trading_system, 'reset_state'):
        trading_system.reset_state()
    else:
        print("⚠️ Метод reset_state не знайдено")

    results_log = []
    n_points_to_process = min(100, len(historical_data))  # Зменшуємо для тесту
    step = max(1, len(historical_data) // n_points_to_process)

    # Отримуємо початковий баланс безпечно
    if hasattr(trading_system, 'initial_balance'):
        initial_balance_display = trading_system.initial_balance
    elif hasattr(trading_system, 'balance'):
        initial_balance_display = trading_system.balance
    else:
        initial_balance_display = 10000
        print("⚠️ Не вдалося знайти початковий баланс, використовується $10,000")

    signals_generated_non_hold = 0

    # Перевірка наявності необхідних колонок
    required_cols = ['close', 'timestamp'] + feature_names_for_model
    missing_cols = [col for col in required_cols if col not in historical_data.columns]
    if missing_cols:
        print(f"⚠️ Відсутні колонки: {missing_cols}")
        # Додаємо відсутні колонки з базовими значеннями
        for col in missing_cols:
            if col == 'close':
                historical_data[col] = 50000
            elif col == 'timestamp':
                historical_data[col] = pd.date_range('2023-01-01', periods=len(historical_data), freq='H')
            else:
                historical_data[col] = 0

    print(f"📊 ПОЧАТКОВИЙ СТАН ПОРТФЕЛЯ:")
    print(f"   💰 Balance: ${initial_balance_display:,.2f}")
    print(f"📊 Обробляємо {len(historical_data) // step} точок з {len(historical_data)} (крок: {step})")

    for i in range(0, len(historical_data), step):
        try:
            row = historical_data.iloc[i]
            current_price = row['close']
            timestamp = row['timestamp']
            trading_system._current_high = row.get('high', current_price)
            trading_system._current_low = row.get('low', current_price)
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
                    # Простий сигнал якщо метод не знайдено
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

            if signal != 'HOLD':
                signals_generated_non_hold += 1

            # ВИКОНАННЯ ТОРГІВЛІ
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
                    # Базова торгівля якщо метод не знайдено
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
            if hasattr(trading_system, 'positions'):
                open_positions = len(trading_system.positions)

            # Логування результату
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
                'executed': trade_info.get('executed', False),
                'total_cost': trade_info.get('total_cost', 0),
                'fee': trade_info.get('fee', 0),
                'slippage': trade_info.get('slippage', 0),
                'effective_price': trade_info.get('effective_price', current_price),
                'reason_not_executed': trade_info.get('reason', None),
                'portfolio_change': portfolio_change,
                'open_positions': open_positions,
                'stop_loss_price': trade_info.get('stop_loss_price', None),
                'take_profit_price': trade_info.get('take_profit_price', None)
            }

            results_log.append(log_entry)

        except Exception as e:
            print(f"⚠️ Помилка на ітерації {i}: {e}")
            continue

        # Прогрес
        if i > 0 and (i // step) % 20 == 0:
            progress_pct = (i + step) / len(historical_data) * 100
            print(f"🔄 Прогрес: {progress_pct:.1f}% | Сигналів (не HOLD): {signals_generated_non_hold}")

    # Створення результатів
    results_df = pd.DataFrame(results_log)

    # Отримуємо статистику безпечно
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

    # Діагностика
    print(f"\n⚖️ РЕЗУЛЬТАТИ БЕКТЕСТУ:")
    print(f"✅ Сигналів згенеровано (не HOLD): {signals_generated_non_hold}")
    print(f"✅ Трейдів виконано: {final_trading_stats.get('total_trades', 0)}")

    if not results_df.empty:
        final_portfolio = results_df['portfolio_value'].iloc[-1]
        total_return = (final_portfolio - initial_balance_display) / initial_balance_display * 100
        print(f"💰 Початковий капітал: ${initial_balance_display:,.2f}")
        print(f"💰 Кінцева вартість портфеля: ${final_portfolio:,.2f}")
        print(f"💰 Прибутковість стратегії: {total_return:.2f}%")

    return results_df


# В файлі aggressive_backtesting_SL.py замініть функцію plot_backtest_results на цю:

def plot_backtest_results(backtest_results_df, system_name="Торгова система"):
    """
    🔧 ВИПРАВЛЕНА візуалізація - показує ТІЛЬКИ ВИКОНАНІ ТРЕЙДИ замість всіх сигналів
    """
    try:
        if backtest_results_df.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 6))

        # График 1: Ціна та ТІЛЬКИ ВИКОНАНІ трейди
        ax1.plot(backtest_results_df['timestamp'], backtest_results_df['price'],
                 label='Ціна BTC', color='blue', linewidth=1.5)

        # 🔧 КЛЮЧОВЕ ВИПРАВЛЕННЯ: Фільтруємо тільки ВИКОНАНІ трейди
        executed_trades = backtest_results_df[
            (backtest_results_df['executed'] == True) &
            (backtest_results_df['signal'] != 'HOLD')
            ]

        # Розділяємо на BUY та SELL ВИКОНАНІ трейди
        buy_trades = executed_trades[executed_trades['signal'] == 'BUY']
        sell_trades = executed_trades[executed_trades['signal'] == 'SELL']

        # Додаємо інформацію про стоп-лоси та тейк-профіти
        stop_loss_trades = backtest_results_df[
            (backtest_results_df.get('close_reason') == 'stop_loss') |
            (backtest_results_df.get('signal_type') == 'close_stop_loss')
            ]

        take_profit_trades = backtest_results_df[
            (backtest_results_df.get('close_reason') == 'take_profit') |
            (backtest_results_df.get('signal_type') == 'close_take_profit')
            ]

        print(f"📊 Статистика відображених трейдів:")
        print(f"   🟢 BUY виконаних: {len(buy_trades)}")
        print(f"   🔴 SELL виконаних: {len(sell_trades)}")
        print(f"   🛡️ Стоп-лос спрацювань: {len(stop_loss_trades)}")
        print(f"   🎯 Тейк-профіт спрацювань: {len(take_profit_trades)}")
        print(f"   📊 Загалом виконаних: {len(executed_trades)}")

        # Відображаємо тільки виконані BUY трейди
        if not buy_trades.empty:
            ax1.scatter(buy_trades['timestamp'], buy_trades['price'],
                        marker='^', color='green', s=80, label=f'BUY виконано ({len(buy_trades)})',
                        zorder=5, edgecolors='darkgreen', linewidth=1)

        # Відображаємо тільки виконані SELL трейди
        if not sell_trades.empty:
            ax1.scatter(sell_trades['timestamp'], sell_trades['price'],
                        marker='v', color='red', s=80, label=f'SELL виконано ({len(sell_trades)})',
                        zorder=5, edgecolors='darkred', linewidth=1)

        # 🆕 Показуємо стоп-лоси та тейк-профіти спеціальними маркерами
        if not stop_loss_trades.empty:
            ax1.scatter(stop_loss_trades['timestamp'], stop_loss_trades['price'],
                        marker='x', color='orange', s=120, label=f'Стоп-лос ({len(stop_loss_trades)})',
                        zorder=6, linewidth=3)

        if not take_profit_trades.empty:
            ax1.scatter(take_profit_trades['timestamp'], take_profit_trades['price'],
                        marker='+', color='gold', s=120, label=f'Тейк-профіт ({len(take_profit_trades)})',
                        zorder=6, linewidth=3)

        ax1.set_title(f'{system_name}: Ціна та виконані трейди')
        ax1.set_ylabel('Ціна (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # График 2: Динаміка портфеля
        ax2.plot(backtest_results_df['timestamp'], backtest_results_df['portfolio_value'],
                 label='Портфель', color='purple', linewidth=2)

        # Додаємо позначки виконаних трейдів на графік портфеля
        if not executed_trades.empty:
            for _, trade in executed_trades.iterrows():
                portfolio_val = backtest_results_df[
                    backtest_results_df['timestamp'] == trade['timestamp']
                    ]['portfolio_value'].iloc[0]

                if trade['signal'] == 'BUY':
                    ax2.scatter(trade['timestamp'], portfolio_val,
                                marker='^', color='green', s=50, alpha=0.7, zorder=5)
                elif trade['signal'] == 'SELL':
                    ax2.scatter(trade['timestamp'], portfolio_val,
                                marker='v', color='red', s=50, alpha=0.7, zorder=5)

        # Лінія початкового капіталу
        initial_value = backtest_results_df['portfolio_value'].iloc[0]
        ax2.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7,
                    label=f'Початковий: ${initial_value:,.0f}')

        ax2.set_title('Динаміка портфеля')
        ax2.set_ylabel('Вартість (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        # 📊 Розрахунок коефіцієнта виконання
        all_signals = backtest_results_df[backtest_results_df['signal'] != 'HOLD']
        total_signals = len(all_signals)
        execution_rate = len(executed_trades) / total_signals * 100 if total_signals > 0 else 0

        print(f"\n📈 Підсумкова статистика:")
        print(f"   📊 Всього сигналів згенеровано: {total_signals}")
        print(f"   ✅ Фактично виконано трейдів: {len(executed_trades)}")
        print(f"   📈 Коефіцієнт виконання: {execution_rate:.1f}%")

        if execution_rate < 50:
            print("   ⚠️ Низький коефіцієнт виконання - потрібно переглянути умови торгівлі")
        elif execution_rate > 80:
            print("   ✅ Високий коефіцієнт виконання - система працює ефективно")

        print("✅ Візуалізацію створено (тільки виконані трейди)")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        import traceback
        traceback.print_exc()


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

        return {
            'Total Return': total_return,
            'Initial Value': initial_value,
            'Final Value': final_value,
            'Number of Trades': trades
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

        return results_path, None

    except Exception as e:
        print(f"❌ Помилка збереження: {e}")
        return None, None


# Приклад використання та тестування
if __name__ == '__main__':
    print("🔧 Тестування виправленого модуля бектестингу...")


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
            return 'HOLD', current_price, 0, 0, 'regular'

        def execute_trade_with_costs(self, signal, current_price, timestamp, predicted_price, confidence, signal_type):
            return {
                'executed': False,
                'portfolio_value': self.balance,
                'balance_after': self.balance,
                'btc_after': self.btc_holdings,
                'total_cost': 0,
                'fee': 0,
                'slippage': 0,
                'effective_price': current_price
            }

        def get_trading_statistics(self):
            return {'total_trades': 0, 'total_fees_paid': 0}


    # Тестові дані
    test_data = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=10, freq='H'),
        'close': [50000 + i * 10 for i in range(10)],
        'rsi': [50] * 10,
        'macd': [0] * 10,
        'volume_ratio_20': [1.0] * 10,
        'close_lag_1': [50000] * 10,
        'sma_20': [50000] * 10
    })

    # Тест
    mock_system = MockTradingSystem()
    feature_names = ['close_lag_1', 'sma_20', 'rsi']

    results = run_aggressive_backtest(mock_system, test_data, feature_names)

    if not results.empty:
        print("✅ Тест пройшов успішно!")
        print(f"Оброблено {len(results)} точок")
    else:
        print("❌ Тест не пройшов")