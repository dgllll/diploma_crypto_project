import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def backtest_strategy(strategy, historical_data, feature_names):
    """
    Проводить бектест стратегії на історичних даних

    Parameters:
    -----------
    strategy : CryptoTradingStrategy
        Торгова стратегія
    historical_data : pandas.DataFrame
        Історичні дані з цінами та ознаками
    feature_names : list
        Список назв ознак для моделі

    Returns:
    --------
    pandas.DataFrame
        Результати бектесту
    """
    print("Запуск бектесту торгової стратегії...")

    try:
        # Перевірка вхідних даних
        if historical_data.empty:
            print("❌ Історичні дані порожні")
            return create_empty_backtest_result()

        # Перевіряємо feature_names
        if feature_names is None or len(feature_names) == 0:
            print("⚠️ feature_names порожні, використовуємо альтернативний підхід")
            return simple_backtest_strategy(strategy, historical_data)

        # Фільтруємо feature_names від None значень
        valid_feature_names = [name for name in feature_names if name is not None and name in historical_data.columns]

        if len(valid_feature_names) == 0:
            print("⚠️ Немає валідних назв ознак, використовуємо спрощений бектест")
            return simple_backtest_strategy(strategy, historical_data)

        print(f"Використовуємо {len(valid_feature_names)} ознак для бектесту")

        # Підготовка даних для бектесту
        backtest_results = []

        # Перевіряємо наявність необхідних колонок
        required_columns = ['close', 'timestamp']
        missing_columns = [col for col in required_columns if col not in historical_data.columns]

        if missing_columns:
            print(f"❌ Відсутні необхідні колонки: {missing_columns}")
            return create_empty_backtest_result()

        # Початкові параметри стратегії
        initial_balance = strategy.balance
        initial_btc = strategy.btc_holdings

        # Ітерація по даних (пропускаємо останні рядки для прогнозування)
        data_length = len(historical_data)
        max_iterations = min(500, data_length - 10)  # Обмежуємо кількість ітерацій

        print(f"Проведення бектесту на {max_iterations} точках...")

        for i in range(0, max_iterations, 5):  # Кожна 5-та точка для швидкості
            try:
                # Отримання поточних даних
                current_row = historical_data.iloc[i]

                # Підготовка ознак для моделі
                # Використовуємо тільки ті ознаки, які існують
                available_features = {}
                for feature_name in valid_feature_names:
                    if feature_name in current_row:
                        available_features[feature_name] = current_row[feature_name]

                # Створюємо DataFrame з ознаками
                if available_features:
                    features_df = pd.DataFrame([available_features])
                else:
                    # Якщо немає ознак, створюємо порожній DataFrame
                    features_df = pd.DataFrame(np.zeros((1, len(valid_feature_names))),
                                               columns=valid_feature_names)

                # Поточна ціна
                current_price = current_row['close']

                # Технічні індикатори (з значеннями за замовчуванням, якщо відсутні)
                technical_indicators = {
                    'rsi': current_row.get('rsi', 50),
                    'macd': current_row.get('macd_12_26', 0),
                    'macd_signal': current_row.get('macd_signal_12_26', 0),
                    'adx': current_row.get('adx', 20)
                }

                # Генерація сигналу
                signal, predicted_price, price_change_pct = strategy.generate_signal(
                    features_df, current_price, technical_indicators
                )

                # Виконання торгівлі
                trade_info = strategy.execute_trade(
                    signal, current_price, current_row['timestamp'], predicted_price
                )

                # Наступна ціна (для розрахунку реальної продуктивності)
                next_price = historical_data.iloc[min(i + 1, data_length - 1)]['close']

                # Додавання результату до списку
                backtest_results.append({
                    'timestamp': current_row['timestamp'],
                    'price': current_price,
                    'next_price': next_price,
                    'predicted_price': predicted_price,
                    'price_change_pct': price_change_pct,
                    'signal': signal,
                    'portfolio_value': trade_info['portfolio_value'],
                    'balance': trade_info['balance_after'],
                    'btc_holdings': trade_info['btc_after']
                })

            except Exception as e:
                print(f"⚠️ Помилка на ітерації {i}: {e}")
                continue

        # Створення датафрейму результатів
        if backtest_results:
            results_df = pd.DataFrame(backtest_results)

            # Розрахунок метрик ефективності
            print_backtest_summary(results_df, initial_balance, initial_btc)

            return results_df
        else:
            print("❌ Не вдалося створити результати бектесту")
            return create_empty_backtest_result()

    except Exception as e:
        print(f"❌ Критична помилка в бектесті: {e}")
        print("Використовуємо спрощений бектест...")
        return simple_backtest_strategy(strategy, historical_data)


def simple_backtest_strategy(strategy, historical_data):
    """Спрощений бектест без використання ML моделі"""
    print("Запуск спрощеного бектесту...")

    try:
        results = []
        n_points = min(100, len(historical_data))
        step = max(1, len(historical_data) // n_points)

        initial_balance = strategy.balance
        current_balance = initial_balance
        btc_holdings = 0

        for i in range(0, len(historical_data), step):
            if i >= len(historical_data):
                break

            row = historical_data.iloc[i]
            current_price = row['close']

            # Простий сигнал на основі технічних індикаторів
            rsi = row.get('rsi', 50)
            signal = 'HOLD'

            if rsi < 30:  # Перепроданість
                signal = 'BUY'
            elif rsi > 70:  # Перекупленість
                signal = 'SELL'

            # Симуляція торгівлі
            if signal == 'BUY' and current_balance > 0:
                btc_to_buy = (current_balance * 0.2) / current_price
                btc_holdings += btc_to_buy
                current_balance -= btc_to_buy * current_price
            elif signal == 'SELL' and btc_holdings > 0:
                btc_to_sell = btc_holdings * 0.2
                current_balance += btc_to_sell * current_price
                btc_holdings -= btc_to_sell

            portfolio_value = current_balance + (btc_holdings * current_price)

            results.append({
                'timestamp': row['timestamp'],
                'price': current_price,
                'next_price': current_price,  # Заглушка
                'predicted_price': current_price,  # Заглушка
                'price_change_pct': 0,
                'signal': signal,
                'portfolio_value': portfolio_value,
                'balance': current_balance,
                'btc_holdings': btc_holdings
            })

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            print_backtest_summary(results_df, initial_balance, 0)

        return results_df

    except Exception as e:
        print(f"❌ Помилка в спрощеному бектесті: {e}")
        return create_empty_backtest_result()


def create_empty_backtest_result():
    """Створює порожній результат бектесту"""
    return pd.DataFrame({
        'timestamp': [pd.Timestamp.now()],
        'price': [50000],
        'next_price': [50000],
        'predicted_price': [50000],
        'price_change_pct': [0],
        'signal': ['HOLD'],
        'portfolio_value': [10000],
        'balance': [10000],
        'btc_holdings': [0]
    })


def print_backtest_summary(results_df, initial_balance, initial_btc):
    """Виводить підсумок бектесту"""
    try:
        if results_df.empty:
            print("Немає даних для підсумку")
            return

        final_portfolio = results_df['portfolio_value'].iloc[-1]
        total_return = (final_portfolio - initial_balance) / initial_balance * 100

        # Кількість угод
        buy_signals = len(results_df[results_df['signal'] == 'BUY'])
        sell_signals = len(results_df[results_df['signal'] == 'SELL'])

        print(f"\n📊 Підсумок бектесту:")
        print(f"Початковий капітал: ${initial_balance:,.2f}")
        print(f"Кінцева вартість портфеля: ${final_portfolio:,.2f}")
        print(f"Загальна доходність: {total_return:.2f}%")
        print(f"Кількість BUY сигналів: {buy_signals}")
        print(f"Кількість SELL сигналів: {sell_signals}")
        print(f"Загальна кількість точок: {len(results_df)}")

    except Exception as e:
        print(f"Помилка в підсумку: {e}")


def plot_backtest_results(backtest_results):
    """Візуалізує результати бектесту"""
    try:
        if backtest_results.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Графік 1: Ціна та сигнали
        ax1.plot(backtest_results['timestamp'], backtest_results['price'],
                 label='Ціна BTC', color='blue', linewidth=2)

        # Додавання сигналів на графік
        buy_signals = backtest_results[backtest_results['signal'] == 'BUY']
        sell_signals = backtest_results[backtest_results['signal'] == 'SELL']

        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['price'],
                        marker='^', color='green', s=100, label='BUY', zorder=5)

        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['price'],
                        marker='v', color='red', s=100, label='SELL', zorder=5)

        ax1.set_title('Ціна Bitcoin та торгові сигнали', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Графік 2: Вартість портфеля
        ax2.plot(backtest_results['timestamp'], backtest_results['portfolio_value'],
                 label='Вартість портфеля', color='purple', linewidth=2)

        # Buy & Hold для порівняння
        if len(backtest_results) > 1:
            initial_price = backtest_results['price'].iloc[0]
            initial_portfolio = backtest_results['portfolio_value'].iloc[0]
            buy_hold_values = initial_portfolio * (backtest_results['price'] / initial_price)
            ax2.plot(backtest_results['timestamp'], buy_hold_values,
                     label='Buy & Hold', color='gray', linestyle='--', linewidth=2)

        ax2.set_title('Динаміка вартості портфеля', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Дата')
        ax2.set_ylabel('Вартість (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

        print("✓ Візуалізацію бектесту створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації бектесту: {e}")


def calculate_trading_metrics(backtest_results):
    """Розраховує метрики ефективності торгової стратегії"""
    try:
        if backtest_results.empty or 'portfolio_value' not in backtest_results.columns:
            return {
                'Total Return': 0.0,
                'Initial Value': 10000,
                'Final Value': 10000,
                'Number of Trades': 0,
                'Max Drawdown': 0.0,
                'Sharpe Ratio': 0.0
            }

        portfolio_values = backtest_results['portfolio_value'].values

        if len(portfolio_values) < 2:
            return {
                'Total Return': 0.0,
                'Initial Value': portfolio_values[0] if len(portfolio_values) > 0 else 10000,
                'Final Value': portfolio_values[0] if len(portfolio_values) > 0 else 10000,
                'Number of Trades': len(backtest_results),
                'Max Drawdown': 0.0,
                'Sharpe Ratio': 0.0
            }

        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value

        # Розрахунок максимальної просадки
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Розрахунок коефіцієнта Шарпа (спрощений)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Анналізований
        else:
            sharpe_ratio = 0.0

        # Підрахунок угод
        trades = len(backtest_results[backtest_results['signal'].isin(['BUY', 'SELL'])])

        return {
            'Total Return': total_return,
            'Initial Value': initial_value,
            'Final Value': final_value,
            'Number of Trades': trades,
            'Max Drawdown': max_drawdown,
            'Sharpe Ratio': sharpe_ratio
        }

    except Exception as e:
        print(f"Помилка розрахунку метрик: {e}")
        return {
            'Total Return': 0.0,
            'Initial Value': 10000,
            'Final Value': 10000,
            'Number of Trades': 0,
            'Max Drawdown': 0.0,
            'Sharpe Ratio': 0.0
        }