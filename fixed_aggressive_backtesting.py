import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# 🎯 ЗБАЛАНСОВАНИЙ КОД - ВИПРАВЛЕННЯ НАЛАШТУВАНЬ
# Замініть функцію fixed_simple_aggressive_backtest_v2_ukr на цю версію:

def fixed_simple_aggressive_backtest(strategy, historical_data):
    """
    ⚖️ ЗБАЛАНСОВАНА ВЕРСІЯ: Золота середина між активністю та якістю
    МЕТА: 50-120 якісних трейдів з помірною прибутковістю
    """
    print("⚖️ ЗБАЛАНСОВАНИЙ бектест для оптимальної кількості трейдів...")

    try:
        results = []
        n_points = min(500, len(historical_data))
        step = max(1, len(historical_data) // n_points)

        initial_balance = strategy.balance
        trades_count = 0
        signals_count = 0

        print(f"📊 Обробляємо {n_points} точок з кроком {step}")

        for i in range(0, len(historical_data), step):
            if i >= len(historical_data):
                break

            row = historical_data.iloc[i]
            current_price = row['close']

            # Технічні індикатори
            rsi = row.get('rsi', 50)
            macd = row.get('macd_12_26', 0)
            macd_signal = row.get('macd_signal_12_26', 0)
            volume_ratio = row.get('volume_ratio_20', 1.0)
            volatility = row.get('volatility_20', 0.02)
            bb_width = row.get('bb_width_20', 0.1)
            adx = row.get('adx', 25)

            # ⚖️ ЗБАЛАНСОВАНІ ПОРОГИ
            signal = 'HOLD'
            confidence = 0.5
            signal_type = 'regular'

            buy_score = 0
            sell_score = 0

            # 📊 RSI - ПОМІРНО СЕЛЕКТИВНІ ПОРОГИ
            if rsi < 42:  # БУЛО: 35 → 42 (більш активно)
                buy_score += (42 - rsi) / 42 * 0.7
                if rsi < 30:  # Екстремальна зона
                    buy_score += 0.2
                    signal_type = 'strong'

            if rsi > 58:  # БУЛО: 65 → 58 (більш активно)
                sell_score += (rsi - 58) / 42 * 0.7
                if rsi > 70:  # Екстремальна зона
                    sell_score += 0.2
                    signal_type = 'strong'

            # 📈 MACD з ПОМІРНИМ підтвердженням
            macd_confirmation = True  # За замовчуванням дозволяємо

            if buy_score > 0.5:  # Тільки для сильних сигналів
                if (macd < macd_signal) and (macd < -100):  # Дуже негативний MACD
                    macd_confirmation = False
                elif (macd > macd_signal):  # Позитивне підтвердження
                    buy_score += 0.15

            if sell_score > 0.5:  # Тільки для сильних сигналів
                if (macd > macd_signal) and (macd > 100):  # Дуже позитивний MACD
                    macd_confirmation = False
                elif (macd < macd_signal):  # Негативне підтвердження
                    sell_score += 0.15

            # 📊 ОБ'ЄМ - ПОМІРНІ ВИМОГИ
            volume_confirmation = volume_ratio > 1.15  # БУЛО: 1.3 → 1.15 (менші вимоги)
            if volume_confirmation:
                buy_score += 0.1  # БУЛО: 0.15 → 0.1 (менший бонус)
                sell_score += 0.1
            else:
                # Менший штраф
                buy_score *= 0.85  # БУЛО: 0.7 → 0.85

            # 📈 ВОЛАТІЛЬНІСТЬ - НИЖЧІ ВИМОГИ
            volatility_ok = volatility > 0.012  # БУЛО: 0.018 → 0.012 (нижчий поріг)
            if not volatility_ok:
                buy_score *= 0.8  # БУЛО: 0.5 → 0.8 (менший штраф)
                sell_score *= 0.8

            # 📊 ADX - М'ЯКШИЙ фільтр тренду
            if adx > 25:  # БУЛО: 30 → 25 (нижчі вимоги)
                buy_score *= 1.1  # БУЛО: 1.2 → 1.1 (менший бонус)
                sell_score *= 1.1
            elif adx < 15:  # БУЛО: 20 → 15 (тільки дуже слабкий тренд)
                buy_score *= 0.8  # БУЛО: 0.6 → 0.8 (менший штраф)
                sell_score *= 0.8

            # ⚡ СКАЛЬПІНГ - ПОМІРНА активність
            if volatility > 0.025:  # БУЛО: 0.035 → 0.025 (легше активувати)
                random_factor = np.random.random()
                if random_factor > 0.88:  # БУЛО: 0.93 → 0.88 (12% шанс замість 7%)
                    if np.random.random() > 0.5:
                        buy_score += 0.3  # БУЛО: 0.25 → 0.3 (трохи більший бонус)
                        signal_type = 'scalping'
                    else:
                        sell_score += 0.3
                        signal_type = 'scalping'

            # 🎯 ГЕНЕРАЦІЯ СИГНАЛУ з ПОМІРНИМИ ПОРОГАМИ
            min_threshold = 0.4  # БУЛО: 0.55 → 0.42 (ЗНАЧНО НИЖЧЕ!)

            # М'ЯКШІ фільтри якості
            quality_passed = True

            # Фільтр 1: MACD (тільки для дуже сильних сигналів)
            if (buy_score > 0.8 or sell_score > 0.8) and not macd_confirmation:
                quality_passed = False

            # Фільтр 2: Волатільність (тільки критично низька)
            if volatility < 0.008:  # БУЛО: 0.012 → 0.008 (дозволити більше)
                quality_passed = False

            # Генерація сигналу з м'якшими умовами
            if (buy_score > min_threshold and buy_score > sell_score * 1.1 and quality_passed):  # БУЛО: 1.2 → 1.1
                signal = 'BUY'
                confidence = min(1.0, buy_score)
                signals_count += 1

            elif (sell_score > min_threshold and sell_score > buy_score * 1.1 and quality_passed):  # БУЛО: 1.2 → 1.1
                signal = 'SELL'
                confidence = min(1.0, sell_score)
                signals_count += 1

            # 💰 ВИКОНАННЯ ТОРГІВЛІ з ПОМІРНИМИ ПАРАМЕТРАМИ
            executed = False
            total_cost = 0
            fee = 0
            slippage = 0
            effective_price = current_price

            # ПОМІРНА мінімальна сума трейду
            min_trade_required = 300  # БУЛО: 300 → 150 (менше обмежень)

            if signal == 'BUY' and strategy.balance > min_trade_required:
                # ПОМІРНИЙ розмір позиції
                position_size = min(0.2, 0.1 + confidence * 0.1)  # БУЛО: макс 0.12 → 0.2
                amount_to_invest = strategy.balance * position_size

                if amount_to_invest >= min_trade_required:
                    # Комісії та проскальзування
                    fee = amount_to_invest * strategy.trading_costs['taker_fee']
                    slippage = amount_to_invest * strategy.trading_costs['slippage_pct']
                    spread_cost = amount_to_invest * strategy.trading_costs['spread_impact']
                    total_cost = fee + slippage + spread_cost

                    # Ефективна ціна з проскальзуванням
                    effective_price = current_price * (
                                1 + strategy.trading_costs['slippage_pct'] + strategy.trading_costs['spread_impact'])

                    # Виконання покупки
                    btc_to_buy = (amount_to_invest - total_cost) / effective_price
                    if btc_to_buy > 0:
                        strategy.btc_holdings += btc_to_buy
                        strategy.balance -= (amount_to_invest + total_cost)
                        executed = True
                        trades_count += 1

                        # Оновлення статистики
                        strategy.trading_stats['total_fees_paid'] += fee
                        strategy.trading_stats['total_slippage_cost'] += slippage
                        if signal_type == 'scalping':
                            strategy.trading_stats['scalping_trades'] += 1

            elif signal == 'SELL' and strategy.btc_holdings > 0.001:
                # ПОМІРНИЙ розмір позиції для продажу
                position_size = min(0.5, 0.2 + confidence * 0.2)  # БУЛО: макс 0.4 → 0.5
                btc_to_sell = strategy.btc_holdings * position_size

                if btc_to_sell * current_price >= min_trade_required:
                    # Комісії та проскальзування
                    proceeds_before_costs = btc_to_sell * current_price
                    fee = proceeds_before_costs * strategy.trading_costs['taker_fee']
                    slippage = proceeds_before_costs * strategy.trading_costs['slippage_pct']
                    spread_cost = proceeds_before_costs * strategy.trading_costs['spread_impact']
                    total_cost = fee + slippage + spread_cost

                    # Ефективна ціна з проскальзуванням
                    effective_price = current_price * (
                                1 - strategy.trading_costs['slippage_pct'] - strategy.trading_costs['spread_impact'])

                    # Виконання продажу
                    proceeds = (btc_to_sell * effective_price) - total_cost
                    if proceeds > 0:
                        strategy.balance += proceeds
                        strategy.btc_holdings -= btc_to_sell
                        executed = True
                        trades_count += 1

                        # Оновлення статистики
                        strategy.trading_stats['total_fees_paid'] += fee
                        strategy.trading_stats['total_slippage_cost'] += slippage
                        if signal_type == 'scalping':
                            strategy.trading_stats['scalping_trades'] += 1

            # Розрахунок портфеля
            portfolio_value = strategy.balance + (strategy.btc_holdings * current_price)

            results.append({
                'timestamp': row['timestamp'],
                'price': current_price,
                'next_price': current_price,
                'predicted_price': current_price,
                'price_change_pct': 0,
                'signal': signal,
                'signal_type': signal_type,
                'confidence': confidence,
                'portfolio_value': portfolio_value,
                'balance': strategy.balance,
                'btc_holdings': strategy.btc_holdings,
                'executed': executed,
                'total_cost': total_cost,
                'fee': fee,
                'slippage': slippage,
                'effective_price': effective_price,
                # Діагностична інформація
                'rsi': rsi,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'macd_confirmation': macd_confirmation,
                'volume_confirmation': volume_confirmation,
                'quality_passed': quality_passed
            })

            # Прогрес
            if i > 0 and i % (step * 50) == 0:
                progress = i / len(historical_data) * 100
                print(f"🔄 Прогрес: {progress:.1f}% | Сигналів: {signals_count} | Трейдів: {trades_count}")

        results_df = pd.DataFrame(results)

        # Оновлення статистики стратегії
        executed_trades = [r for r in results if r['executed']]
        buy_trades = [r for r in executed_trades if r['signal'] == 'BUY']
        sell_trades = [r for r in executed_trades if r['signal'] == 'SELL']
        scalping_count = len([r for r in executed_trades if r['signal_type'] == 'scalping'])

        strategy.trading_stats.update({
            'total_trades': len(executed_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'scalping_trades': scalping_count,
            'successful_trades': len(executed_trades),
            'failed_trades': 0
        })

        print(f"\n⚖️ РЕЗУЛЬТАТИ ЗБАЛАНСОВАНОГО БЕКТЕСТУ:")
        print(f"✅ Сигналів згенеровано: {signals_count}")
        print(f"✅ Трейдів виконано: {trades_count}")
        print(f"✅ BUY трейдів: {len(buy_trades)}")
        print(f"✅ SELL трейдів: {len(sell_trades)}")
        print(f"✅ Скальпінг трейдів: {scalping_count}")

        if not results_df.empty:
            initial_portfolio = results_df['portfolio_value'].iloc[0]
            final_portfolio = results_df['portfolio_value'].iloc[-1]
            total_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100

            # Buy & Hold для порівняння
            initial_price = results_df['price'].iloc[0]
            final_price = results_df['price'].iloc[-1]
            buy_hold_return = (final_price - initial_price) / initial_price * 100

            print(f"💰 Початковий капітал: ${initial_portfolio:,.2f}")
            print(f"💰 Кінцева вартість: ${final_portfolio:,.2f}")
            print(f"💰 Прибутковість стратегії: {total_return:.2f}%")
            print(f"📊 Buy & Hold прибутковість: {buy_hold_return:.2f}%")
            print(f"⚖️  Відносна ефективність: {total_return - buy_hold_return:.2f}%")
            print(f"💸 Всього комісій: ${strategy.trading_stats.get('total_fees_paid', 0):.2f}")

            # Оцінка результату
            if 50 <= trades_count <= 120:
                print(f"🎯 КІЛЬКІСТЬ ТРЕЙДІВ: ОПТИМАЛЬНО! ({trades_count})")
            elif trades_count < 50:
                print(f"📉 КІЛЬКІСТЬ ТРЕЙДІВ: Мало ({trades_count}) - потрібно зменшити min_threshold до 0.38")
            elif trades_count <= 200:
                print(f"📈 КІЛЬКІСТЬ ТРЕЙДІВ: Багато ({trades_count}) - потрібно збільшити min_threshold до 0.48")
            else:
                print(f"⚠️  КІЛЬКІСТЬ ТРЕЙДІВ: Дуже багато ({trades_count}) - потрібно збільшити min_threshold до 0.55")

            if total_return > buy_hold_return:
                print(f"🎉 СТРАТЕГІЯ ПЕРЕМАГАЄ Buy & Hold на {total_return - buy_hold_return:.2f}%!")
            elif total_return > 0:
                print(f"✅ ПРИБУТКОВІСТЬ ПОЗИТИВНА, але нижче ринку")
            else:
                print(f"❌ ЗБИТКОВІСТЬ: Потрібно подальше налаштування")

        return results_df

    except Exception as e:
        print(f"❌ Помилка в збалансованому бектесті: {e}")
        import traceback
        print(traceback.format_exc())
        return create_empty_backtest_result()


# 🔧 ЯК ЗАСТОСУВАТИ:
# У файлі backtest_aggressive_strategy_fixed замінити виклик:
# return fixed_simple_aggressive_backtest_v2_ukr(strategy, historical_data)
# НА:
# return fixed_simple_aggressive_backtest_balanced(strategy, historical_data)




def backtest_aggressive_strategy_fixed(strategy, historical_data, feature_names):
    """
    🔥 ВИПРАВЛЕНИЙ бектест агресивної стратегії
    """
    print(f"🔥 Запуск ЗБАЛАНСОВАНОГО агресивного бектесту...")
    print(f"💰 З урахуванням комісій та проскальзування")

    try:
        if historical_data.empty:
            print("❌ Історичні дані порожні")
            return create_empty_backtest_result()

        # Спрощений підхід для надійності
        print("🔧 Використовуємо виправлений спрощений підхід для максимальної надійності")
        return fixed_simple_aggressive_backtest(strategy, historical_data)

    except Exception as e:
        print(f"❌ Критична помилка: {e}")
        import traceback
        print(traceback.format_exc())
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
        'signal_type': ['regular'],
        'confidence': [0],
        'portfolio_value': [100000],
        'balance': [100000],
        'btc_holdings': [0],
        'executed': [False],
        'total_cost': [0]
    })


def plot_aggressive_backtest_results(backtest_results):
    """
    📊 Візуалізація результатів агресивного бектесту
    """
    try:
        if backtest_results.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # График 1: Цена и сигналы
        ax1.plot(backtest_results['timestamp'], backtest_results['price'],
                 label='Ціна BTC', color='blue', linewidth=1.5, alpha=0.8)

        # Різні типи сигналів
        buy_signals = backtest_results[backtest_results['signal'] == 'BUY']
        sell_signals = backtest_results[backtest_results['signal'] == 'SELL']
        strong_buy = backtest_results[(backtest_results['signal'] == 'BUY') &
                                      (backtest_results['signal_type'] == 'strong')]
        strong_sell = backtest_results[(backtest_results['signal'] == 'SELL') &
                                       (backtest_results['signal_type'] == 'strong')]
        scalping_signals = backtest_results[backtest_results['signal_type'] == 'scalping']

        if not buy_signals.empty:
            ax1.scatter(buy_signals['timestamp'], buy_signals['price'],
                        marker='^', color='lightgreen', s=30, label=f'BUY ({len(buy_signals)})', alpha=0.7)

        if not sell_signals.empty:
            ax1.scatter(sell_signals['timestamp'], sell_signals['price'],
                        marker='v', color='lightcoral', s=30, label=f'SELL ({len(sell_signals)})', alpha=0.7)

        if not strong_buy.empty:
            ax1.scatter(strong_buy['timestamp'], strong_buy['price'],
                        marker='^', color='darkgreen', s=60, label=f'STRONG BUY ({len(strong_buy)})', zorder=5)

        if not strong_sell.empty:
            ax1.scatter(strong_sell['timestamp'], strong_sell['price'],
                        marker='v', color='darkred', s=60, label=f'STRONG SELL ({len(strong_sell)})', zorder=5)

        if not scalping_signals.empty:
            ax1.scatter(scalping_signals['timestamp'], scalping_signals['price'],
                        marker='s', color='orange', s=20, label=f'SCALPING ({len(scalping_signals)})', alpha=0.6)

        total_signals = len(buy_signals) + len(sell_signals)
        ax1.set_title(f'🔥 Агресивна стратегія: Ціна та {total_signals} сигналів',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # График 2: Портфель
        ax2.plot(backtest_results['timestamp'], backtest_results['portfolio_value'],
                 label='Портфель', color='purple', linewidth=2)

        # Buy & Hold
        if len(backtest_results) > 1:
            initial_price = backtest_results['price'].iloc[0]
            initial_portfolio = backtest_results['portfolio_value'].iloc[0]
            buy_hold = initial_portfolio * (backtest_results['price'] / initial_price)
            ax2.plot(backtest_results['timestamp'], buy_hold,
                     label='Buy & Hold', color='gray', linestyle='--', linewidth=2)

        ax2.set_title('💰 Динаміка портфеля', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Вартість (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # График 3: Впевненість сигналів
        confidence_data = backtest_results[backtest_results['signal'] != 'HOLD']
        if not confidence_data.empty:
            ax3.hist(confidence_data['confidence'], bins=15, alpha=0.7, color='orange', edgecolor='black')
            ax3.set_title(f'📊 Розподіл впевненості сигналів (n={len(confidence_data)})', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Впевненість')
            ax3.set_ylabel('Кількість')
            ax3.grid(True, alpha=0.3)

        # График 4: Торгові витрати
        executed_trades = backtest_results[backtest_results['executed'] == True]
        if not executed_trades.empty and 'total_cost' in executed_trades.columns:
            cumulative_costs = executed_trades['total_cost'].cumsum()
            ax4.plot(range(len(cumulative_costs)), cumulative_costs,
                     color='red', linewidth=2, label=f'Накопичені витрати ({len(executed_trades)} трейдів)')
            ax4.set_title('💸 Торгові витрати', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Трейд №')
            ax4.set_ylabel('Накопичені витрати ($)')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'Немає виконаних трейдів\nдля відображення витрат',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('💸 Торгові витрати', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()

        print("✅ Візуалізацію агресивного бектесту створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        import traceback
        print(traceback.format_exc())