# aggressive_trading_system_with_time_exits.py
# ФІНАЛЬНА інтегрована система: ваш існуючий код + часові виходи Джона Генрі

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import traceback

# Використовуйте ваш існуючий CONFIG
try:
    from config import CONFIG
except ImportError:
    CONFIG = {
        'trading': {'initial_balance': 100000, 'stop_loss': 0.05, 'take_profit': 0.1}
    }


class IntegratedTradingSystemWithTimeExits:
    """
    ІНТЕГРОВАНА система: ваша існуюча логіка + часові виходи Джона Генрі
    """

    def __init__(self, prediction_model, initial_balance=None):
        self.model = prediction_model
        self.initial_balance = initial_balance or CONFIG['trading']['initial_balance']

        # Ваші існуючі параметри (зберігаємо як є)
        self.balance = self.initial_balance
        self.btc_holdings = 0
        self.positions = []
        self.trade_history = []
        self.trading_stats = self._get_enhanced_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()

        # Ваші торгові параметри (зберігаємо)
        self.trading_params = {
            'base_confidence_threshold': 0.45,
            'position_size_base': 0.12,
            'max_position_size': 0.20,
            'min_position_size': 0.05,
            'rsi_oversold': 42,
            'rsi_overbought': 58,
            'enable_scalping': True,
            'scalping_threshold': 0.3,
        }

        # Ваші витрати (зберігаємо)
        self.trading_costs = {
            'maker_fee': 0.0004,
            'taker_fee': 0.0006,
            'slippage_pct': 0.0005,
            'min_trade_amount': 150,
            'spread_impact': 0.0002,
        }

        # Ваші ризик-параметри (зберігаємо)
        self.risk_params = {
            'stop_loss_pct': CONFIG['trading']['stop_loss'],
            'take_profit_pct': CONFIG['trading']['take_profit'],
            'trailing_stop_enabled': False,
            'trailing_stop_pct': 0.03,
            'max_open_positions': 1,
            'position_timeout_hours': 100
        }

        # 🆕 НОВІ параметри часових виходів (додаємо до існуючих)
        # В методі __init__ змінити параметри:
        self.time_exit_params = {
            'enable_time_exits': True,

            # Основні часові параметри
            'max_position_bars': 24,  # Зменшити з 40 до 24 годин
            'min_position_bars': 4,  # Збільшити з 2 до 4 годин

            # Адаптивні виходи
            'profitable_only_exit_bars': 16,  # Збільшити з 12 до 16
            'losing_only_exit_bars': 6,  # Зменшити з 8 до 6
            'neutral_exit_bars': 20,  # Збільшити з 16 до 20

            # Налаштувати пороги
            'profit_threshold': 0.01,  # Знизити з 0.02 до 0.01 (1%)
            'loss_threshold': -0.01,  # З -0.015 до -0.01 (-1%)
            'neutral_range': 0.003,  # Зменшити з 0.005 до 0.003 (0.3%)

            # Режими
            'pure_john_henry_mode': False,
            'hybrid_mode': True,

            # Нові параметри
            'enable_volatility_adjustment': True,  # Адаптація до волатильності
            'volatility_multiplier': 1.2,  # Множник для волатильних періодів

        }

    def calculate_avg_pnl_by_exit_type(self, exit_type):
        """
        Розраховує середній P&L для конкретного типу виходу
        """
        pnl_values = []

        for trade in self.trade_history:
            # Перевіряємо чи це закриття позиції відповідного типу
            if (trade.get('signal_type', '').startswith('close_') and
                    trade.get('close_reason') == exit_type and
                    'pnl_pct' in trade):
                pnl_values.append(trade['pnl_pct'])

        if pnl_values:
            return np.mean(pnl_values)
        return 0.0
    def calculate_current_volatility(self):
        """
        Розраховує поточну волатильність на основі історії цін
        """
        if len(self.trade_history) < 20:
            return 0.02  # Значення за замовчуванням

        # Беремо останні 20 точок
        recent_prices = []
        for trade in self.trade_history[-20:]:
            if 'price' in trade:
                recent_prices.append(trade['price'])

        if len(recent_prices) < 2:
            return 0.02

        # Розрахунок стандартного відхилення прибутковості
        returns = []
        for i in range(1, len(recent_prices)):
            ret = (recent_prices[i] - recent_prices[i - 1]) / recent_prices[i - 1]
            returns.append(ret)

        if returns:
            volatility = np.std(returns)
            return volatility

        return 0.02
    def calculate_volatility_adjusted_exits(self, volatility, base_params):
        """
        Адаптує часові виходи до поточної волатильності ринку
        """
        if not self.time_exit_params.get('enable_volatility_adjustment', False):
            return base_params

        # Якщо висока волатильність - скорочуємо час утримання
        if volatility > 0.03:  # 3% волатильність
            multiplier = 0.8
        elif volatility > 0.02:  # 2% волатильність
            multiplier = 0.9
        else:
            multiplier = 1.0

        adjusted_params = base_params.copy()
        adjusted_params['max_position_bars'] = int(base_params['max_position_bars'] * multiplier)
        adjusted_params['profitable_only_exit_bars'] = int(base_params['profitable_only_exit_bars'] * multiplier)
        adjusted_params['losing_only_exit_bars'] = int(base_params['losing_only_exit_bars'] * multiplier)

        return adjusted_params
    def _get_enhanced_trading_stats(self):
        """Розширена статистика з часовими метриками"""
        return {
            # Ваша існуюча статистика
            'total_fees_paid': 0,
            'total_slippage_cost': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'scalping_trades': 0,
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'stop_loss_triggered': 0,
            'take_profit_triggered': 0,
            'trailing_stop_triggered': 0,
            'position_timeouts': 0,

            # 🆕 Нові метрики для часових виходів
            'time_exit_triggered': 0,
            'john_henry_exits': 0,  # Виходи за максимальним часом
            'profitable_time_exits': 0,
            'losing_time_exits': 0,
            'neutral_time_exits': 0,

            # Статистика тривалості
            'total_position_duration': 0,
            'position_count_for_avg': 0,
            'min_duration': float('inf'),
            'max_duration': 0,
        }

    def _get_initial_market_conditions(self):
        """Ваші існуючі ринкові умови"""
        return {
            'volatility_threshold': 0.015,
            'volume_spike_multiplier': 1.2,
            'trend_momentum': 0,
            'recent_prices': [],
            'price_velocity': 0,
        }

    def reset_state(self):
        """Ваш існуючий метод скидання стану"""
        self.balance = self.initial_balance
        self.btc_holdings = 0
        self.positions = []
        self.trade_history = []
        self.trading_stats = self._get_enhanced_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()
        print("📈 Стан системи скинуто (з часовими виходами)")

    def calculate_position_performance(self, position, current_price):
        """🆕 Розрахунок ефективності позиції для часових виходів"""
        if position['type'] == 'BUY':
            entry_price = position['entry_price']
            return (current_price - entry_price) / entry_price
        elif position['type'] == 'SELL':
            # Для коротких позицій
            entry_price = position['entry_price']
            return (entry_price - current_price) / entry_price
        return 0

    def determine_time_exit_type(self, position, current_price, bars_held):
        """
        🆕 Покращена логіка визначення часового виходу
        """
        if not self.time_exit_params['enable_time_exits']:
            return None, None

        # Мінімальний час утримання
        if bars_held < self.time_exit_params['min_position_bars']:
            return None, None

        pnl_pct = self.calculate_position_performance(position, current_price)

        # Розрахунок волатильності (якщо доступна)
        volatility = self.calculate_current_volatility()  # Новий метод
        adjusted_params = self.calculate_volatility_adjusted_exits(
            volatility, self.time_exit_params
        )

        # 1. ГОЛОВНЕ ПРАВИЛО: Максимальний час
        if bars_held >= adjusted_params['max_position_bars']:
            return 'john_henry', f"John Henry exit: {bars_held} bars (P&L: {pnl_pct:+.2%})"

        # 2. Чистий режим Джона Генрі
        if self.time_exit_params['pure_john_henry_mode']:
            return None, None

        # 3. Гібридний режим з покращеною логікою
        if self.time_exit_params['hybrid_mode']:
            # Динамічні пороги на основі тривалості
            time_factor = bars_held / adjusted_params['max_position_bars']

            # Чим довше тримаємо, тим нижчі пороги для виходу
            dynamic_profit_threshold = self.time_exit_params['profit_threshold'] * (1 - time_factor * 0.5)
            dynamic_loss_threshold = self.time_exit_params['loss_threshold'] * (1 + time_factor * 0.5)

            # Прибуткові позиції
            if pnl_pct > dynamic_profit_threshold and bars_held >= adjusted_params['profitable_only_exit_bars']:
                return 'profitable_time', f"Profitable time exit: {bars_held}h (+{pnl_pct:.2%})"

            # Збиткові позиції (виходимо швидше)
            if pnl_pct < dynamic_loss_threshold and bars_held >= adjusted_params['losing_only_exit_bars']:
                return 'losing_time', f"Cut losses time exit: {bars_held}h ({pnl_pct:.2%})"

            # Нейтральні позиції (зменшуємо діапазон з часом)
            dynamic_neutral_range = self.time_exit_params['neutral_range'] * (1 - time_factor * 0.3)
            if abs(pnl_pct) <= dynamic_neutral_range and bars_held >= adjusted_params['neutral_exit_bars']:
                return 'neutral_time', f"Neutral time exit: {bars_held}h ({pnl_pct:.2%})"

        return None, None

    def check_all_exit_conditions_integrated(self, current_price, timestamp, volatility=0.03, trend_strength=0):
        """
        🔧 ІНТЕГРОВАНА перевірка виходів: часові + ваші існуючі стоп-лоси
        """
        positions_to_close = []

        for i, position in enumerate(self.positions):
            # Розрахунок тривалості позиції
            time_diff = timestamp - position['timestamp']
            bars_held = max(1, int(time_diff.total_seconds() / 3600))  # Години

            close_reason = None
            should_close = False
            exit_type = None

            # 🆕 1. ЧАСОВІ ВИХОДИ (принцип Джона Генрі) - ПЕРШИЙ ПРІОРИТЕТ
            if self.time_exit_params['enable_time_exits']:
                exit_type, exit_description = self.determine_time_exit_type(
                    position, current_price, bars_held
                )

                if exit_type:
                    should_close = True
                    close_reason = exit_type

                    # Оновлюємо статистику
                    self.trading_stats['time_exit_triggered'] += 1

                    if exit_type == 'john_henry':
                        self.trading_stats['john_henry_exits'] += 1
                        print(f"🕐 JOHN HENRY EXIT: {exit_description}")
                    elif exit_type == 'profitable_time':
                        self.trading_stats['profitable_time_exits'] += 1
                        print(f"🟢 PROFITABLE TIME: {exit_description}")
                    elif exit_type == 'losing_time':
                        self.trading_stats['losing_time_exits'] += 1
                        print(f"🔴 LOSING TIME: {exit_description}")
                    elif exit_type == 'neutral_time':
                        self.trading_stats['neutral_time_exits'] += 1
                        print(f"⚪ NEUTRAL TIME: {exit_description}")

            # 2. ВАШІ ІСНУЮЧІ СТОП-ЛОСИ та ТЕЙК-ПРОФІТИ (другий пріоритет)
            if not should_close and position['type'] == 'BUY':
                entry_price = position['entry_price']

                # Традиційний стоп-лос
                stop_loss_price = entry_price * (1 - self.risk_params['stop_loss_pct'])
                if current_price <= stop_loss_price:
                    should_close = True
                    close_reason = 'stop_loss'
                    self.trading_stats['stop_loss_triggered'] += 1
                    print(f"🛑 STOP LOSS: ${current_price:.0f} <= ${stop_loss_price:.0f}")

                # Традиційний тейк-профіт
                elif current_price >= entry_price * (1 + self.risk_params['take_profit_pct']):
                    should_close = True
                    close_reason = 'take_profit'
                    self.trading_stats['take_profit_triggered'] += 1
                    print(
                        f"🎯 TAKE PROFIT: ${current_price:.0f} >= ${entry_price * (1 + self.risk_params['take_profit_pct']):.0f}")

                # Ваш трейлінг стоп (якщо увімкнено)
                elif self.risk_params['trailing_stop_enabled']:
                    if 'highest_price' not in position:
                        position['highest_price'] = entry_price
                    if current_price > position['highest_price']:
                        position['highest_price'] = current_price
                    trailing_stop_price = position['highest_price'] * (1 - self.risk_params['trailing_stop_pct'])
                    if current_price <= trailing_stop_price and current_price > entry_price:
                        should_close = True
                        close_reason = 'trailing_stop'
                        self.trading_stats['trailing_stop_triggered'] += 1

            # Додаємо до списку закриття
            if should_close:
                positions_to_close.append({
                    'index': i,
                    'position': position,
                    'reason': close_reason,
                    'current_price': current_price,
                    'timestamp': timestamp,
                    'bars_held': bars_held,
                    'exit_type': exit_type
                })

        # Закриваємо позиції
        for close_info in reversed(positions_to_close):
            self._close_position_enhanced(close_info)

        return len(positions_to_close)

    def _close_position_enhanced(self, close_info):
        """
        🔧 РОЗШИРЕНЕ закриття позиції з урахуванням часових виходів
        """
        position = close_info['position']
        current_price = close_info['current_price']
        timestamp = close_info['timestamp']
        reason = close_info['reason']
        bars_held = close_info['bars_held']

        amount_to_close = position['amount']

        if position['type'] == 'BUY':
            # Використовуємо ваші існуючі розрахунки витрат
            costs = self.calculate_trading_costs(amount_to_close, current_price, 'taker')
            effective_price = costs['effective_price_sell']
            proceeds = (amount_to_close * effective_price) - costs['total_cost']

            if proceeds > 0 and amount_to_close <= self.btc_holdings:
                self.balance += proceeds
                self.btc_holdings -= amount_to_close

                # Оновлюємо вашу існуючу статистику
                self.trading_stats['total_fees_paid'] += costs['fee']
                self.trading_stats['total_slippage_cost'] += costs['slippage']
                self.trading_stats['total_trades'] += 1
                self.trading_stats['sell_trades'] += 1

                # 🆕 Оновлюємо статистику тривалості
                self.trading_stats['total_position_duration'] += bars_held
                self.trading_stats['position_count_for_avg'] += 1
                self.trading_stats['min_duration'] = min(self.trading_stats['min_duration'], bars_held)
                self.trading_stats['max_duration'] = max(self.trading_stats['max_duration'], bars_held)

                # Розрахунок P&L
                entry_value = position['amount'] * position['entry_price']
                exit_value = proceeds
                pnl = exit_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                # Розширене логування
                print(f"🔒 ЗАКРИТТЯ BUY ({reason.upper()}): {amount_to_close:.6f} BTC @ ${effective_price:.0f}")
                print(f"   ⏱️ Тривалість: {bars_held} годин | P&L: ${pnl:+.0f} ({pnl_pct:+.2f}%)")
                print(f"   💰 Balance: ${self.balance:.0f} | BTC: {self.btc_holdings:.6f}")

                # Записуємо в історію з розширеною інформацією
                trade_info = {
                    'timestamp': timestamp,
                    'signal': 'SELL',
                    'signal_type': f'close_{reason}',
                    'price': current_price,
                    'executed': True,
                    'amount': amount_to_close,
                    'proceeds': proceeds,
                    'total_cost': costs['total_cost'],
                    'fee': costs['fee'],
                    'slippage': costs['slippage'],
                    'effective_price': effective_price,
                    'balance_after': self.balance,
                    'btc_after': self.btc_holdings,
                    'portfolio_value': self.balance + (self.btc_holdings * current_price),
                    'close_reason': reason,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'entry_price': position['entry_price'],
                    # 🆕 Нові поля для часових виходів
                    'bars_held': bars_held,
                    'position_duration_hours': bars_held,
                    'exit_type': close_info.get('exit_type', 'traditional'),
                    'is_time_exit': reason in ['john_henry', 'profitable_time', 'losing_time', 'neutral_time'],
                }

                self.trade_history.append(trade_info)

        # Видаляємо позицію
        self.positions.pop(close_info['index'])

    # Використовуємо ваші існуючі методи без змін:
    def calculate_trading_costs(self, amount, price, trade_type='taker'):
        """Ваша існуюча логіка розрахунку витрат"""
        trade_value = amount * price

        if trade_type == 'maker':
            fee = trade_value * self.trading_costs['maker_fee']
        else:
            fee = trade_value * self.trading_costs['taker_fee']

        slippage = trade_value * self.trading_costs['slippage_pct']
        spread_cost = trade_value * self.trading_costs['spread_impact']
        total_cost = fee + slippage + spread_cost

        return {
            'fee': fee,
            'slippage': slippage,
            'spread_cost': spread_cost,
            'total_cost': total_cost,
            'effective_price_buy': price * (
                        1 + self.trading_costs['slippage_pct'] + self.trading_costs['spread_impact']),
            'effective_price_sell': price * (
                        1 - self.trading_costs['slippage_pct'] - self.trading_costs['spread_impact'])
        }

    def update_market_conditions(self, current_price, volume=None, technical_indicators=None):
        """Ваша існуюча логіка оновлення ринкових умов"""
        self.market_conditions['recent_prices'].append(current_price)
        if len(self.market_conditions['recent_prices']) > 20:
            self.market_conditions['recent_prices'].pop(0)

        if len(self.market_conditions['recent_prices']) >= 5:
            prices = np.array(self.market_conditions['recent_prices'])
            price_changes = np.diff(prices)
            self.market_conditions['price_velocity'] = np.mean(price_changes[-3:])

            if len(prices) >= 10:
                short_avg = np.mean(prices[-5:])
                long_avg = np.mean(prices[-10:])
                self.market_conditions['trend_momentum'] = (short_avg - long_avg) / long_avg * 100

    def generate_aggressive_signal(self, features, current_price, technical_indicators):
        """Ваша існуюча логіка генерації сигналів (без змін)"""
        self.update_market_conditions(current_price, technical_indicators=technical_indicators)

        try:
            predicted_price = self.model.predict(features)[0]
            price_change_pct = (predicted_price - current_price) / current_price * 100
        except Exception as e:
            predicted_price = current_price
            price_change_pct = 0

        rsi = technical_indicators.get('rsi', 50)
        macd = technical_indicators.get('macd', 0)
        macd_signal = technical_indicators.get('macd_signal', 0)
        volume_ratio = technical_indicators.get('volume_ratio_20', 1.0)

        buy_score = 0
        sell_score = 0

        # Ваша існуюча логіка скорингу
        ml_threshold = self.trading_params['base_confidence_threshold']
        if price_change_pct > ml_threshold:
            buy_score += 0.3 * min(2.0, price_change_pct / ml_threshold)
        elif price_change_pct < -ml_threshold:
            sell_score += 0.3 * min(2.0, abs(price_change_pct) / ml_threshold)

        if rsi < self.trading_params['rsi_oversold']:
            buy_score += 0.2 * (self.trading_params['rsi_oversold'] - rsi) / self.trading_params['rsi_oversold']
        elif rsi > self.trading_params['rsi_overbought']:
            sell_score += 0.2 * (rsi - self.trading_params['rsi_overbought']) / (
                        100 - self.trading_params['rsi_overbought'])

        # Решта вашої логіки...
        signal = 'HOLD'
        confidence = 0
        signal_type = 'regular'
        min_signal_threshold = 0.4

        if buy_score > min_signal_threshold and buy_score > sell_score * 1.1:
            signal = 'BUY'
            confidence = min(1.0, buy_score)
            signal_type = 'strong' if buy_score > 0.8 else 'regular'
        elif sell_score > min_signal_threshold and sell_score > buy_score * 1.1:
            signal = 'SELL'
            confidence = min(1.0, sell_score)
            signal_type = 'strong' if sell_score > 0.8 else 'regular'

        return signal, predicted_price, price_change_pct, confidence, signal_type

    def execute_trade_with_costs(self, signal, current_price, timestamp, predicted_price, confidence, signal_type):
        """
        🔧 МОДИФІКОВАНА версія вашого методу виконання торгівлі
        Тепер використовує нову перевірку виходів
        """
        # 🆕 ГОЛОВНА ЗМІНА: використовуємо інтегровану перевірку виходів
        closed_positions = self.check_all_exit_conditions_integrated(current_price, timestamp)
        if closed_positions > 0:
            print(f"🔒 Закрито {closed_positions} позицій (час + традиційні виходи)")

        # Решта коду залишається як у вас (ваша логіка виконання нових трейдів)
        initial_portfolio_value = self.balance + (self.btc_holdings * current_price)

        trade_info = {
            'timestamp': timestamp,
            'signal': signal,
            'signal_type': signal_type,
            'price': current_price,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'balance_before': self.balance,
            'btc_before': self.btc_holdings,
            'executed': False,
            'total_cost': 0,
            'fee': 0,
            'slippage': 0,
            'effective_price': current_price,
            'reason': None,
            'portfolio_value_before': initial_portfolio_value,
        }

        executed_this_step = False

        # Ваша існуюча логіка відкриття нових позицій
        if (signal == 'BUY' and self.balance > self.trading_costs['min_trade_amount'] and
                len(self.positions) < self.risk_params['max_open_positions']):

            position_size = self.calculate_position_size(signal, confidence, signal_type, current_price)
            full_portfolio_value = self.balance + (self.btc_holdings * current_price)
            desired_investment = full_portfolio_value * position_size
            amount_to_invest = min(desired_investment, self.balance * 0.95)

            if amount_to_invest >= self.trading_costs['min_trade_amount']:
                costs = self.calculate_trading_costs(amount_to_invest / current_price, current_price, 'taker')
                effective_price = costs['effective_price_buy']
                total_needed = amount_to_invest + costs['total_cost']

                if total_needed <= self.balance:
                    btc_to_buy = (amount_to_invest - costs['total_cost']) / effective_price

                    if btc_to_buy > 0:
                        self.btc_holdings += btc_to_buy
                        self.balance -= total_needed

                        # Ваша статистика
                        self.trading_stats['total_fees_paid'] += costs['fee']
                        self.trading_stats['total_slippage_cost'] += costs['slippage']
                        self.trading_stats['total_trades'] += 1
                        self.trading_stats['buy_trades'] += 1

                        # 🆕 РОЗШИРЕНА позиція з часовими параметрами
                        position = {
                            'type': 'BUY',
                            'entry_price': effective_price,
                            'amount': btc_to_buy,
                            'timestamp': timestamp,
                            'signal_type': signal_type,
                            'confidence': confidence,
                            'stop_loss_price': effective_price * (1 - self.risk_params['stop_loss_pct']),
                            'take_profit_price': effective_price * (1 + self.risk_params['take_profit_pct']),
                            # 🆕 Додаємо часові параметри
                            'max_exit_time': timestamp + timedelta(hours=self.time_exit_params['max_position_bars']),
                            'profitable_exit_time': timestamp + timedelta(
                                hours=self.time_exit_params['profitable_only_exit_bars']),
                            'losing_exit_time': timestamp + timedelta(
                                hours=self.time_exit_params['losing_only_exit_bars']),
                        }
                        self.positions.append(position)

                        trade_info.update({
                            'executed': True,
                            'amount': btc_to_buy,
                            'total_cost': costs['total_cost'],
                            'fee': costs['fee'],
                            'slippage': costs['slippage'],
                            'effective_price': effective_price,
                            'amount_invested': amount_to_invest,
                        })

                        executed_this_step = True

                        print(f"🟢 BUY: ${amount_to_invest:.0f} → {btc_to_buy:.6f} BTC @ ${effective_price:.0f}")
                        print(
                            f"   🕐 Макс. час: {self.time_exit_params['max_position_bars']}h | SL: ${position['stop_loss_price']:.0f} | TP: ${position['take_profit_price']:.0f}")

        # Розрахунок фінального портфеля
        final_portfolio_value = self.balance + (self.btc_holdings * current_price)
        trade_info.update({
            'balance_after': self.balance,
            'btc_after': self.btc_holdings,
            'portfolio_value': final_portfolio_value,
            'portfolio_change': final_portfolio_value - initial_portfolio_value
        })

        if signal != 'HOLD' or executed_this_step:
            self.trade_history.append(trade_info)

        return trade_info

    def calculate_position_size(self, signal, confidence, signal_type, current_price):
        """Ваш існуючий метод розрахунку розміру позиції"""
        base_size = self.trading_params['position_size_base']
        confidence_multiplier = 0.5 + (confidence * 1.5)

        if signal_type == 'strong':
            type_multiplier = 1.3
        elif signal_type == 'scalping':
            type_multiplier = 0.6
        else:
            type_multiplier = 1.0

        portfolio_value = self.balance + (self.btc_holdings * current_price)
        if portfolio_value < self.initial_balance * 0.8:
            balance_multiplier = 0.7
        else:
            balance_multiplier = 1.0

        position_size = base_size * confidence_multiplier * type_multiplier * balance_multiplier
        position_size = max(self.trading_params['min_position_size'],
                            min(self.trading_params['max_position_size'], position_size))
        return position_size

    def get_enhanced_trading_statistics(self):
        """
        📊 РОЗШИРЕНА статистика з часовими метриками
        """
        stats = self.trading_stats.copy()

        # Розрахунок середньої тривалості позицій
        if stats['position_count_for_avg'] > 0:
            stats['avg_position_duration'] = stats['total_position_duration'] / stats['position_count_for_avg']
        else:
            stats['avg_position_duration'] = 0

        if stats['min_duration'] == float('inf'):
            stats['min_duration'] = 0

        # Розрахунок відсотків виходів
        total_exits = (stats['stop_loss_triggered'] + stats['take_profit_triggered'] +
                       stats['time_exit_triggered'] + stats.get('trailing_stop_triggered', 0))

        if total_exits > 0:
            stats['stop_loss_rate'] = (stats['stop_loss_triggered'] / total_exits) * 100
            stats['take_profit_rate'] = (stats['take_profit_triggered'] / total_exits) * 100
            stats['time_exit_rate'] = (stats['time_exit_triggered'] / total_exits) * 100

            # Ефективність часових виходів
            if stats['time_exit_triggered'] > 0:
                stats['john_henry_effectiveness'] = (stats['john_henry_exits'] / stats['time_exit_triggered']) * 100
                stats['profitable_time_rate'] = (stats['profitable_time_exits'] / stats['time_exit_triggered']) * 100
        if stats['time_exit_triggered'] > 0:
            # Розрахунок ефективності по типах виходів
            total_time_exits = stats['time_exit_triggered']

            stats['john_henry_rate'] = (stats['john_henry_exits'] / total_time_exits) * 100
            stats['profitable_time_rate'] = (stats['profitable_time_exits'] / total_time_exits) * 100
            stats['losing_time_rate'] = (stats['losing_time_exits'] / total_time_exits) * 100
            stats['neutral_time_rate'] = (stats['neutral_time_exits'] / total_time_exits) * 100

            # Середній P&L для кожного типу виходу
            stats['avg_pnl_john_henry'] = self.calculate_avg_pnl_by_exit_type('john_henry')
            stats['avg_pnl_profitable_time'] = self.calculate_avg_pnl_by_exit_type('profitable_time')
            stats['avg_pnl_losing_time'] = self.calculate_avg_pnl_by_exit_type('losing_time')
            stats['avg_pnl_neutral_time'] = self.calculate_avg_pnl_by_exit_type('neutral_time')

        return stats

    def print_comprehensive_summary(self):
        """
        🆕 КОМПЛЕКСНИЙ підсумок: традиційні + часові метрики
        """
        stats = self.get_enhanced_trading_statistics()

        print(f"\n📊 КОМПЛЕКСНИЙ ПІДСУМОК ТОРГОВОЇ СИСТЕМИ:")
        print("=" * 60)

        # Загальні метрики
        print(f"📈 Загальні результати:")
        print(f"   Всього трейдів: {stats['total_trades']}")
        print(f"   BUY трейдів: {stats['buy_trades']}")
        print(f"   SELL трейдів: {stats['sell_trades']}")
        print(f"   Загальні комісії: ${stats['total_fees_paid']:.2f}")

        # Традиційні виходи
        print(f"\n🛑 Традиційні виходи:")
        print(f"   Стоп-лоси: {stats['stop_loss_triggered']} ({stats.get('stop_loss_rate', 0):.1f}%)")
        print(f"   Тейк-профіти: {stats['take_profit_triggered']} ({stats.get('take_profit_rate', 0):.1f}%)")
        print(f"   Трейлінг стопи: {stats.get('trailing_stop_triggered', 0)}")

        # 🆕 Часові виходи (принцип Джона Генрі)
        print(f"\n🕐 ЧАСОВІ ВИХОДИ (принцип Джона Генрі):")
        print(f"   Всього часових виходів: {stats['time_exit_triggered']} ({stats.get('time_exit_rate', 0):.1f}%)")
        print(f"   'Чистих' виходів Джона Генрі: {stats['john_henry_exits']}")
        print(f"   Прибуткових часових: {stats['profitable_time_exits']}")
        print(f"   Збиткових часових: {stats['losing_time_exits']}")
        print(f"   Нейтральних часових: {stats['neutral_time_exits']}")

        # Статистика тривалості
        print(f"\n⏱️ СТАТИСТИКА ТРИВАЛОСТІ ПОЗИЦІЙ:")
        print(f"   Середня тривалість: {stats['avg_position_duration']:.1f} годин")
        print(f"   Мінімальна: {stats['min_duration']} годин")
        print(f"   Максимальна: {stats['max_duration']} годин")

        # Оцінка ефективності
        if stats['time_exit_triggered'] > 0:
            john_henry_ratio = stats.get('john_henry_effectiveness', 0)
            profitable_ratio = stats.get('profitable_time_rate', 0)

            print(f"\n🎯 ОЦІНКА ЕФЕКТИВНОСТІ:")
            if john_henry_ratio > 50:
                print(f"✅ Принцип Джона Генрі ДОМІНУЄ ({john_henry_ratio:.1f}% часових виходів)")
            else:
                print(f"⚡ Гібридний підхід працює ({john_henry_ratio:.1f}% чистих виходів за часом)")

            if profitable_ratio > 60:
                print(f"🟢 Часові виходи ЕФЕКТИВНІ ({profitable_ratio:.1f}% прибуткових)")
            elif profitable_ratio > 40:
                print(f"⚪ Часові виходи ЗБАЛАНСОВАНІ ({profitable_ratio:.1f}% прибуткових)")
            else:
                print(f"🔴 Часові виходи потребують НАЛАШТУВАННЯ ({profitable_ratio:.1f}% прибуткових)")

    def run_backtest(self, historical_data, feature_names_for_model):
        """
        🔧 МОДИФІКОВАНИЙ бектест з інтегрованими часовими виходами
        """
        print("⚖️ Запуск бектесту з ІНТЕГРОВАНИМИ часовими виходами...")
        print(f"🕐 Режим: {self.time_exit_params}")

        self.reset_state()

        # Ваша існуюча логіка бектесту...
        results_log = []
        n_points_to_process = min(500, len(historical_data))
        step = max(1, len(historical_data) // n_points_to_process)

        print(f"📊 Обробка {len(historical_data) // step} точок з кроком {step}")

        for i in range(0, len(historical_data), step):
            row = historical_data.iloc[i]
            current_price = row['close']
            timestamp = row['timestamp']

            # Ваша існуюча підготовка ознак
            model_input_features_series = row[feature_names_for_model]
            model_input_features_for_prediction = pd.DataFrame([model_input_features_series],
                                                               columns=feature_names_for_model)

            technical_indicators_dict = {
                'rsi': row.get('rsi', 50),
                'macd': row.get('macd_12_26', row.get('macd', 0)),
                'macd_signal': row.get('macd_signal_12_26', row.get('macd_signal', 0)),
                'adx': row.get('adx', 20),
                'volume_ratio_20': row.get('volume_ratio_20', 1.0)
            }

            # Генерація сигналу (ваша логіка)
            signal, predicted_price, price_change_pct, confidence, signal_type = self.generate_aggressive_signal(
                features=model_input_features_for_prediction,
                current_price=current_price,
                technical_indicators=technical_indicators_dict
            )

            # 🆕 ВИКОНАННЯ ТОРГІВЛІ з інтегрованими виходами
            trade_info = self.execute_trade_with_costs(
                signal=signal,
                current_price=current_price,
                timestamp=timestamp,
                predicted_price=predicted_price,
                confidence=confidence,
                signal_type=signal_type
            )

            # Логування
            log_entry = {
                'timestamp': timestamp,
                'price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'signal': signal,
                'signal_type': signal_type,
                'confidence': confidence,
                'portfolio_value': trade_info['portfolio_value'],
                'balance': self.balance,
                'btc_holdings': self.btc_holdings,
                'executed': trade_info.get('executed', False),
                'open_positions': len(self.positions),
                # 🆕 Додаткова інформація про часові виходи
                'max_position_age': max([
                    int((timestamp - pos['timestamp']).total_seconds() / 3600)
                    for pos in self.positions
                ], default=0)
            }

            results_log.append(log_entry)

            # Прогрес
            if i > 0 and (i // step) % 50 == 0:
                progress_pct = (i + step) / len(historical_data) * 100
                print(
                    f"🔄 {progress_pct:.1f}% | Трейдів: {self.trading_stats['total_trades']} | Відкритих: {len(self.positions)}")

        # 🆕 Закриваємо всі позиції в кінці
        if len(self.positions) > 0:
            final_price = historical_data.iloc[-1]['close']
            final_timestamp = historical_data.iloc[-1]['timestamp']
            print(f"\n🔒 Закриття {len(self.positions)} позицій в кінці бектесту...")

            for position in self.positions.copy():
                self._close_position_enhanced({
                    'index': self.positions.index(position),
                    'position': position,
                    'reason': 'backtest_end',
                    'current_price': final_price,
                    'timestamp': final_timestamp,
                    'bars_held': int((final_timestamp - position['timestamp']).total_seconds() / 3600),
                    'exit_type': 'forced'
                })

        results_df = pd.DataFrame(results_log)

        # 🆕 КОМПЛЕКСНИЙ підсумок
        self.print_comprehensive_summary()

        return results_df


# ШВИДКИЙ КОНФІГУРАТОР для експериментів
def create_time_exit_config(strategy_type='balanced'):
    """
    🚀 Оновлені конфігурації з кращими параметрами
    """
    configs = {
        'john_henry_pure': {
            'enable_time_exits': True,
            'max_position_bars': 24,
            'pure_john_henry_mode': True,
            'hybrid_mode': False,
        },

        'balanced': {
            'enable_time_exits': True,
            'max_position_bars': 24,
            'profitable_only_exit_bars': 16,
            'losing_only_exit_bars': 6,
            'profit_threshold': 0.01,
            'loss_threshold': -0.01,
            'neutral_range': 0.003,
            'pure_john_henry_mode': False,
            'hybrid_mode': True,
            'enable_volatility_adjustment': True,
        },

        'aggressive': {
            'enable_time_exits': True,
            'max_position_bars': 12,
            'profitable_only_exit_bars': 8,
            'losing_only_exit_bars': 4,
            'profit_threshold': 0.008,
            'loss_threshold': -0.008,
            'neutral_range': 0.002,
            'pure_john_henry_mode': False,
            'hybrid_mode': True,
            'enable_volatility_adjustment': True,
        },

        'patient': {
            'enable_time_exits': True,
            'max_position_bars': 36,
            'profitable_only_exit_bars': 24,
            'losing_only_exit_bars': 8,
            'profit_threshold': 0.015,
            'loss_threshold': -0.012,
            'neutral_range': 0.004,
            'pure_john_henry_mode': False,
            'hybrid_mode': True,
            'enable_volatility_adjustment': True,
        },

        # Нова конфігурація
        'adaptive': {
            'enable_time_exits': True,
            'max_position_bars': 20,
            'profitable_only_exit_bars': 14,
            'losing_only_exit_bars': 5,
            'profit_threshold': 0.012,
            'loss_threshold': -0.008,
            'neutral_range': 0.0025,
            'pure_john_henry_mode': False,
            'hybrid_mode': True,
            'enable_volatility_adjustment': True,
            'volatility_multiplier': 1.5,
        }
    }

    return configs.get(strategy_type, configs['balanced'])

