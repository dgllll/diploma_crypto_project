# aggressive_trading_system.py з додаванням стоп-лосу та тейк-профіту

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import traceback

try:
    from config import CONFIG
except ImportError:
    print("УВАГА: Не вдалося імпортувати CONFIG з файлу config.py. Використовується базовий конфіг.")
    CONFIG = {
        'data': {
            'symbol': 'BTC-USD',
            'timeframe': '1h',
            'period': '4y',
            'test_size': 0.2,
            'validation_size': 0.1
        },
        'features': {
            'use_multitimeframe': True,
            'timeframes': ['1H', '4H', '1D'],
            'use_fib_lags': True,
            'max_lags': 34,
            'n_selected_features': 50
        },
        'models': {
            'optimize_hyperparams': False,
            'n_forecast_periods': 24,
            'n_iterations': 10,
            'cv_folds': 3
        },
        'trading': {
            'initial_balance': 100000,
            'position_size': 0.2,
            'confidence_threshold': 0.5,
            'stop_loss': 0.05,  # 5% стоп-лос
            'take_profit': 0.1  # 10% тейк-профіт
        },
        'saving': {
            'models_dir': 'models/saved',
            'results_dir': 'results',
            'data_dir': 'data'
        }
    }


def ensure_dir(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def timestamp_to_string():
    return datetime.now().strftime('%Y%m%d_%H%M%S')


class AggressiveTradingSystem:
    """
    Об'єднана система для агресивної торгової стратегії з стоп-лосом та тейк-профітом
    """

    def __init__(self, prediction_model, initial_balance=None):
        self.model = prediction_model
        if initial_balance is None:
            self.initial_balance = CONFIG['trading']['initial_balance']
        else:
            self.initial_balance = initial_balance

        # Стан стратегії, що скидається
        self.balance = self.initial_balance
        self.btc_holdings = 0
        self.positions = []  # 🔥 ВАЖЛИВО: Тепер зберігаємо відкриті позиції для стоп-лосу/тейк-профіту
        self.trade_history = []
        self.trading_stats = self._get_initial_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()

        # 🔥 АГРЕСИВНІ ПАРАМЕТРИ для більшої кількості трейдів
        self.trading_params = {
            'base_confidence_threshold': 1,
            'min_confidence_threshold': 0.8,
            'position_size_base': 0.1,
            'max_position_size': 0.2,
            'min_position_size': 0.05,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'enable_scalping': True,
            'scalping_threshold': 0.3,
        }
        self.time_exit_params = {
            'enable_time_exits': True,
            'max_position_bars': 24,  # Як у Джона Генрі - вихід після X барів
            'min_position_bars': 2,

            # Варіації (експериментуйте з цими значеннями):
            'profitable_only_exit_bars': 12,  # Вихід після 12 барів якщо +прибуток
            'losing_only_exit_bars': 8,  # Вихід після 8 барів якщо -збиток
            'neutral_exit_bars': 16,  # Вихід після 16 барів якщо ~нейтрально

            'profit_threshold': 0.02,  # 2% - межа "прибуткового"
            'loss_threshold': -0.015,  # -1.5% - межа "збиткового"
            'neutral_range': 0.005,  # ±0.5% - нейтральна зона
        }

        # 💰 КОМІСІЇ ТА ПРОСКАЛЬЗУВАННЯ
        self.trading_costs = {
            'maker_fee': 0.0004,
            'taker_fee': 0.0006,
            'slippage_pct': 0.0005,
            'min_trade_amount': 150,
            'spread_impact': 0.0002,
        }

        # 🛡️ НОВІ ПАРАМЕТРИ ДЛЯ СТОП-ЛОСУ ТА ТЕЙК-ПРОФІТУ
        self.risk_params = {
            'stop_loss_pct': CONFIG['trading']['stop_loss'],  # 5% стоп-лос
            'take_profit_pct': CONFIG['trading']['take_profit'],  # 10% тейк-профіт
            'trailing_stop_enabled': False,  # Увімкнути трейлінг стоп
            'trailing_stop_pct': 0.03,  # 3% трейлінг стоп
            'max_open_positions': 1,  # Максимум відкритих позицій
            'position_timeout_hours': 100  # Максимальний час утримання позиції (години)
        }

    def _get_initial_trading_stats(self):
        return {
            'total_fees_paid': 0,
            'total_slippage_cost': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'scalping_trades': 0,
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            # 🆕 НОВІ СТАТИСТИКИ ДЛЯ СТОП-ЛОСУ/ТЕЙК-ПРОФІТУ
            'stop_loss_triggered': 0,
            'take_profit_triggered': 0,
            'trailing_stop_triggered': 0,
            'position_timeouts': 0,
            'time_exit_triggered': 0,
            'profitable_time_exits': 0,
            'losing_time_exits': 0,
            'max_bars_exits': 0,
            'avg_position_duration': 0,
        }

    def _get_initial_market_conditions(self):
        return {
            'volatility_threshold': 0.015,
            'volume_spike_multiplier': 1.2,
            'trend_momentum': 0,
            'recent_prices': [],
            'price_velocity': 0,
        }

    def reset_state(self):
        """Скидає стан стратегії до початкового для нового бектесту."""
        self.balance = self.initial_balance
        self.btc_holdings = 0
        self.positions = []  # 🔥 ВАЖЛИВО: Скидаємо відкриті позиції
        self.trade_history = []
        self.trading_stats = self._get_initial_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()
        print("📈 Стан стратегії скинуто до початкових значень.")

    def calculate_trading_costs(self, amount, price, trade_type='taker'):
        """💰 Розрахунок реальних торгових витрат"""
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
        """🎯 Оновлення ринкових умов для агресивної торгівлі"""
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

    def check_all_exit_conditions(self, current_price, timestamp):
        """
        Перевіряє ВСІ умови виходу: часові + традиційні стоп-лоси
        """
        positions_to_close = []

        for i, position in enumerate(self.positions):
            # Розрахунок тривалості в барах (годинах)
            time_diff = timestamp - position['timestamp']
            bars_held = max(1, int(time_diff.total_seconds() / 3600))

            close_reason = None
            should_close = False

            # 🆕 1. ЧАСОВІ ВИХОДИ (принцип Джона Генрі)
            if self.time_exit_params['enable_time_exits']:
                pnl_pct = self.calculate_position_performance(position, current_price)

                # Максимальний час (головне правило Джона Генрі)
                if bars_held >= self.time_exit_params['max_position_bars']:
                    should_close = True
                    close_reason = 'max_time'
                    print(f"🕐 Вихід за максимальним часом: {bars_held} барів")

                # Вихід для прибуткових після X барів
                elif (pnl_pct > self.time_exit_params['profit_threshold'] and
                      bars_held >= self.time_exit_params['profitable_only_exit_bars']):
                    should_close = True
                    close_reason = 'profitable_time'
                    print(f"🟢 Прибутковий часовий вихід: {bars_held} барів (+{pnl_pct:.2%})")

                # Вихід для збиткових після X барів
                elif (pnl_pct < self.time_exit_params['loss_threshold'] and
                      bars_held >= self.time_exit_params['losing_only_exit_bars']):
                    should_close = True
                    close_reason = 'losing_time'
                    print(f"🔴 Збитковий часовий вихід: {bars_held} барів ({pnl_pct:.2%})")

            # 2. Ваші існуючі стоп-лоси та тейк-профіти
            if not should_close and position['type'] == 'BUY':
                entry_price = position['entry_price']

                # Стоп-лос
                stop_loss_price = entry_price * (1 - self.risk_params['stop_loss_pct'])
                if current_price <= stop_loss_price:
                    should_close = True
                    close_reason = 'stop_loss'

                # Тейк-профіт
                take_profit_price = entry_price * (1 + self.risk_params['take_profit_pct'])
                if current_price >= take_profit_price:
                    should_close = True
                    close_reason = 'take_profit'

            if should_close:
                positions_to_close.append({
                    'index': i, 'position': position, 'reason': close_reason,
                    'current_price': current_price, 'timestamp': timestamp,
                    'bars_held': bars_held
                })

        # Закриваємо позиції
        for close_info in reversed(positions_to_close):
            self._close_position(close_info)

        return len(positions_to_close)

    def calculate_position_performance(self, position, current_price):
        """Розраховує поточну ефективність позиції у відсотках"""
        if position['type'] == 'BUY':
            entry_price = position['entry_price']
            return (current_price - entry_price) / entry_price
        # Додайте логіку для SELL якщо потрібно
        return 0

    def _close_position(self, close_info):
        """
        🔒 НОВА ФУНКЦІЯ: Закриває позицію з указаною причиною
        """
        position = close_info['position']
        current_price = close_info['current_price']
        timestamp = close_info['timestamp']
        reason = close_info['reason']

        # Розраховуємо розмір для закриття (весь обсяг позиції)
        amount_to_close = position['amount']

        if position['type'] == 'BUY':
            # Закриваємо BUY позицію (продаємо BTC)
            costs = self.calculate_trading_costs(amount_to_close, current_price, 'taker')
            effective_price = costs['effective_price_sell']
            proceeds = (amount_to_close * effective_price) - costs['total_cost']

            if proceeds > 0 and amount_to_close <= self.btc_holdings:
                self.balance += proceeds
                self.btc_holdings -= amount_to_close

                # Розраховуємо прибуток/збиток
                entry_value = position['amount'] * position['entry_price']
                exit_value = proceeds
                pnl = exit_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                # Логування
                print(f"🔒 ЗАКРИТТЯ BUY позиції ({reason.upper()}): {amount_to_close:.6f} BTC @ ${effective_price:.0f}")
                print(f"   PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%) | Balance: ${self.balance:.0f}")

                # Оновлюємо статистику
                self.trading_stats['total_fees_paid'] += costs['fee']
                self.trading_stats['total_slippage_cost'] += costs['slippage']
                self.trading_stats['sell_trades'] += 1
                self.trading_stats['total_trades'] += 1

                # Записуємо в історію
                trade_info = {
                    'timestamp': timestamp,
                    'signal': 'SELL',
                    'signal_type': f'close_{reason}',
                    'price': current_price,
                    'predicted_price': current_price,
                    'confidence': 1.0,  # Високу впевненість для закриття позицій
                    'balance_before': self.balance - proceeds,
                    'btc_before': self.btc_holdings + amount_to_close,
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
                    'entry_price': position['entry_price']
                }

                self.trade_history.append(trade_info)

        # Видаляємо позицію зі списку відкритих
        self.positions.pop(close_info['index'])

    def generate_aggressive_signal(self, features, current_price, technical_indicators):
        """🔥 Агресивна генерація сигналів для більшої кількості трейдів"""
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

        # 🛡️ НОВЕ: Перевіряємо максимальну кількість відкритих позицій
        if len(self.positions) >= self.risk_params['max_open_positions']:
            print(f"⚠️ Досягнуто максимум відкритих позицій ({len(self.positions)})")
            return 'HOLD', predicted_price, price_change_pct, 0, 'max_positions_reached'

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

        if macd > macd_signal:
            buy_score += 0.15
        elif macd < macd_signal:
            sell_score += 0.15

        if self.market_conditions['price_velocity'] > 0:
            buy_score += 0.2 * min(1.0, abs(self.market_conditions['price_velocity']) / 100)
        elif self.market_conditions['price_velocity'] < 0:
            sell_score += 0.2 * min(1.0, abs(self.market_conditions['price_velocity']) / 100)

        if volume_ratio > self.market_conditions['volume_spike_multiplier']:
            volume_boost = min(0.15, (volume_ratio - 1) * 0.3)
            buy_score += volume_boost if buy_score > sell_score else 0
            sell_score += volume_boost if sell_score > buy_score else 0

        if self.trading_params['enable_scalping']:
            scalping_threshold = self.trading_params['scalping_threshold']
            if abs(price_change_pct) > scalping_threshold and abs(price_change_pct) < ml_threshold:
                if price_change_pct > 0 and rsi < 60:
                    buy_score += 0.4
                elif price_change_pct < 0 and rsi > 40:
                    sell_score += 0.4

        signal = 'HOLD'
        confidence = 0
        signal_type = 'regular'
        min_signal_threshold = 0.4

        if buy_score > min_signal_threshold and buy_score > sell_score * 1.1:
            signal = 'BUY'
            confidence = min(1.0, buy_score)
            if buy_score > 0.8:
                signal_type = 'strong'
            elif self.trading_params['enable_scalping'] and \
                    abs(price_change_pct) < self.trading_params['scalping_threshold'] and \
                    abs(price_change_pct) > 0.01:
                signal_type = 'scalping'

        elif sell_score > min_signal_threshold and sell_score > buy_score * 1.1:
            signal = 'SELL'
            confidence = min(1.0, sell_score)
            if sell_score > 0.8:
                signal_type = 'strong'
            elif self.trading_params['enable_scalping'] and \
                    abs(price_change_pct) < self.trading_params['scalping_threshold'] and \
                    abs(price_change_pct) > 0.01:
                signal_type = 'scalping'

        return signal, predicted_price, price_change_pct, confidence, signal_type

    def calculate_position_size(self, signal, confidence, signal_type, current_price):
        """📊 Розрахунок розміру позиції з урахуванням типу сигналу"""
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

    def execute_trade_with_costs(self, signal, current_price, timestamp, predicted_price,
                                 confidence, signal_type):
        """
        💰 ВИПРАВЛЕНЕ виконання торгівлі з стоп-лосом та тейк-профітом
        """
        # 🛡️ НОВЕ: Спочатку перевіряємо стоп-лос/тейк-профіт для існуючих позицій
        closed_positions = self.check_all_exit_conditions(current_price, timestamp)
        if closed_positions > 0:
            print(f"🔒 Закрито {closed_positions} позицій через стоп-лос/тейк-профіт")

        # Початковий розрахунок (ДО торгівлі)
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

        if (signal == 'BUY' and self.balance > self.trading_costs['min_trade_amount']):

            position_size = self.calculate_position_size(signal, confidence, signal_type, current_price)

            full_portfolio_value = self.balance + (self.btc_holdings * current_price)
            desired_investment = full_portfolio_value * position_size
            amount_to_invest = min(desired_investment, self.balance * 0.95)

            if amount_to_invest < self.trading_costs['min_trade_amount']:
                trade_info['reason'] = 'Amount to invest is less than min_trade_amount'
                trade_info['portfolio_value'] = initial_portfolio_value
                return trade_info

            costs = self.calculate_trading_costs(
                amount_to_invest / current_price, current_price, 'taker'
            )

            effective_price = costs['effective_price_buy']
            total_needed = amount_to_invest + costs['total_cost']

            if total_needed <= self.balance:
                btc_to_buy = (amount_to_invest - costs['total_cost']) / effective_price

                if btc_to_buy > 0:
                    # 🔧 ВИКОНУЄМО ТОРГІВЛЮ
                    self.btc_holdings += btc_to_buy
                    self.balance -= total_needed

                    # Оновлюємо статистику
                    self.trading_stats['total_fees_paid'] += costs['fee']
                    self.trading_stats['total_slippage_cost'] += costs['slippage']
                    if signal_type == 'scalping':
                        self.trading_stats['scalping_trades'] += 1
                    self.trading_stats['total_trades'] += 1
                    self.trading_stats['buy_trades'] += 1

                    # 🛡️ НОВЕ: Додаємо позицію до списку відкритих з рівнями стоп-лосу/тейк-профіту
                    position = {
                        'type': 'BUY',
                        'entry_price': effective_price,
                        'amount': btc_to_buy,
                        'timestamp': timestamp,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'stop_loss_price': effective_price * (1 - self.risk_params['stop_loss_pct']),
                        'take_profit_price': effective_price * (1 + self.risk_params['take_profit_pct']),
                        'highest_price': effective_price  # Для трейлінг стопу
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
                        'stop_loss_price': position['stop_loss_price'],
                        'take_profit_price': position['take_profit_price']
                    })

                    executed_this_step = True

                    # 📊 ДІАГНОСТИКА з рівнями стоп-лосу/тейк-профіту
                    print(f"🟢 BUY: ${amount_to_invest:.0f} → {btc_to_buy:.6f} BTC @ ${effective_price:.0f}")
                    print(f"   🛡️ SL: ${position['stop_loss_price']:.0f} | 🎯 TP: ${position['take_profit_price']:.0f}")
                    print(f"   Balance: ${self.balance:.0f}, BTC: {self.btc_holdings:.6f}")
                else:
                    trade_info['reason'] = "btc_to_buy <= 0 due to high costs"
            else:
                trade_info['reason'] = f"Not enough balance: need ${total_needed:.0f}, have ${self.balance:.0f}"

        elif (signal == 'SELL' and self.btc_holdings > 0):
            # Для SELL сигналів ми зараз просто продаємо BTC (не відкриваємо шорт позиції)
            # Якщо потрібні шорт позиції, це можна додати окремо

            position_size = self.calculate_position_size(signal, confidence, signal_type, current_price)
            btc_to_sell = min(self.btc_holdings, self.btc_holdings * position_size)

            if btc_to_sell * current_price < self.trading_costs['min_trade_amount']:
                trade_info['reason'] = 'Sell amount is less than min_trade_amount'
            else:
                costs = self.calculate_trading_costs(btc_to_sell, current_price, 'taker')
                effective_price = costs['effective_price_sell']
                proceeds = (btc_to_sell * effective_price) - costs['total_cost']

                if proceeds > 0:
                    self.balance += proceeds
                    self.btc_holdings -= btc_to_sell

                    self.trading_stats['total_fees_paid'] += costs['fee']
                    self.trading_stats['total_slippage_cost'] += costs['slippage']
                    if signal_type == 'scalping':
                        self.trading_stats['scalping_trades'] += 1
                    self.trading_stats['total_trades'] += 1
                    self.trading_stats['sell_trades'] += 1

                    trade_info.update({
                        'executed': True,
                        'amount': btc_to_sell,
                        'proceeds': proceeds,
                        'total_cost': costs['total_cost'],
                        'fee': costs['fee'],
                        'slippage': costs['slippage'],
                        'effective_price': effective_price
                    })

                    executed_this_step = True

                    print(f"🔴 SELL: {btc_to_sell:.6f} BTC → ${proceeds:.0f} @ ${effective_price:.0f}")
                    print(f"   Balance: ${self.balance:.0f}, BTC: {self.btc_holdings:.6f}")
                else:
                    trade_info['reason'] = "Proceeds <= 0 due to high costs"

        # Розраховуємо portfolio_value ПІСЛЯ торгівлі
        final_portfolio_value = self.balance + (self.btc_holdings * current_price)

        trade_info.update({
            'balance_after': self.balance,
            'btc_after': self.btc_holdings,
            'portfolio_value': final_portfolio_value,
            'portfolio_change': final_portfolio_value - initial_portfolio_value
        })

        # Записуємо в історію
        if signal != 'HOLD' or executed_this_step:
            self.trade_history.append(trade_info)

        return trade_info

    def get_trading_statistics(self):
        """
        📊 Детальна статистика торгівлі з новими метриками стоп-лосу/тейк-профіту
        """
        if not self.trade_history:
            initial_stats = self._get_initial_trading_stats()
            initial_stats.update({
                'avg_confidence': 0,
                'portfolio_volatility': 0,
                'cost_ratio': 0,
                'total_trading_costs': 0,
                'open_positions': 0,
                'stop_loss_rate': 0,
                'take_profit_rate': 0
            })
            return initial_stats

        executed_trades_history = [t for t in self.trade_history if t.get('executed', False)]
        portfolio_values = [t['portfolio_value'] for t in self.trade_history if 'portfolio_value' in t]

        stats = {
            'total_trades': self.trading_stats['total_trades'],
            'buy_trades': self.trading_stats['buy_trades'],
            'sell_trades': self.trading_stats['sell_trades'],
            'scalping_trades': self.trading_stats['scalping_trades'],
            'total_fees_paid': self.trading_stats['total_fees_paid'],
            'total_slippage_cost': self.trading_stats['total_slippage_cost'],
            'total_trading_costs': self.trading_stats['total_fees_paid'] + self.trading_stats['total_slippage_cost'],
            'avg_confidence': np.mean(
                [t.get('confidence', 0) for t in executed_trades_history]) if executed_trades_history else 0,
            'portfolio_volatility': np.std(np.diff(portfolio_values)) if len(portfolio_values) > 1 else 0,
            # 🆕 НОВІ МЕТРИКИ ДЛЯ СТОП-ЛОСУ/ТЕЙК-ПРОФІТУ
            'open_positions': len(self.positions),
            'stop_loss_triggered': self.trading_stats['stop_loss_triggered'],
            'take_profit_triggered': self.trading_stats['take_profit_triggered'],
            'trailing_stop_triggered': self.trading_stats['trailing_stop_triggered'],
            'position_timeouts': self.trading_stats['position_timeouts'],
        }

        # Розрахунок відсотків спрацьовування стоп-лосу/тейк-профіту
        total_closes = (stats['stop_loss_triggered'] + stats['take_profit_triggered'] +
                        stats['trailing_stop_triggered'] + stats['position_timeouts'])

        if total_closes > 0:
            stats['stop_loss_rate'] = stats['stop_loss_triggered'] / total_closes * 100
            stats['take_profit_rate'] = stats['take_profit_triggered'] / total_closes * 100
            stats['trailing_stop_rate'] = stats['trailing_stop_triggered'] / total_closes * 100
            stats['timeout_rate'] = stats['position_timeouts'] / total_closes * 100
        else:
            stats['stop_loss_rate'] = 0
            stats['take_profit_rate'] = 0
            stats['trailing_stop_rate'] = 0
            stats['timeout_rate'] = 0

        # Розрахунок cost_ratio
        current_or_initial_balance = self.balance if self.balance > 0 else self.initial_balance
        if current_or_initial_balance > 0:
            stats['cost_ratio'] = (stats['total_trading_costs'] / current_or_initial_balance) * 100
        else:
            stats['cost_ratio'] = 0

        return stats

    def run_backtest(self, historical_data, feature_names_for_model):
        """
        🔧 ВИПРАВЛЕНИЙ бектест з стоп-лосом та тейк-профітом
        """
        print("⚖️ Запуск бектесту з СТОП-ЛОСОМ та ТЕЙК-ПРОФІТОМ...")
        print(f"🛡️ Стоп-лос: {self.risk_params['stop_loss_pct'] * 100:.1f}%")
        print(f"🎯 Тейк-профіт: {self.risk_params['take_profit_pct'] * 100:.1f}%")
        print(f"📈 Трейлінг стоп: {self.risk_params['trailing_stop_pct'] * 100:.1f}%")

        self.reset_state()

        results_log = []
        n_points_to_process = min(500, len(historical_data))
        step = max(1, len(historical_data) // n_points_to_process)

        initial_balance_display = self.balance
        initial_btc_display = self.btc_holdings
        signals_generated_non_hold = 0

        # Перевірка наявності необхідних колонок
        required_cols = ['close', 'timestamp'] + feature_names_for_model
        indicator_cols = ['rsi', 'macd', 'macd_signal', 'adx', 'volume_ratio_20']

        missing_cols = [col for col in required_cols if col not in historical_data.columns]
        if missing_cols:
            raise ValueError(f"Відсутні необхідні колонки в historical_data: {', '.join(missing_cols)}")

        # 📊 Додаткова діагностика
        significant_changes = 0
        trade_impacts = []

        print(f"📊 ПОЧАТКОВИЙ СТАН ПОРТФЕЛЯ:")
        print(f"   💰 Balance: ${self.balance:,.2f}")
        print(f"   ₿ BTC Holdings: {self.btc_holdings:.6f}")
        print(f"   📈 Total Portfolio: ${self.balance:,.2f}")
        print(f"📊 Обробляємо приблизно {len(historical_data) // step} точок з {len(historical_data)} (крок: {step})")

        for i in range(0, len(historical_data), step):
            row = historical_data.iloc[i]
            current_price = row['close']
            timestamp = row['timestamp']

            # Підготовка ознак для моделі
            model_input_features_series = row[feature_names_for_model]
            model_input_features_for_prediction = pd.DataFrame([model_input_features_series],
                                                               columns=feature_names_for_model)

            # Технічні індикатори
            technical_indicators_dict = {
                'rsi': row.get('rsi', 50),
                'macd': row.get('macd_12_26', row.get('macd', 0)),
                'macd_signal': row.get('macd_signal_12_26', row.get('macd_signal', 0)),
                'adx': row.get('adx', 20),
                'volume_ratio_20': row.get('volume_ratio_20', 1.0)
            }

            # Стан портфеля ДО сигналу
            portfolio_before_signal = self.balance + (self.btc_holdings * current_price)

            # Генерація сигналу
            signal, predicted_price, price_change_pct, confidence, signal_type = \
                self.generate_aggressive_signal(
                    features=model_input_features_for_prediction,
                    current_price=current_price,
                    technical_indicators=technical_indicators_dict
                )

            if signal != 'HOLD':
                signals_generated_non_hold += 1

            # ВИКОНАННЯ ТОРГІВЛІ (включає перевірку стоп-лосу/тейк-профіту)
            trade_info = self.execute_trade_with_costs(
                signal=signal,
                current_price=current_price,
                timestamp=timestamp,
                predicted_price=predicted_price,
                confidence=confidence,
                signal_type=signal_type
            )

            # Стан портфеля ПІСЛЯ торгівлі
            portfolio_after_trade = trade_info['portfolio_value']
            portfolio_change = portfolio_after_trade - portfolio_before_signal

            if trade_info.get('executed', False):
                trade_impact = {
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': current_price,
                    'portfolio_before': portfolio_before_signal,
                    'portfolio_after': portfolio_after_trade,
                    'change': portfolio_change,
                    'change_pct': (
                                              portfolio_change / portfolio_before_signal) * 100 if portfolio_before_signal > 0 else 0,
                    'balance': self.balance,
                    'btc_holdings': self.btc_holdings
                }
                trade_impacts.append(trade_impact)

                # Показуємо перші 5 значних змін для діагностики
                if abs(portfolio_change) > initial_balance_display * 0.005:
                    significant_changes += 1
                    if significant_changes <= 5:
                        change_pct = (portfolio_change / portfolio_before_signal) * 100
                        print(
                            f"📊 Зміна #{significant_changes}: {signal} → {change_pct:+.2f}% (${portfolio_change:+,.0f})")
                        print(f"   Balance: ${self.balance:,.0f}, BTC: {self.btc_holdings:.6f}")

            # Логування результату з додатковою інформацією про позиції
            log_entry = {
                'timestamp': timestamp,
                'price': current_price,
                'predicted_price': predicted_price,
                'price_change_pct': price_change_pct,
                'signal': signal,
                'signal_type': signal_type,
                'confidence': confidence,
                'portfolio_value': portfolio_after_trade,
                'balance': self.balance,
                'btc_holdings': self.btc_holdings,
                'executed': trade_info.get('executed', False),
                'total_cost': trade_info.get('total_cost', 0),
                'fee': trade_info.get('fee', 0),
                'slippage': trade_info.get('slippage', 0),
                'effective_price': trade_info.get('effective_price', current_price),
                'reason_not_executed': trade_info.get('reason', None),
                'portfolio_change': portfolio_change,
                'open_positions': len(self.positions),  # 🆕 Кількість відкритих позицій
                'stop_loss_price': trade_info.get('stop_loss_price', None),
                'take_profit_price': trade_info.get('take_profit_price', None)
            }

            results_log.append(log_entry)

            # Прогрес
            if i > 0 and (i // step) % 50 == 0:
                current_loop_trades = self.trading_stats['total_trades']
                open_pos = len(self.positions)
                progress_pct = (i + step) / len(historical_data) * 100
                print(
                    f"🔄 Прогрес: {progress_pct:.1f}% | Сигналів (не HOLD): {signals_generated_non_hold} | Трейдів: {current_loop_trades} | Відкритих позицій: {open_pos}")

        # 🛡️ ЗАКРИВАЄМО ВСІ ВІДКРИТІ ПОЗИЦІЇ В КІНЦІ БЕКТЕСТУ
        if len(self.positions) > 0:
            print(f"\n🔒 Закриття {len(self.positions)} відкритих позицій в кінці бектесту...")
            final_price = historical_data.iloc[-1]['close']
            final_timestamp = historical_data.iloc[-1]['timestamp']

            for position in self.positions.copy():  # Копіюємо список, бо будемо його змінювати
                self._close_position({
                    'index': self.positions.index(position),
                    'position': position,
                    'reason': 'backtest_end',
                    'current_price': final_price,
                    'timestamp': final_timestamp
                })

        # Створення результатів
        results_df = pd.DataFrame(results_log)
        final_trading_stats = self.get_trading_statistics()

        # 🔧 РОЗШИРЕНА ДІАГНОСТИКА з метриками стоп-лосу/тейк-профіту
        print(f"\n⚖️ РЕЗУЛЬТАТИ БЕКТЕСТУ З СТОП-ЛОСОМ/ТЕЙК-ПРОФІТОМ:")
        print(f"✅ Сигналів згенеровано (не HOLD): {signals_generated_non_hold}")
        print(f"✅ Трейдів виконано: {final_trading_stats.get('total_trades', 0)}")
        print(f"✅ BUY трейдів: {final_trading_stats.get('buy_trades', 0)}")
        print(f"✅ SELL трейдів: {final_trading_stats.get('sell_trades', 0)}")
        print(f"✅ Скальпінг трейдів: {final_trading_stats.get('scalping_trades', 0)}")

        # 🛡️ СТАТИСТИКА СТОП-ЛОСУ/ТЕЙК-ПРОФІТУ
        print(f"\n🛡️ СТАТИСТИКА РИЗИК-МЕНЕДЖМЕНТУ:")
        print(
            f"🔴 Стоп-лос спрацював: {final_trading_stats.get('stop_loss_triggered', 0)} разів ({final_trading_stats.get('stop_loss_rate', 0):.1f}%)")
        print(
            f"🟢 Тейк-профіт спрацював: {final_trading_stats.get('take_profit_triggered', 0)} разів ({final_trading_stats.get('take_profit_rate', 0):.1f}%)")
        print(
            f"📈 Трейлінг стоп спрацював: {final_trading_stats.get('trailing_stop_triggered', 0)} разів ({final_trading_stats.get('trailing_stop_rate', 0):.1f}%)")
        print(
            f"⏰ Тайм-аути позицій: {final_trading_stats.get('position_timeouts', 0)} разів ({final_trading_stats.get('timeout_rate', 0):.1f}%)")

        if not results_df.empty:
            final_portfolio = results_df['portfolio_value'].iloc[-1]
            total_return = (final_portfolio - initial_balance_display) / initial_balance_display * 100

            initial_price_bh = results_df['price'].iloc[0]
            final_price_bh = results_df['price'].iloc[-1]
            buy_hold_return = (final_price_bh - initial_price_bh) / initial_price_bh * 100

            print(f"\n💰 ФІНАНСОВІ РЕЗУЛЬТАТИ:")
            print(f"💰 Початковий капітал: ${initial_balance_display:,.2f}")
            print(f"💰 Кінцева вартість портфеля: ${final_portfolio:,.2f}")
            print(f"💰 Прибутковість стратегії: {total_return:.2f}%")
            print(f"📊 Buy & Hold прибутковість: {buy_hold_return:.2f}%")
            print(f"⚖️ Відносна ефективність: {total_return - buy_hold_return:.2f}%")
            print(f"💸 Всього комісій: ${final_trading_stats.get('total_fees_paid', 0):.2f}")
            print(f"💸 Всього витрат на проскальзування: ${final_trading_stats.get('total_slippage_cost', 0):.2f}")

            # Оцінка ефективності стоп-лосу/тейк-профіту
            total_stops = (final_trading_stats.get('stop_loss_triggered', 0) +
                           final_trading_stats.get('take_profit_triggered', 0) +
                           final_trading_stats.get('trailing_stop_triggered', 0))

            if total_stops > 0:
                profit_ratio = final_trading_stats.get('take_profit_triggered', 0) / total_stops
                if profit_ratio > 0.6:
                    print(
                        f"🎯 ВІДМІННА ефективність стоп-лосу/тейк-профіту: {profit_ratio * 100:.1f}% прибуткових закриттів")
                elif profit_ratio > 0.4:
                    print(
                        f"✅ ДОБРА ефективність стоп-лосу/тейк-профіту: {profit_ratio * 100:.1f}% прибуткових закриттів")
                else:
                    print(
                        f"⚠️ НИЗЬКА ефективність стоп-лосу/тейк-профіту: {profit_ratio * 100:.1f}% прибуткових закриттів")

            # Оцінка кількості трейдів
            trades_count_from_stats = final_trading_stats.get('total_trades', 0)
            if 50 <= trades_count_from_stats <= 120:
                print(f"🎯 КІЛЬКІСТЬ ТРЕЙДІВ: В ЦІЛЬОВОМУ ДІАПАЗОНІ! ({trades_count_from_stats})")
            elif trades_count_from_stats < 50:
                print(f"📉 КІЛЬКІСТЬ ТРЕЙДІВ: Мало ({trades_count_from_stats}). Потрібно зменшити пороги.")
            else:
                print(f"📈 КІЛЬКІСТЬ ТРЕЙДІВ: Багато ({trades_count_from_stats}). Потрібно збільшити пороги.")

            # Оцінка прибутковості
            if total_return > buy_hold_return:
                print(f"🎉 СТРАТЕГІЯ ПЕРЕМАГАЄ Buy & Hold на {total_return - buy_hold_return:.2f}%!")
            elif total_return > 0:
                print(f"✅ ПРИБУТКОВІСТЬ ПОЗИТИВНА, але нижче ринку на {buy_hold_return - total_return:.2f}%")
            else:
                print(f"❌ ЗБИТКОВІСТЬ: {total_return:.2f}%. Потрібне налаштування параметрів.")
        else:
            print("⚠️ DataFrame результатів порожній.")
        self.print_time_exit_summary()
        return results_df


# Функції для створення порожнього результату та візуалізації
def create_empty_backtest_result_standalone():
    """Створює порожній результат бектесту"""
    now_utc = pd.Timestamp.now(tz='UTC') if pd.Timestamp.now().tzinfo is None else pd.Timestamp.now()
    return pd.DataFrame({
        'timestamp': [now_utc],
        'price': [50000],
        'predicted_price': [50000],
        'price_change_pct': [0],
        'signal': ['HOLD'],
        'signal_type': ['regular'],
        'confidence': [0],
        'portfolio_value': [CONFIG['trading']['initial_balance']],
        'balance': [CONFIG['trading']['initial_balance']],
        'btc_holdings': [0],
        'executed': [False],
        'total_cost': [0],
        'fee': [0],
        'slippage': [0],
        'effective_price': [50000],
        'reason_not_executed': ['No data or error'],
        'open_positions': [0],
        'stop_loss_price': [None],
        'take_profit_price': [None]
    })


def plot_backtest_results_standalone(backtest_results_df, system_name="Агресивна стратегія з стоп-лосом"):
    """
    📊 ПОВНІСТЮ ВИПРАВЛЕНА візуалізація результатів бектесту з стоп-лосом/тейк-профітом
    """
    try:
        if backtest_results_df.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # График 1: Ціна та сигнали з рівнями стоп-лосу/тейк-профіту
        ax1.plot(backtest_results_df['timestamp'], backtest_results_df['price'],
                 label='Ціна BTC', color='blue', linewidth=1.5, alpha=0.8)

        # Різні типи сигналів
        buy_signals = backtest_results_df[backtest_results_df['signal'] == 'BUY']
        sell_signals = backtest_results_df[backtest_results_df['signal'] == 'SELL']

        # Фільтруємо тільки виконані трейди
        executed_trades_df = backtest_results_df[backtest_results_df['executed'] == True]

        # Сигнали закриття позицій через стоп-лос/тейк-профіт
        stop_loss_closes = executed_trades_df[
            executed_trades_df['signal_type'].str.contains('close_stop_loss', na=False)]
        take_profit_closes = executed_trades_df[
            executed_trades_df['signal_type'].str.contains('close_take_profit', na=False)]
        trailing_stop_closes = executed_trades_df[
            executed_trades_df['signal_type'].str.contains('close_trailing_stop', na=False)]

        # Звичайні сигнали
        regular_buys = executed_trades_df[(executed_trades_df['signal'] == 'BUY') &
                                          (~executed_trades_df['signal_type'].str.contains('close_', na=False))]
        regular_sells = executed_trades_df[(executed_trades_df['signal'] == 'SELL') &
                                           (~executed_trades_df['signal_type'].str.contains('close_', na=False))]

        # Відображення різних типів сигналів
        if not regular_buys.empty:
            ax1.scatter(regular_buys['timestamp'], regular_buys['price'],
                        marker='^', color='green', s=60, label=f'BUY ({len(regular_buys)})', alpha=0.8)

        if not regular_sells.empty:
            ax1.scatter(regular_sells['timestamp'], regular_sells['price'],
                        marker='v', color='red', s=60, label=f'SELL ({len(regular_sells)})', alpha=0.8)

        if not stop_loss_closes.empty:
            ax1.scatter(stop_loss_closes['timestamp'], stop_loss_closes['price'],
                        marker='x', color='darkred', s=80, label=f'STOP LOSS ({len(stop_loss_closes)})', alpha=0.9)

        if not take_profit_closes.empty:
            ax1.scatter(take_profit_closes['timestamp'], take_profit_closes['price'],
                        marker='*', color='darkgreen', s=100, label=f'TAKE PROFIT ({len(take_profit_closes)})',
                        alpha=0.9)

        if not trailing_stop_closes.empty:
            ax1.scatter(trailing_stop_closes['timestamp'], trailing_stop_closes['price'],
                        marker='s', color='orange', s=60, label=f'TRAILING STOP ({len(trailing_stop_closes)})',
                        alpha=0.9)

        total_executed_signals = len(executed_trades_df)
        ax1.set_title(f'🛡️ {system_name}: Ціна та {total_executed_signals} сигналів з ризик-менеджментом',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # График 2: Портфель з порівнянням Buy & Hold
        ax2.plot(backtest_results_df['timestamp'], backtest_results_df['portfolio_value'],
                 label='Стратегія з стоп-лосом', color='purple', linewidth=2.5)

        if len(backtest_results_df) > 1:
            initial_price = backtest_results_df['price'].iloc[0]
            initial_portfolio = backtest_results_df['portfolio_value'].iloc[0]
            btc_equivalent_full = initial_portfolio / initial_price
            buy_hold_values_full = btc_equivalent_full * backtest_results_df['price']

            ax2.plot(backtest_results_df['timestamp'], buy_hold_values_full,
                     label='Buy & Hold (100% BTC)', color='gray', linestyle='--', linewidth=2)

            # Результати порівняння
            final_strategy = backtest_results_df['portfolio_value'].iloc[-1]
            final_buy_hold = buy_hold_values_full.iloc[-1]
            difference = final_strategy - final_buy_hold
            difference_pct = (difference / final_buy_hold) * 100

            result_text = f'Стратегія vs Buy&Hold:\n${difference:+,.0f} ({difference_pct:+.1f}%)'
            text_color = 'green' if difference > 0 else 'red'

            ax2.text(0.02, 0.98, result_text,
                     transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor=text_color, alpha=0.2),
                     fontsize=11, fontweight='bold')

        ax2.set_title('💰 Динаміка портфеля зі стоп-лосом та тейк-профітом', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Вартість (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # График 3: Кількість відкритих позицій в часі
        if 'open_positions' in backtest_results_df.columns:
            ax3.plot(backtest_results_df['timestamp'], backtest_results_df['open_positions'],
                     color='orange', linewidth=2, label='Відкриті позиції')
            ax3.fill_between(backtest_results_df['timestamp'], backtest_results_df['open_positions'],
                             alpha=0.3, color='orange')

            max_positions = backtest_results_df['open_positions'].max()
            avg_positions = backtest_results_df['open_positions'].mean()

            ax3.axhline(y=avg_positions, color='red', linestyle='--', alpha=0.7,
                        label=f'Середня: {avg_positions:.1f}')

            ax3.set_title(f'📊 Відкриті позиції в часі (макс: {max_positions})', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Кількість позицій')
            ax3.set_xlabel('Час')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Дані про відкриті позиції\nнедоступні',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title('📊 Відкриті позиції в часі', fontsize=14, fontweight='bold')

        # График 4: Ефективність стоп-лосу/тейк-профіту
        close_types = []
        close_counts = []

        if not stop_loss_closes.empty:
            close_types.append('Stop Loss')
            close_counts.append(len(stop_loss_closes))

        if not take_profit_closes.empty:
            close_types.append('Take Profit')
            close_counts.append(len(take_profit_closes))

        if not trailing_stop_closes.empty:
            close_types.append('Trailing Stop')
            close_counts.append(len(trailing_stop_closes))

        if close_types:
            colors = ['red', 'green', 'orange']
            ax4.pie(close_counts, labels=close_types, autopct='%1.1f%%',
                    colors=colors[:len(close_types)], startangle=90)
            ax4.set_title(f'🛡️ Розподіл типів закриття позицій\n(Всього: {sum(close_counts)})',
                          fontsize=14, fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Немає даних про\nзакриття позицій\nчерез стоп-лос/тейк-профіт',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('🛡️ Розподіл типів закриття позицій', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.show()
        print("✅ Візуалізацію бектесту зі стоп-лосом та тейк-профітом створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        print(traceback.format_exc())


def print_time_exit_summary(self):
    """Підсумок ефективності часових виходів"""
    stats = self.get_trading_statistics()

    print(f"\n🕐 ЧАСОВІ ВИХОДИ (принцип Джона Генрі):")
    print(f"   📊 Всього: {stats.get('time_exit_triggered', 0)}")
    print(f"   🟢 Прибуткових: {stats.get('profitable_time_exits', 0)}")
    print(f"   🔴 Збиткових: {stats.get('losing_time_exits', 0)}")
    print(f"   ⏰ За максимальним часом: {stats.get('max_bars_exits', 0)}")
# Приклад використання
if __name__ == '__main__':
    print("🛡️ Запуск системи з стоп-лосом та тейк-профітом...")


    # Фіктивна модель для тестування
    class DummyPredictionModel:
        def predict(self, features):
            if 'close_lag_1' in features.columns and 'sma_20' in features.columns:
                current_close = features['close_lag_1'].iloc[0]
                sma_20 = features['sma_20'].iloc[0]
                if current_close > sma_20 + current_close * 0.001:
                    return [current_close * 1.005]  # Прогноз росту на 0.5%
                elif current_close < sma_20 - current_close * 0.001:
                    return [current_close * 0.995]  # Прогноз падіння на 0.5%
            return [features['close_lag_1'].iloc[0] if 'close_lag_1' in features.columns else 50000]


    dummy_model = DummyPredictionModel()

    # Створення екземпляра системи з параметрами стоп-лосу/тейк-профіту з конфігу
    trading_system = AggressiveTradingSystem(
        prediction_model=dummy_model,
        initial_balance=CONFIG['trading']['initial_balance']
    )

    print(f"🛡️ Параметри ризик-менеджменту:")
    print(f"   📉 Стоп-лос: {trading_system.risk_params['stop_loss_pct'] * 100:.1f}%")
    print(f"   📈 Тейк-профіт: {trading_system.risk_params['take_profit_pct'] * 100:.1f}%")
    print(f"   🔄 Трейлінг стоп: {trading_system.risk_params['trailing_stop_pct'] * 100:.1f}%")
    print(f"   🔢 Макс. позицій: {trading_system.risk_params['max_open_positions']}")
    print(f"   ⏰ Тайм-аут позицій: {trading_system.risk_params['position_timeout_hours']} годин")

    # Підготовка тестових даних
    dates = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00',
                            '2023-01-01 03:00:00', '2023-01-01 04:00:00', '2023-01-01 05:00:00',
                            '2023-01-01 06:00:00', '2023-01-01 07:00:00', '2023-01-01 08:00:00',
                            '2023-01-01 09:00:00', '2023-01-01 10:00:00'])

    # Ціни з волатильністю для тестування стоп-лосу/тейк-профіту
    prices = [50000, 50500, 49800, 51200, 50800, 49500, 52000, 51500, 48900, 52500, 51800]

    historical_data_dict = {
        'timestamp': dates,
        'close': prices,
        'open': [p - np.random.randint(-50, 50) for p in prices],
        'high': [p + np.random.randint(0, 200) for p in prices],
        'low': [p - np.random.randint(0, 200) for p in prices],
        'volume': [np.random.randint(10, 100) for _ in prices],
        'rsi': [25, 35, 45, 30, 40, 70, 20, 55, 75, 28, 45],  # Варіативний RSI для тестування
        'macd': [10, 15, -5, 20, 10, -10, 25, 15, -15, 30, 20],
        'macd_signal': [8, 12, -2, 15, 8, -5, 20, 12, -8, 25, 18],
        'adx': [20, 25, 30, 22, 28, 35, 25, 30, 40, 28, 32],
        'volume_ratio_20': [1.0, 1.3, 0.8, 1.5, 1.1, 0.9, 1.8, 1.2, 0.7, 1.6, 1.3],
        'close_lag_1': [50000] + prices[:-1],
        'sma_20': [p - np.random.uniform(-200, 200) for p in prices]
    }
    historical_df = pd.DataFrame(historical_data_dict)

    feature_names = ['close_lag_1', 'sma_20', 'rsi', 'macd', 'volume_ratio_20']

    # Тестування системи
    try:
        print("\n🔥 Запуск тестового бектесту з стоп-лосом та тейк-профітом...")
        backtest_results_df = trading_system.run_backtest(historical_df, feature_names)

        if not backtest_results_df.empty:
            plot_backtest_results_standalone(backtest_results_df, "Тестова система з ризик-менеджментом")

        # Статистика з новими метриками
        final_stats = trading_system.get_trading_statistics()
        print("\n📊 Детальна статистика торгівлі з ризик-менеджментом:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Збереження результатів
        if not backtest_results_df.empty:
            ensure_dir(CONFIG['saving']['results_dir'])
            results_filename = f"TEST_aggressive_with_stops_{timestamp_to_string()}.csv"
            results_path = os.path.join(CONFIG['saving']['results_dir'], results_filename)
            try:
                backtest_results_df.to_csv(results_path, index=False)
                print(f"✓ Результати збережено в {results_path}")
            except Exception as e_save:
                print(f"❌ Помилка збереження: {e_save}")

    except Exception as ex:
        print(f"❌ Помилка під час тестування: {ex}")
        print(traceback.format_exc())