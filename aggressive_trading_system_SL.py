# aggressive_trading_system.py - Основна торгова система

import pandas as pd
import numpy as np
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
            'stop_loss': 0.02,  # 5% стоп-лос
            'take_profit': 0.04  # 10% тейк-профіт
        },
        'saving': {
            'models_dir': 'models/saved',
            'results_dir': 'results',
            'data_dir': 'data'
        }
    }


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
            'base_confidence_threshold': 1.0,
            'min_confidence_threshold': 0.3,
            'position_size_base': 0.1,
            'max_position_size': 0.2,
            'min_position_size': 0.05,
            'rsi_oversold': 30,
            'rsi_overbought': 60,
            'enable_scalping': False,
            'scalping_threshold': 0.3,
        }

        # 💰 КОМІСІЇ ТА ПРОСКАЛЬЗУВАННЯ
        self.trading_costs = {
            'maker_fee': 0.0004,
            'taker_fee': 0.0006,
            'slippage_pct': 0.00005,
            'min_trade_amount': 1000,
            'spread_impact': 0.0002,
        }

        # 🛡️ ПАРАМЕТРИ ДЛЯ СТОП-ЛОСУ ТА ТЕЙК-ПРОФІТУ
        self.risk_params = {
            'stop_loss_pct': CONFIG['trading']['stop_loss'],  # 2% стоп-лос
            'take_profit_pct': CONFIG['trading']['take_profit'],  # 4% тейк-профіт
            'trailing_stop_enabled': False,  # Увімкнути трейлінг стоп
            'trailing_stop_pct': 0.03,  # 3% трейлінг стоп
            'max_open_positions': 3,  # Максимум відкритих позицій
            'position_timeout_hours': 100  # Максимальний час утримання позиції (години)
        }

        # Перевіряємо, що risk_params створено правильно
        if not hasattr(self, 'risk_params') or not self.risk_params:
            print("⚠️ Увага: risk_params не створено, використовуємо значення за замовчуванням")
            self.risk_params = {
                'stop_loss_pct': 0.02,
                'take_profit_pct': 0.04,
                'trailing_stop_enabled': False,
                'trailing_stop_pct': 0.03,
                'max_open_positions': 2,
                'position_timeout_hours': 50
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
            'position_timeouts': 0
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

    def check_stop_loss_take_profit(self, current_price, timestamp, high_price=None, low_price=None):
        """
        🛡️ ВИПРАВЛЕНА ФУНКЦІЯ: Точний стоп-лос -2% та тейк-профіт
        """
        positions_to_close = []

        for i, position in enumerate(self.positions):
            close_reason = None
            should_close = False
            actual_exit_price = current_price

            if position['type'] == 'BUY':
                entry_price = position['entry_price']

                # Точні рівні для -2% стоп-лос
                stop_loss_price = entry_price * (1 - self.risk_params['stop_loss_pct'])  # entry * 0.98
                take_profit_price = entry_price * (1 + self.risk_params['take_profit_pct'])

                # 🔧 КЛЮЧОВЕ ВИПРАВЛЕННЯ: Перевіряємо HIGH/LOW бару
                if high_price is not None and low_price is not None:

                    # Take Profit має ПРІОРИТЕТ (перевіряємо спочатку)
                    if high_price >= take_profit_price:
                        should_close = True
                        close_reason = 'take_profit'
                        actual_exit_price = take_profit_price
                        self.trading_stats['take_profit_triggered'] += 1
                        print(
                            f"🎯 TP EXACT: Entry ${entry_price:.0f} → Exit ${take_profit_price:.0f} (+{self.risk_params['take_profit_pct'] * 100:.1f}%)")

                    # Stop Loss тільки якщо TP не спрацював
                    elif low_price <= stop_loss_price:
                        should_close = True
                        close_reason = 'stop_loss'
                        actual_exit_price = stop_loss_price  # 🎯 ТОЧНО -2%
                        self.trading_stats['stop_loss_triggered'] += 1
                        print(
                            f"🛡️ SL EXACT: Entry ${entry_price:.0f} → Exit ${stop_loss_price:.0f} (-{self.risk_params['stop_loss_pct'] * 100:.1f}%)")

                else:
                    # Fallback до close ціни (але це менш точно)
                    if current_price >= take_profit_price:
                        should_close = True
                        close_reason = 'take_profit'
                        actual_exit_price = current_price
                        self.trading_stats['take_profit_triggered'] += 1
                    elif current_price <= stop_loss_price:
                        should_close = True
                        close_reason = 'stop_loss'
                        actual_exit_price = current_price
                        self.trading_stats['stop_loss_triggered'] += 1

            # Тайм-аут (без змін)
            if not should_close:
                position_age_hours = (timestamp - position['timestamp']).total_seconds() / 3600
                if position_age_hours >= self.risk_params['position_timeout_hours']:
                    should_close = True
                    close_reason = 'timeout'
                    actual_exit_price = current_price
                    self.trading_stats['position_timeouts'] += 1

            if should_close:
                positions_to_close.append({
                    'index': i,
                    'position': position,
                    'reason': close_reason,
                    'current_price': actual_exit_price,  # 🎯 ТОЧНА ціна виходу
                    'timestamp': timestamp
                })

        # Закриваємо позиції
        for close_info in reversed(positions_to_close):
            self._close_position(close_info)

        return len(positions_to_close)

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
            predicted_price = self.model.predict(features)[0] * 1.025
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
        if signal == 'SELL':
            signal = 'HOLD'
            confidence = 0
            print(f"🚫 SELL сигнал заблоковано")
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
        high_price = getattr(self, '_current_high', current_price)
        low_price = getattr(self, '_current_low', current_price)
        # 🛡️ НОВЕ: Спочатку перевіряємо стоп-лос/тейк-профіт для існуючих позицій
        closed_positions = self.check_stop_loss_take_profit(
            current_price, timestamp, high_price, low_price
        )
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



        elif signal == 'SELL' and self.btc_holdings > 0:

            # Для SELL сигналів ми зараз просто продаємо BTC (не відкриваємо шорт позиції)

            position_size: float = self.calculate_position_size(signal, confidence, signal_type, current_price)

            # 🔧 ВИПРАВЛЕННЯ: Явно приводимо до float для уникнення помилок типів

            btc_holdings: float = float(self.btc_holdings)

            max_btc_to_sell: float = btc_holdings * position_size

            btc_to_sell: float = min(btc_holdings, max_btc_to_sell)

            # Перевіряємо мінімальну суму торгівлі

            trade_value: float = btc_to_sell * current_price

            if trade_value < self.trading_costs['min_trade_amount']:

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

    def show_trading_predictions_summary(self, historical_data, feature_names, n_examples=8):
        """
        📊 Показує як модель прогнозує під час торгівлі
        """
        print(f"\n📊 ПРИКЛАДИ ПРОГНОЗУВАННЯ ПІД ЧАС ТОРГІВЛІ:")
        print("=" * 70)

        sample_indices = np.linspace(0, len(historical_data) - 1, n_examples, dtype=int)

        print(f"{'Час':<12} {'Поточна':<9} {'Прогноз':<9} {'Зміна%':<7} {'Сигнал':<6} {'Впевненість':<10}")
        print("-" * 70)

        for i in sample_indices:
            row = historical_data.iloc[i]
            current_price = row['close']
            timestamp = row['timestamp']

            # Підготовка ознак
            model_features = row[feature_names].to_frame().T

            # Технічні індикатори
            tech_indicators = {
                'rsi': row.get('rsi', 50),
                'macd': row.get('macd_12_26', 0),
                'macd_signal': row.get('macd_signal_12_26', 0),
                'adx': row.get('adx', 20),
                'volume_ratio_20': row.get('volume_ratio_20', 1.0)
            }

            # Генерація сигналу
            signal, predicted_price, price_change_pct, confidence, signal_type = \
                self.generate_aggressive_signal(model_features, current_price, tech_indicators)

            time_str = timestamp.strftime('%H:%M') if hasattr(timestamp, 'strftime') else str(timestamp)[-5:]

            print(f"{time_str:<12} ${current_price:<8.0f} ${predicted_price:<8.0f} "
                  f"{price_change_pct:+6.2f}% {signal:<6} {confidence:<9.2f}")

        print("-" * 70)