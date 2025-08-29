# aggressive_trading_system.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from datetime import datetime
import traceback

# Імпорт конфігурації
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
            'use_multitimeframe': False,
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
            'stop_loss': 0.05,
            'take_profit': 0.1
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
    🔧 ПОКРАЩЕНА об'єднана система для агресивної торгової стратегії з
    інтелектуальним менеджментом позицій
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
        self.positions = []
        self.trade_history = []
        self.trading_stats = self._get_initial_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()

        # 🔧 ПОКРАЩЕНІ параметри стратегії з інтелектуальним менеджментом позицій
        self.trading_params = {
            'base_confidence_threshold': 0.6,
            'min_confidence_threshold': 0.4,

            # 📉 ЗМЕНШЕНІ розміри позицій для кращого контролю
            'position_size_base': 0.1,  # ЗМЕНШЕНО з 0.25 до 0.12 (12%)
            'max_position_size': 0.20,  # ЗМЕНШЕНО з 0.4 до 0.20 (20%)
            'min_position_size': 0.05,  # ЗМЕНШЕНО з 0.08 до 0.05 (5%)

            # 🎯 НОВІ параметри для контролю балансу портфеля
            'max_btc_allocation': 0.75,  # Максимум 75% в BTC (замість 100%)
            'min_cash_reserve': 0.15,  # Мінімум 15% готівки завжди
            'target_btc_ratio': 0.50,  # Цільовий розподіл 50/50
            'rebalance_threshold': 0.20,  # Ребалансування при 20% відхиленні

            # RSI та інші пороги
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'enable_scalping': True,
            'scalping_threshold': 0.3,
        }

        # 💰 КОМІСІЇ ТА ПРОСКАЛЬЗУВАННЯ (залишаємо як є)
        self.trading_costs = {
            'maker_fee': 0.0004,
            'taker_fee': 0.0006,
            'fixed_slippage_usd': 5.0,
            'min_trade_amount': 300,
            'spread_impact': 0.0002,
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
            'blocked_trades': 0,  # 🆕 Лічильник заблокованих трейдів
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
        self.positions = []
        self.trade_history = []
        self.trading_stats = self._get_initial_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()
        print("📈 Стан стратегії скинуто до початкових значень.")

    def calculate_current_allocation(self, current_price):
        """
        🆕 Розраховує поточний розподіл портфеля
        """
        portfolio_value = self.balance + (self.btc_holdings * current_price)

        if portfolio_value <= 0:
            return {'btc_ratio': 0, 'cash_ratio': 1, 'portfolio_value': 0}

        btc_value = self.btc_holdings * current_price
        cash_value = self.balance

        return {
            'btc_ratio': btc_value / portfolio_value,
            'cash_ratio': cash_value / portfolio_value,
            'portfolio_value': portfolio_value,
            'btc_value': btc_value,
            'cash_value': cash_value
        }

    def check_position_limits(self, signal, current_price):
        """
        🆕 Перевіряє чи можна відкрити позицію з урахуванням лімітів
        """
        allocation = self.calculate_current_allocation(current_price)

        # Блокування BUY при високому BTC ratio
        if signal == 'BUY':
            if allocation['btc_ratio'] >= self.trading_params['max_btc_allocation']:
                return False, f"BTC allocation too high: {allocation['btc_ratio']:.1%} >= {self.trading_params['max_btc_allocation']:.1%}"

            if allocation['cash_ratio'] <= self.trading_params['min_cash_reserve']:
                return False, f"Cash reserve too low: {allocation['cash_ratio']:.1%} <= {self.trading_params['min_cash_reserve']:.1%}"

        # Блокування SELL при низькому BTC ratio
        elif signal == 'SELL':
            min_btc_for_sell = 0.05  # Мінімум 5% BTC для продажу
            if allocation['btc_ratio'] <= min_btc_for_sell:
                return False, f"BTC allocation too low for selling: {allocation['btc_ratio']:.1%} <= {min_btc_for_sell:.1%}"

        return True, "OK"

    def should_rebalance(self, current_price):
        """
        🆕 Перевіряє чи потрібне ребалансування портфеля
        """
        allocation = self.calculate_current_allocation(current_price)
        target_btc_ratio = self.trading_params['target_btc_ratio']

        deviation = abs(allocation['btc_ratio'] - target_btc_ratio)

        if deviation > self.trading_params['rebalance_threshold']:
            if allocation['btc_ratio'] > target_btc_ratio:
                return 'SELL', deviation
            else:
                return 'BUY', deviation

        return None, deviation

    def calculate_trading_costs(self, amount, price, trade_type='taker'):
        """💰 Розрахунок реальних торгових витрат з фіксованим проскальзуванням"""
        trade_value = amount * price

        # Комісія брокера
        if trade_type == 'maker':
            fee = trade_value * self.trading_costs['maker_fee']
        else:
            fee = trade_value * self.trading_costs['taker_fee']  # 0.06%

        # Фіксоване проскальзування $5
        slippage = self.trading_costs['fixed_slippage_usd']

        # Відсотковий спред (як було)
        spread_cost = trade_value * self.trading_costs['spread_impact']  # 0.02%

        total_cost = fee + slippage + spread_cost

        # Ефективні ціни з урахуванням фіксованого проскальзування
        slippage_per_unit = self.trading_costs['fixed_slippage_usd'] / amount
        spread_impact = price * self.trading_costs['spread_impact']

        effective_price_buy = price + slippage_per_unit + spread_impact
        effective_price_sell = price - slippage_per_unit - spread_impact

        return {
            'fee': fee,
            'slippage': slippage,
            'spread_cost': spread_cost,
            'total_cost': total_cost,
            'effective_price_buy': effective_price_buy,
            'effective_price_sell': effective_price_sell
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

        # 🆕 ДОДАЄМО АДАПТАЦІЮ ДО ПОТОЧНОГО БАЛАНСУ ПОРТФЕЛЯ
        allocation = self.calculate_current_allocation(current_price)

        # Зменшуємо BUY score якщо багато BTC
        if allocation['btc_ratio'] > 0.6:
            buy_score *= (1.0 - allocation['btc_ratio'] * 0.5)  # Штраф до 50%

        # Збільшуємо SELL score якщо багато BTC
        if allocation['btc_ratio'] > 0.7:
            sell_score *= 1.3  # Бонус 30%

        # Зменшуємо SELL score якщо мало BTC
        if allocation['btc_ratio'] < 0.3:
            sell_score *= (allocation['btc_ratio'] / 0.3)  # Пропорційний штраф

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

    def calculate_adaptive_position_size(self, signal, confidence, signal_type, current_price):
        """
        🔧 ПОКРАЩЕНИЙ розрахунок розміру позиції з урахуванням поточного балансу
        """
        allocation = self.calculate_current_allocation(current_price)
        base_size = self.trading_params['position_size_base']

        # 📊 Адаптація до поточного розподілу портфеля
        if signal == 'BUY':
            # Чим більше BTC, тим менше купуємо
            btc_penalty = allocation['btc_ratio'] * 1.5  # Штраф за високий BTC ratio
            position_multiplier = max(0.3, 1.0 - btc_penalty)

            # Додатковий штраф якщо мало готівки
            if allocation['cash_ratio'] < 0.3:  # Менше 30% готівки
                position_multiplier *= 0.6

        elif signal == 'SELL':
            # Чим менше BTC, тим менше продаємо
            cash_penalty = allocation['cash_ratio'] * 1.2  # Штраф за високий cash ratio
            position_multiplier = max(0.3, 1.0 - cash_penalty)

            # Бонус за високий BTC ratio (потрібно ребалансувати)
            if allocation['btc_ratio'] > 0.7:  # Більше 70% BTC
                position_multiplier *= 1.3
        else:
            position_multiplier = 1.0

        # 🎯 Адаптація до впевненості (зменшено вплив)
        confidence_multiplier = 0.5 + (confidence * 1.0)  # ЗМЕНШЕНО з 1.5 до 1.0

        # 📈 Адаптація до типу сигналу (зменшено вплив)
        if signal_type == 'strong':
            type_multiplier = 1.2  # ЗМЕНШЕНО з 1.3 до 1.2
        elif signal_type == 'scalping':
            type_multiplier = 0.4  # ЗМЕНШЕНО з 0.6 до 0.4
        else:
            type_multiplier = 1.0

        # 💰 Адаптація до performance портфеля
        portfolio_performance = allocation['portfolio_value'] / self.initial_balance
        if portfolio_performance < 0.9:  # Якщо втратили >10%
            performance_multiplier = 0.7  # Торгуємо обережніше
        else:
            performance_multiplier = 1.0

        # 🧮 Фінальний розрахунок
        final_position_size = (
                base_size *
                position_multiplier *
                confidence_multiplier *
                type_multiplier *
                performance_multiplier
        )

        # 📏 Обмеження мін/макс
        final_position_size = max(
            self.trading_params['min_position_size'],
            min(self.trading_params['max_position_size'], final_position_size)
        )

        return final_position_size

    def _create_blocked_trade_info(self, signal, current_price, timestamp, reason):
        """🆕 Створює інформацію про заблокований трейд"""
        portfolio_value = self.balance + (self.btc_holdings * current_price)

        return {
            'timestamp': timestamp,
            'signal': signal,
            'price': current_price,
            'executed': False,
            'reason': reason,
            'portfolio_value': portfolio_value,
            'balance_after': self.balance,
            'btc_after': self.btc_holdings,
            'blocked_by_position_management': True
        }

    def execute_trade_with_costs(self, signal, current_price, timestamp, predicted_price,
                                 confidence, signal_type):
        """
        💰 ПОКРАЩЕНЕ виконання торгівлі з новим менеджментом позицій
        """
        # 🔍 Перевірка лімітів позицій ПЕРЕД усім іншим
        can_trade, reason = self.check_position_limits(signal, current_price)
        if not can_trade:
            self.trading_stats['blocked_trades'] += 1
            return self._create_blocked_trade_info(signal, current_price, timestamp, reason)

        # Початковий розрахунок (ДО торгівлі)
        initial_portfolio_value = self.balance + (self.btc_holdings * current_price)
        allocation = self.calculate_current_allocation(current_price)

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
            'allocation_before': allocation,
        }

        executed_this_step = False

        if (signal == 'BUY' and self.balance > self.trading_costs['min_trade_amount']):

            # 🔧 Використовуємо новий адаптивний розрахунок
            position_size = self.calculate_adaptive_position_size(signal, confidence, signal_type, current_price)

            # Розрахунок від повного портфеля, але обмежено балансом
            full_portfolio_value = self.balance + (self.btc_holdings * current_price)
            desired_investment = full_portfolio_value * position_size

            # Обмежуємо доступним балансом (залишаємо 5% готівки для комісій)
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

            # Перевіряємо, чи достатньо коштів
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

                    trade_info.update({
                        'executed': True,
                        'amount': btc_to_buy,
                        'total_cost': costs['total_cost'],
                        'fee': costs['fee'],
                        'slippage': costs['slippage'],
                        'effective_price': effective_price,
                        'amount_invested': amount_to_invest,
                        'position_size_used': position_size
                    })

                    self.positions.append({
                        'type': 'BUY', 'price': effective_price, 'amount': btc_to_buy,
                        'timestamp': timestamp, 'signal_type': signal_type, 'confidence': confidence
                    })
                    executed_this_step = True

                    # 📊 ДІАГНОСТИКА з новою інформацією
                    new_allocation = self.calculate_current_allocation(current_price)
                    print(f"🟢 BUY: ${amount_to_invest:.0f} → {btc_to_buy:.6f} BTC @ ${effective_price:.0f}")
                    print(
                        f"   Position size: {position_size:.1%}, BTC ratio: {allocation['btc_ratio']:.1%} → {new_allocation['btc_ratio']:.1%}")
                else:
                    trade_info['reason'] = "btc_to_buy <= 0 due to high costs"
            else:
                trade_info['reason'] = f"Not enough balance: need ${total_needed:.0f}, have ${self.balance:.0f}"

        elif (signal == 'SELL' and self.btc_holdings > 0):

            # 🔧 Використовуємо новий адаптивний розрахунок
            position_size = self.calculate_adaptive_position_size(signal, confidence, signal_type, current_price)
            btc_to_sell = min(self.btc_holdings, self.btc_holdings * position_size)

            if btc_to_sell * current_price < self.trading_costs['min_trade_amount']:
                trade_info['reason'] = 'Sell amount is less than min_trade_amount'
            else:
                costs = self.calculate_trading_costs(btc_to_sell, current_price, 'taker')
                effective_price = costs['effective_price_sell']
                proceeds = (btc_to_sell * effective_price) - costs['total_cost']

                if proceeds > 0:
                    # 🔧 ВИКОНУЄМО ТОРГІВЛЮ
                    self.balance += proceeds
                    self.btc_holdings -= btc_to_sell

                    # Оновлюємо статистику
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
                        'effective_price': effective_price,
                        'position_size_used': position_size
                    })

                    self.positions.append({
                        'type': 'SELL', 'price': effective_price, 'amount': btc_to_sell,
                        'timestamp': timestamp, 'signal_type': signal_type, 'confidence': confidence
                    })
                    executed_this_step = True

                    # 📊 ДІАГНОСТИКА з новою інформацією
                    new_allocation = self.calculate_current_allocation(current_price)
                    print(f"🔴 SELL: {btc_to_sell:.6f} BTC → ${proceeds:.0f} @ ${effective_price:.0f}")
                    print(
                        f"   Position size: {position_size:.1%}, BTC ratio: {allocation['btc_ratio']:.1%} → {new_allocation['btc_ratio']:.1%}")
                else:
                    trade_info['reason'] = "Proceeds <= 0 due to high costs"

        # 🔧 КРИТИЧНО ВАЖЛИВО: Розраховуємо portfolio_value ПІСЛЯ торгівлі
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
        """📊 Детальна статистика торгівлі з новими метриками"""
        if not self.trade_history:
            initial_stats = self._get_initial_trading_stats()
            initial_stats.update({
                'avg_confidence': 0,
                'portfolio_volatility': 0,
                'cost_ratio': 0,
                'total_trading_costs': 0,
                'block_rate': 0,
                'execution_rate': 0
            })
            return initial_stats

        executed_trades_history = [t for t in self.trade_history if t.get('executed', False)]
        portfolio_values = [t['portfolio_value'] for t in self.trade_history if 'portfolio_value' in t]

        # 🆕 Статистика заблокованих трейдів
        blocked_trades = len([t for t in self.trade_history if t.get('blocked_by_position_management', False)])

        stats = {
            'total_trades': self.trading_stats['total_trades'],
            'buy_trades': self.trading_stats['buy_trades'],
            'sell_trades': self.trading_stats['sell_trades'],
            'scalping_trades': self.trading_stats['scalping_trades'],
            'blocked_trades': blocked_trades,  # 🆕
            'total_fees_paid': self.trading_stats['total_fees_paid'],
            'total_slippage_cost': self.trading_stats['total_slippage_cost'],
            'total_trading_costs': self.trading_stats['total_fees_paid'] + self.trading_stats['total_slippage_cost'],
            'avg_confidence': np.mean(
                [t.get('confidence', 0) for t in executed_trades_history]) if executed_trades_history else 0,
            'portfolio_volatility': np.std(np.diff(portfolio_values)) if len(portfolio_values) > 1 else 0,
            'block_rate': blocked_trades / len(self.trade_history) if self.trade_history else 0,  # 🆕
            'execution_rate': len(executed_trades_history) / len(self.trade_history) if self.trade_history else 0,  # 🆕
        }

        # Розрахунок cost_ratio від поточного балансу або початкового
        current_or_initial_balance = self.balance if self.balance > 0 else self.initial_balance
        if current_or_initial_balance > 0:
            stats['cost_ratio'] = (stats['total_trading_costs'] / current_or_initial_balance) * 100
        else:
            stats['cost_ratio'] = 0

        return stats

    def get_position_management_stats(self):
        """
        🆕 Статистика менеджменту позицій
        """
        if not self.trade_history:
            return {
                'avg_btc_ratio': 0,
                'avg_cash_ratio': 0,
                'position_size_stats': {},
                'blocked_reasons': {}
            }

        # Збираємо статистику по розподілу портфеля
        allocations = []
        position_sizes = []
        blocked_reasons = {}

        for trade in self.trade_history:
            if 'allocation_before' in trade:
                allocations.append(trade['allocation_before'])

            if trade.get('executed', False) and 'position_size_used' in trade:
                position_sizes.append(trade['position_size_used'])

            if trade.get('blocked_by_position_management', False):
                reason = trade.get('reason', 'Unknown')
                blocked_reasons[reason] = blocked_reasons.get(reason, 0) + 1

        # Розрахунок середніх значень
        avg_btc_ratio = np.mean([a['btc_ratio'] for a in allocations]) if allocations else 0
        avg_cash_ratio = np.mean([a['cash_ratio'] for a in allocations]) if allocations else 0

        position_size_stats = {}
        if position_sizes:
            position_size_stats = {
                'mean': np.mean(position_sizes),
                'std': np.std(position_sizes),
                'min': np.min(position_sizes),
                'max': np.max(position_sizes)
            }

        return {
            'avg_btc_ratio': avg_btc_ratio,
            'avg_cash_ratio': avg_cash_ratio,
            'position_size_stats': position_size_stats,
            'blocked_reasons': blocked_reasons,
            'total_allocations_tracked': len(allocations)
        }

    def run_backtest(self, historical_data, feature_names_for_model):
        """
        🔧 ПОКРАЩЕНИЙ бектест з новим менеджментом позицій
        """
        print("⚖️ Запуск ПОКРАЩЕНОГО бектесту з інтелектуальним менеджментом позицій...")
        self.reset_state()

        results_log = []
        n_points_to_process = min(500, len(historical_data))
        step = max(1, len(historical_data) // n_points_to_process)

        initial_balance_display = self.balance
        initial_btc_display = self.btc_holdings
        signals_generated_non_hold = 0

        # Перевірка наявності необхідних колонок
        required_cols = ['close', 'timestamp'] + feature_names_for_model
        missing_cols = [col for col in required_cols if col not in historical_data.columns]
        if missing_cols:
            raise ValueError(f"Відсутні необхідні колонки в historical_data: {', '.join(missing_cols)}")

        # 📊 Додаткова діагностика
        trade_impacts = []
        rebalance_signals = 0
        position_blocks = 0

        print(f"📊 ПОЧАТКОВИЙ СТАН ПОРТФЕЛЯ:")
        print(f"   💰 Balance: ${self.balance:,.2f}")
        print(f"   ₿ BTC Holdings: {self.btc_holdings:.6f}")
        print(f"   📈 Total Portfolio: ${self.balance:,.2f}")
        print(f"   🎯 Max BTC allocation: {self.trading_params['max_btc_allocation']:.1%}")
        print(f"   💵 Min cash reserve: {self.trading_params['min_cash_reserve']:.1%}")
        print(f"📊 Обробляємо {len(historical_data) // step} точок з {len(historical_data)} (крок: {step})")

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

            # 🔧 Стан портфеля ДО сигналу
            portfolio_before_signal = self.balance + (self.btc_holdings * current_price)
            allocation_before = self.calculate_current_allocation(current_price)

            # 🆕 Перевірка потреби в ребалансуванні
            rebalance_signal, deviation = self.should_rebalance(current_price)
            if rebalance_signal:
                rebalance_signals += 1
                # Можна додати логіку примусового ребалансування тут

            # Генерація сигналу
            signal, predicted_price, price_change_pct, confidence, signal_type = \
                self.generate_aggressive_signal(
                    features=model_input_features_for_prediction,
                    current_price=current_price,
                    technical_indicators=technical_indicators_dict
                )

            if signal != 'HOLD':
                signals_generated_non_hold += 1

            # 🔧 ВИКОНАННЯ ТОРГІВЛІ з новим менеджментом
            trade_info = self.execute_trade_with_costs(
                signal=signal,
                current_price=current_price,
                timestamp=timestamp,
                predicted_price=predicted_price,
                confidence=confidence,
                signal_type=signal_type
            )

            # Лічильник заблокованих позицій
            if trade_info.get('blocked_by_position_management', False):
                position_blocks += 1

            # 🔧 Стан портфеля ПІСЛЯ торгівлі
            portfolio_after_trade = trade_info['portfolio_value']
            allocation_after = self.calculate_current_allocation(current_price)

            # 📊 Діагностика значних змін портфеля
            portfolio_change = portfolio_after_trade - portfolio_before_signal

            if trade_info.get('executed', False):
                trade_impact = {
                    'timestamp': timestamp,
                    'signal': signal,
                    'price': current_price,
                    'portfolio_before': portfolio_before_signal,
                    'portfolio_after': portfolio_after_trade,
                    'change': portfolio_change,
                    'btc_ratio_before': allocation_before['btc_ratio'],
                    'btc_ratio_after': allocation_after['btc_ratio'],
                    'position_size': trade_info.get('position_size_used', 0)
                }
                trade_impacts.append(trade_impact)

            # Логування результату з додатковою інформацією
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
                'btc_ratio': allocation_after['btc_ratio'],
                'cash_ratio': allocation_after['cash_ratio'],
                'position_size_used': trade_info.get('position_size_used', 0),
                'blocked_by_position_management': trade_info.get('blocked_by_position_management', False)
            }

            results_log.append(log_entry)

            # Прогрес з додатковою інформацією
            if i > 0 and (i // step) % 50 == 0:
                current_loop_trades = self.trading_stats['total_trades']
                progress_pct = (i + step) / len(historical_data) * 100
                current_allocation = self.calculate_current_allocation(current_price)
                print(f"🔄 {progress_pct:.1f}% | Сигналів: {signals_generated_non_hold} | "
                      f"Трейдів: {current_loop_trades} | Заблоковано: {position_blocks} | "
                      f"BTC: {current_allocation['btc_ratio']:.1%}")

        # Створення результатів
        results_df = pd.DataFrame(results_log)
        final_trading_stats = self.get_trading_statistics()
        position_management_stats = self.get_position_management_stats()

        # 🔧 РОЗШИРЕНА ДІАГНОСТИКА з новою інформацією
        print(f"\n⚖️ РЕЗУЛЬТАТИ ПОКРАЩЕНОГО БЕКТЕСТУ:")
        print(f"✅ Сигналів згенеровано (не HOLD): {signals_generated_non_hold}")
        print(f"✅ Трейдів виконано: {final_trading_stats.get('total_trades', 0)}")
        print(f"✅ BUY трейдів: {final_trading_stats.get('buy_trades', 0)}")
        print(f"✅ SELL трейдів: {final_trading_stats.get('sell_trades', 0)}")
        print(f"✅ Скальпінг трейдів: {final_trading_stats.get('scalping_trades', 0)}")
        print(f"🚫 Заблоковано позиційним менеджментом: {final_trading_stats.get('blocked_trades', 0)}")
        print(f"🔄 Сигналів ребалансування: {rebalance_signals}")

        if not results_df.empty:
            final_portfolio = results_df['portfolio_value'].iloc[-1]
            final_allocation = self.calculate_current_allocation(results_df['price'].iloc[-1])
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

            print(f"\n🎯 МЕНЕДЖМЕНТ ПОЗИЦІЙ:")
            print(f"📊 Фінальний розподіл: BTC {final_allocation['btc_ratio']:.1%}, "
                  f"Готівка {final_allocation['cash_ratio']:.1%}")
            print(f"📊 Середній розподіл: BTC {position_management_stats['avg_btc_ratio']:.1%}, "
                  f"Готівка {position_management_stats['avg_cash_ratio']:.1%}")
            print(f"🚫 Коефіцієнт блокування: {final_trading_stats.get('block_rate', 0):.1%}")
            print(f"✅ Коефіцієнт виконання: {final_trading_stats.get('execution_rate', 0):.1%}")

            # Причини блокування
            blocked_reasons = position_management_stats.get('blocked_reasons', {})
            if blocked_reasons:
                print(f"🚫 Причини блокування трейдів:")
                for reason, count in blocked_reasons.items():
                    print(f"   - {reason}: {count}")

            # Оцінка ефективності менеджменту позицій
            if final_allocation['btc_ratio'] > 0.9:
                print("⚠️ УВАГА: Кінцевий розподіл занадто зміщений до BTC!")
            elif 0.3 <= final_allocation['btc_ratio'] <= 0.7:
                print("✅ ВІДМІННО: Збалансований фінальний розподіл портфеля!")

            if final_trading_stats.get('block_rate', 0) > 0.5:
                print("⚠️ УВАГА: Високий коефіцієнт блокування! Можливо треба послабити обмеження.")
            elif final_trading_stats.get('block_rate', 0) > 0:
                print("✅ ДОБРЕ: Менеджмент позицій активно працює!")

        return results_df


# Функції, що раніше були в fixed_aggressive_backtesting.py
def create_empty_backtest_result_standalone():
    """Створює порожній результат бектесту (стандартна версія)"""
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
        'btc_ratio': [0],
        'cash_ratio': [1],
        'blocked_by_position_management': [False]
    })


def plot_backtest_results_standalone(backtest_results_df, system_name="Покращена Агресивна стратегія"):
    """
    📊 ПОКРАЩЕНА візуалізація результатів бектесту з інформацією про менеджмент позицій
    """
    try:
        if backtest_results_df.empty:
            print("❌ Немає даних для візуалізації")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

        # График 1: Ціна та сигнали з інформацією про блокування
        ax1.plot(backtest_results_df['timestamp'], backtest_results_df['price'],
                 label='Ціна BTC', color='blue', linewidth=1.5, alpha=0.8)

        # Виконані трейди
        executed_trades_df = backtest_results_df[backtest_results_df['executed'] == True]
        buy_trades = executed_trades_df[executed_trades_df['signal'] == 'BUY']
        sell_trades = executed_trades_df[executed_trades_df['signal'] == 'SELL']

        # Заблоковані трейди
        blocked_trades_df = backtest_results_df[backtest_results_df.get('blocked_by_position_management', False)]

        if not buy_trades.empty:
            ax1.scatter(buy_trades['timestamp'], buy_trades['price'],
                        marker='^', color='darkgreen', s=60, label=f'BUY ({len(buy_trades)})', zorder=5)

        if not sell_trades.empty:
            ax1.scatter(sell_trades['timestamp'], sell_trades['price'],
                        marker='v', color='darkred', s=60, label=f'SELL ({len(sell_trades)})', zorder=5)

        if not blocked_trades_df.empty:
            ax1.scatter(blocked_trades_df['timestamp'], blocked_trades_df['price'],
                        marker='x', color='gray', s=30, label=f'BLOCKED ({len(blocked_trades_df)})', alpha=0.7)

        ax1.set_title(
            f'🎯 {system_name}: Виконано {len(executed_trades_df)} трейдів, заблоковано {len(blocked_trades_df)}',
            fontsize=14, fontweight='bold')
        ax1.set_ylabel('Ціна (USD)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # График 2: Портфель з додатковою інформацією
        ax2.plot(backtest_results_df['timestamp'], backtest_results_df['portfolio_value'],
                 label='Портфель Стратегії', color='purple', linewidth=2.5)

        if len(backtest_results_df) > 1:
            initial_price = backtest_results_df['price'].iloc[0]
            initial_portfolio = backtest_results_df['portfolio_value'].iloc[0]
            btc_equivalent_full = initial_portfolio / initial_price
            buy_hold_values_full = btc_equivalent_full * backtest_results_df['price']

            ax2.plot(backtest_results_df['timestamp'], buy_hold_values_full,
                     label='Buy & Hold (100% BTC)', color='gray', linestyle='--', linewidth=2)

            # Результати
            final_strategy = backtest_results_df['portfolio_value'].iloc[-1]
            final_buy_hold = buy_hold_values_full.iloc[-1]
            difference = final_strategy - final_buy_hold
            difference_pct = (difference / final_buy_hold) * 100

            result_text = f'vs Buy&Hold: ${difference:+,.0f} ({difference_pct:+.1f}%)'
            text_color = 'green' if difference > 0 else 'red'
            ax2.text(0.02, 0.98, result_text, transform=ax2.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor=text_color, alpha=0.2), fontweight='bold')

        ax2.set_title('💰 Динаміка портфеля з покращеним менеджментом', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Вартість (USD)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # График 3: Розподіл портфеля (BTC ratio в часі)
        if 'btc_ratio' in backtest_results_df.columns:
            ax3.plot(backtest_results_df['timestamp'], backtest_results_df['btc_ratio'] * 100,
                     color='orange', linewidth=2, label='BTC %')
            ax3.axhline(y=75, color='red', linestyle='--', alpha=0.7, label='Max BTC (75%)')
            ax3.axhline(y=15, color='green', linestyle='--', alpha=0.7, label='Min Cash (85% BTC)')
            ax3.axhline(y=50, color='blue', linestyle=':', alpha=0.7, label='Target (50%)')

            ax3.set_title('📊 Розподіл портфеля (BTC %)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('BTC відсоток (%)')
            ax3.set_ylim(0, 100)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'Дані про розподіл\nпортфеля недоступні',
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)

        # График 4: Розмір позицій
        position_sizes = backtest_results_df[backtest_results_df['executed'] == True]['position_size_used']
        if not position_sizes.empty and 'position_size_used' in backtest_results_df.columns:
            ax4.hist(position_sizes * 100, bins=15, alpha=0.7, color='skyblue', edgecolor='black')

            mean_size = position_sizes.mean() * 100
            ax4.axvline(mean_size, color='red', linestyle='--', linewidth=2,
                        label=f'Середній: {mean_size:.1f}%')

            ax4.set_title(f'📏 Розподіл розмірів позицій (n={len(position_sizes)})', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Розмір позиції (%)')
            ax4.set_ylabel('Кількість')
            ax4.grid(True, alpha=0.3)
            ax4.legend()

            # Статистика
            stats_text = f'Min: {position_sizes.min() * 100:.1f}%\nMax: {position_sizes.max() * 100:.1f}%\nStd: {position_sizes.std() * 100:.1f}%'
            ax4.text(0.98, 0.98, stats_text, transform=ax4.transAxes, verticalalignment='top',
                     horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Дані про розміри\nпозицій недоступні',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)

        plt.tight_layout()
        plt.show()
        print("✅ ПОКРАЩЕНУ візуалізацію бектесту створено")

    except Exception as e:
        print(f"❌ Помилка візуалізації: {e}")
        import traceback
        print(traceback.format_exc())


# --- Приклад використання ---
if __name__ == '__main__':
    print("Запуск прикладу покращеної системи AggressiveTradingSystem з інтелектуальним менеджментом позицій...")


    # Тестовий код залишається тим же...


    # 1. Створення фіктивної моделі прогнозування
    class DummyPredictionModel:
        def predict(self, features):
            # Проста логіка: якщо остання ціна (з фіч) вища за середню, прогнозуємо ріст, інакше падіння
            # Це дуже спрощено, реальна модель буде складнішою.
            # Припустимо, що 'close_lag_1' - це одна з ознак
            if 'close_lag_1' in features.columns and 'sma_20' in features.columns:
                current_close = features['close_lag_1'].iloc[0]
                sma_20 = features['sma_20'].iloc[0]
                if current_close > sma_20 + current_close * 0.001:  # Невеликий поріг
                    return [current_close * 1.005]  # Прогноз росту на 0.5%
                elif current_close < sma_20 - current_close * 0.001:
                    return [current_close * 0.995]  # Прогноз падіння на 0.5%
            return [features['close_lag_1'].iloc[
                        0] if 'close_lag_1' in features.columns else 50000]  # За замовчуванням - без змін


    dummy_model = DummyPredictionModel()

    # 2. Створення екземпляра системи
    # initial_balance можна взяти з CONFIG або передати явно
    trading_system = AggressiveTradingSystem(
        prediction_model=dummy_model,
        initial_balance=CONFIG['trading']['initial_balance']  #
    )

    # 3. Підготовка фіктивних історичних даних для бектесту
    # Ці дані мають містити колонки 'timestamp', 'close' та всі ознаки,
    # які очікує ваша модель ('feature_names_for_model') та технічні індикатори.
    dates = pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00',
                            '2023-01-01 03:00:00', '2023-01-01 04:00:00', '2023-01-01 05:00:00',
                            '2023-01-01 06:00:00', '2023-01-01 07:00:00', '2023-01-01 08:00:00',
                            '2023-01-01 09:00:00', '2023-01-01 10:00:00'])

    # Приклад цін та індикаторів
    prices = [50000, 50100, 50050, 50200, 50150, 49900, 49800, 50000, 50300, 50250, 50400]
    historical_data_dict = {
        'timestamp': dates,
        'close': prices,
        'open': [p - np.random.randint(-50, 50) for p in prices],
        'high': [p + np.random.randint(0, 100) for p in prices],
        'low': [p - np.random.randint(0, 100) for p in prices],
        'volume': [np.random.randint(10, 100) for _ in prices],
        'rsi': [30, 35, 32, 45, 40, 65, 70, 55, 60, 58, 62],  # Приклад RSI
        'macd': [10, 12, 8, 15, 12, -5, -8, 5, 18, 15, 20],  # Приклад MACD
        'macd_signal': [8, 9, 8.5, 10, 11, -2, -5, 0, 10, 12, 15],  # Приклад MACD Signal
        'adx': [20, 22, 21, 25, 24, 28, 30, 26, 27, 26, 28],  # Приклад ADX
        'volume_ratio_20': [1.0, 1.1, 1.05, 1.2, 1.15, 0.9, 0.95, 1.3, 1.4, 1.35, 1.25],  # Приклад Volume Ratio
        # Додамо фіктивні ознаки, які очікує модель
        'close_lag_1': [50000] + prices[:-1],  # Зсунута ціна закриття
        'sma_20': [p - np.random.uniform(-100, 100) for p in prices]  # Приклад SMA_20
        # ... додайте інші ознаки, якщо ваша модель їх використовує
    }
    historical_df = pd.DataFrame(historical_data_dict)

    # Назви колонок, які модель використовує для прогнозу (приклад)
    # Ці назви мають відповідати тим, на яких навчалася модель.
    feature_names = ['close_lag_1', 'sma_20', 'rsi', 'macd', 'volume_ratio_20']  # Приклад

    # Переконаємося, що всі feature_names є в historical_df
    missing_model_features = [f for f in feature_names if f not in historical_df.columns]
    if missing_model_features:
        print(f"ПОПЕРЕДЖЕННЯ: Відсутні деякі ознаки для моделі в historical_df: {missing_model_features}")
        # Можна додати фіктивні значення для відсутніх колонок, якщо це потрібно для тестування
        for col in missing_model_features:
            historical_df[col] = 0

            # 4. Запуск бектесту
    try:
        backtest_results_df = trading_system.run_backtest(historical_df, feature_names)

        # 5. Візуалізація результатів (якщо є дані)
        if not backtest_results_df.empty:
            plot_backtest_results_standalone(backtest_results_df, "Тестова Агресивна Система")

        # 6. Отримання статистики
        final_stats = trading_system.get_trading_statistics()
        print("\n📊 Фінальна статистика торгівлі:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        # Збереження результатів бектесту
        if not backtest_results_df.empty:
            ensure_dir(CONFIG['saving']['results_dir'])  #
            results_filename = f"TEST_aggressive_backtest_results_{timestamp_to_string()}.csv"
            results_path = os.path.join(CONFIG['saving']['results_dir'], results_filename)  #
            try:
                backtest_results_df.to_csv(results_path, index=False)
                print(f"✓ Результати тестового бектесту збережено в {results_path}")
            except Exception as e_save:
                print(f"❌ Помилка при збереженні результатів тестового бектесту: {e_save}")


    except ValueError as ve:
        print(f"Помилка значення під час запуску бектесту: {ve}")
    except Exception as ex:
        print(f"Непередбачена помилка під час запуску бектесту: {ex}")
        print(traceback.format_exc())