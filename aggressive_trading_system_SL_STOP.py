# aggressive_trading_system_SL_STOP.py - СПРОЩЕНА ВЕРСІЯ З ШТУЧНИМИ ШОРТАМИ
# 🔧 Видалено плече, ліквідацію, позичкові відсотки - тільки штучні шорти

import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from aggressive_backtesting_SL_STOP import capture_pnl_from_log

try:
    from config import CONFIG
except ImportError:
    print("УВАГА: Не вдалося імпортувати CONFIG з файлу config.py. Використовується базовий конфіг.")
    CONFIG = {
        'data': {
            'symbol': 'BTC-USD',
            'timeframe': '1h',
            'period': '1y',
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
            'position_size': 0.1,
            'confidence_threshold': 0.4,
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'enable_short_positions': True,
            'short_stop_loss': 0.025,
            'short_take_profit': 0.04,
            'max_short_positions': 3,
            'allow_simultaneous_long_short': True,
            'short_confidence_multiplier': 3.0,
            'max_open_positions': 4,
            'position_timeout_hours': 36,
        },
        'saving': {
            'models_dir': 'models/saved',
            'results_dir': 'results',
            'data_dir': 'data'
        }
    }


class AggressiveTradingSystem:
    """
    🔧 СПРОЩЕНА агресивна торгова система зі штучними шортами
    """

    def __init__(self, prediction_model, initial_balance=None):
        """
        Ініціалізація торгової системи зі спрощеними параметрами
        """
        self.model = prediction_model
        if initial_balance is None:
            self.initial_balance = CONFIG['trading']['initial_balance']
        else:
            self.initial_balance = initial_balance

        # 🔧 СПРОЩЕНІ параметри для штучних шортів
        self.enable_shorts = CONFIG['trading'].get('enable_short_positions', True)

        # Стан стратегії
        self.balance = self.initial_balance
        self.btc_holdings = 0
        self.positions = []
        self.trade_history = []
        self.trading_stats = self._get_initial_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()

        # 🔧 ПОКРАЩЕНІ агресивні параметри
        self.trading_params = {
            'base_confidence_threshold': CONFIG['trading'].get('confidence_threshold', 1.0),
            'min_confidence_threshold': 0.5,
            'position_size_base': CONFIG['trading'].get('position_size', 0.01),
            'max_position_size': 0.02,
            'min_position_size': 0.005,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'enable_scalping': False,
            'scalping_threshold': 0.3,
            'short_threshold_multiplier': 5.0,
            'allow_simultaneous_positions': CONFIG['trading'].get('allow_simultaneous_long_short', True),
        }

        # 🔧 СПРОЩЕНІ торгові витрати
        self.trading_costs = {
            'maker_fee': 0.0004,
            'taker_fee': 0.0006,
            'slippage_pct': 0.00005,
            'min_trade_amount': 1000,
            'spread_impact': 0.0002,
        }

        # 🔧 ПОКРАЩЕНІ параметри ризик-менеджменту
        self.risk_params = {
            'stop_loss_pct': CONFIG['trading']['stop_loss'],
            'take_profit_pct': CONFIG['trading']['take_profit'],
            'trailing_stop_enabled': False,
            'trailing_stop_pct': 0.03,
            'max_open_positions': CONFIG['trading'].get('max_open_positions', 2),
            'position_timeout_hours': CONFIG['trading'].get('position_timeout_hours', 168),
            'short_stop_loss_pct': CONFIG['trading'].get('short_stop_loss', 0.02),
            'short_take_profit_pct': CONFIG['trading'].get('short_take_profit', 0.04),
            'max_short_positions': CONFIG['trading'].get('max_short_positions', 1),
        }

        print(f"🔧 Система ініціалізована зі штучними шортами:")
        print(f"   Max позицій: {self.risk_params['max_open_positions']}")
        print(f"   Поріг впевненості: {self.trading_params['base_confidence_threshold']}")
        print(f"   Timeout позицій: {self.risk_params['position_timeout_hours']} годин")

    def get_portfolio_value(self, current_price):
        """✅ Спрощений розрахунок без leverage"""
        return self.balance + (self.btc_holdings * current_price)

    def _get_initial_trading_stats(self):
        """Початкова статистика торгівлі"""
        return {
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
            'short_trades': 0,
            'short_cover_trades': 0,
            'short_pnl': 0,
            'long_pnl': 0,
        }

    def _get_initial_market_conditions(self):
        """Початкові ринкові умови"""
        return {
            'volatility_threshold': 0.015,
            'volume_spike_multiplier': 1.2,
            'trend_momentum': 0,
            'recent_prices': [],
            'price_velocity': 0,
            'bear_market_indicator': 0,
            'volatility_spike': False,
        }

    def reset_state(self):
        """Скидає стан стратегії до початкового"""
        self.balance = self.initial_balance
        self.btc_holdings = 0
        self.positions = []
        self.trade_history = []
        self.trading_stats = self._get_initial_trading_stats()
        self.market_conditions = self._get_initial_market_conditions()
        print("📈 Стан стратегії скинуто до початкових значень (ШТУЧНІ ШОРТИ)")

    def calculate_trading_costs(self, amount, price, trade_type='taker'):
        """💰 Спрощений розрахунок торгових витрат"""
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
        """🎯 Оновлення ринкових умов"""
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

                # Індикатор ведмежого ринку для шортів
                if self.market_conditions['trend_momentum'] < -2.0:
                    self.market_conditions['bear_market_indicator'] = min(1.0, abs(
                        self.market_conditions['trend_momentum']) / 5.0)
                else:
                    self.market_conditions['bear_market_indicator'] = 0

        # Детекція волатільності
        if technical_indicators and len(self.market_conditions['recent_prices']) >= 10:
            volatility = technical_indicators.get('volatility', 0.02)
            self.market_conditions['volatility_spike'] = volatility > self.market_conditions[
                'volatility_threshold'] * 1.5

    def check_stop_loss_take_profit(self, current_price, timestamp, high_price=None, low_price=None):
        """🛡️ ВИПРАВЛЕНА перевірка стоп-лос, тейк-профіт та TIMEOUT з обов'язковим PnL"""
        positions_to_close = []

        for i, position in enumerate(self.positions):
            close_reason = None
            should_close = False
            actual_exit_price = current_price

            # Перевіряємо вік позиції (TIMEOUT)
            position_age_hours = (timestamp - position['timestamp']).total_seconds() / 3600
            timeout_hours = self.risk_params['position_timeout_hours']

            # 🆕 ОБОВ'ЯЗКОВЕ ЗАКРИТТЯ ПО TIMEOUT
            if position_age_hours >= timeout_hours:
                should_close = True
                close_reason = 'timeout'
                actual_exit_price = current_price
                self.trading_stats['position_timeouts'] += 1
                print(
                    f"⏰ TIMEOUT: Позиція {position['type']} відкрита {position_age_hours:.1f}h (ліміт: {timeout_hours}h)")

            # Якщо не timeout, перевіряємо стоп-лос/тейк-профіт
            elif position['type'] == 'BUY':  # ЛОНГ ПОЗИЦІЯ
                entry_price = position['entry_price']
                stop_loss_price = entry_price * (1 - self.risk_params['stop_loss_pct'])
                take_profit_price = entry_price * (1 + self.risk_params['take_profit_pct'])

                if high_price is not None and low_price is not None:
                    if high_price >= take_profit_price:
                        should_close = True
                        close_reason = 'take_profit'
                        actual_exit_price = take_profit_price
                        self.trading_stats['take_profit_triggered'] += 1
                        print(
                            f"🎯 LONG TP: Entry ${entry_price:.0f} → Exit ${take_profit_price:.0f} (+{self.risk_params['take_profit_pct'] * 100:.1f}%)")
                    elif low_price <= stop_loss_price:
                        should_close = True
                        close_reason = 'stop_loss'
                        actual_exit_price = stop_loss_price
                        self.trading_stats['stop_loss_triggered'] += 1
                        print(
                            f"🛡️ LONG SL: Entry ${entry_price:.0f} → Exit ${stop_loss_price:.0f} (-{self.risk_params['stop_loss_pct'] * 100:.1f}%)")
                else:
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

            elif position['type'] == 'SHORT':  # ШТУЧНА ШОРТ ПОЗИЦІЯ
                entry_price = position['entry_price']
                stop_loss_price = entry_price * (1 + self.risk_params['short_stop_loss_pct'])
                take_profit_price = entry_price * (1 - self.risk_params['short_take_profit_pct'])

                if high_price is not None and low_price is not None:
                    if low_price <= take_profit_price:
                        should_close = True
                        close_reason = 'take_profit'
                        actual_exit_price = take_profit_price
                        self.trading_stats['take_profit_triggered'] += 1
                        print(
                            f"🎯 SHORT TP: Entry ${entry_price:.0f} → Exit ${take_profit_price:.0f} (+{self.risk_params['short_take_profit_pct'] * 100:.1f}%)")
                    elif high_price >= stop_loss_price:
                        should_close = True
                        close_reason = 'stop_loss'
                        actual_exit_price = stop_loss_price
                        self.trading_stats['stop_loss_triggered'] += 1
                        print(
                            f"🛡️ SHORT SL: Entry ${entry_price:.0f} → Exit ${stop_loss_price:.0f} (-{self.risk_params['short_stop_loss_pct'] * 100:.1f}%)")
                else:
                    if current_price <= take_profit_price:
                        should_close = True
                        close_reason = 'take_profit'
                        actual_exit_price = current_price
                        self.trading_stats['take_profit_triggered'] += 1
                    elif current_price >= stop_loss_price:
                        should_close = True
                        close_reason = 'stop_loss'
                        actual_exit_price = current_price
                        self.trading_stats['stop_loss_triggered'] += 1

            # 🆕 ДОДАТКОВІ УМОВИ ЗАКРИТТЯ ДЛЯ ЗБІЛЬШЕННЯ PnL ЗАПИСІВ
            if not should_close:
                # Закриваємо при великій втраті (аварійний стоп)
                if position['type'] == 'BUY':
                    current_loss_pct = (position['entry_price'] - current_price) / position['entry_price']
                    if current_loss_pct > 0.15:  # 15% втрата
                        should_close = True
                        close_reason = 'emergency_stop'
                        actual_exit_price = current_price
                        print(f"🚨 EMERGENCY STOP LONG: втрата {current_loss_pct * 100:.1f}%")

                elif position['type'] == 'SHORT':
                    current_loss_pct = (current_price - position['entry_price']) / position['entry_price']
                    if current_loss_pct > 0.15:  # 15% втрата
                        should_close = True
                        close_reason = 'emergency_stop'
                        actual_exit_price = current_price
                        print(f"🚨 EMERGENCY STOP SHORT: втрата {current_loss_pct * 100:.1f}%")

                # Закриваємо при великому прибутку (profit taking)
                if position['type'] == 'BUY':
                    current_profit_pct = (current_price - position['entry_price']) / position['entry_price']
                    if current_profit_pct > 0.12:  # 12% прибуток
                        should_close = True
                        close_reason = 'profit_taking'
                        actual_exit_price = current_price
                        print(f"💰 PROFIT TAKING LONG: прибуток {current_profit_pct * 100:.1f}%")

                elif position['type'] == 'SHORT':
                    current_profit_pct = (position['entry_price'] - current_price) / position['entry_price']
                    if current_profit_pct > 0.12:  # 12% прибуток
                        should_close = True
                        close_reason = 'profit_taking'
                        actual_exit_price = current_price
                        print(f"💰 PROFIT TAKING SHORT: прибуток {current_profit_pct * 100:.1f}%")

            if should_close:
                positions_to_close.append({
                    'index': i,
                    'position': position,
                    'reason': close_reason,
                    'current_price': actual_exit_price,
                    'timestamp': timestamp
                })

        # Закриваємо позиції (ЗІ ЗВОРОТНИМ ПОРЯДКОМ для правильних індексів)
        for close_info in reversed(positions_to_close):
            self._close_position(close_info)

        return len(positions_to_close)



    def _close_position(self, close_info):
        """🔒 Закриття позицій зі штучними шортами та правильним записуванням PnL"""
        position = close_info['position']
        current_price = close_info['current_price']
        timestamp = close_info['timestamp']
        reason = close_info['reason']

        amount_to_close = position['amount']

        # 🔧 ІНІЦІАЛІЗУЄМО ЗМІННІ НА ПОЧАТКУ
        pnl = 0
        pnl_pct = 0

        if position['type'] == 'BUY':  # ЛОНГ ПОЗИЦІЯ
            costs = self.calculate_trading_costs(amount_to_close, current_price, 'taker')
            effective_price = costs['effective_price_sell']
            proceeds = (amount_to_close * effective_price) - costs['total_cost']

            if proceeds > 0 and amount_to_close <= self.btc_holdings:
                self.balance += proceeds
                self.btc_holdings -= amount_to_close

                entry_value = position['amount'] * position['entry_price']
                exit_value = proceeds
                pnl = exit_value - entry_value
                pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

                print(f" ЗАКРИТТЯ LONG позиції ({reason.upper()}): {amount_to_close:.6f} BTC @ ${effective_price:.0f}")
                print(f"   PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%) | Balance: ${self.balance:.0f}")

                # 🆕 ВАЖЛИВО: Захоплюємо PnL для логів
                capture_pnl_from_log(f"PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%)")

                self.trading_stats['total_fees_paid'] += costs['fee']
                self.trading_stats['total_slippage_cost'] += costs['slippage']
                self.trading_stats['sell_trades'] += 1
                self.trading_stats['total_trades'] += 1
                self.trading_stats['long_pnl'] += pnl

        elif position['type'] == 'SHORT':  # 🔧 ШТУЧНИЙ ШОРТ
            costs = self.calculate_trading_costs(amount_to_close, current_price, 'taker')
            effective_price = costs['effective_price_sell']

            # Штучний шорт: просто розраховуємо PnL як різницю цін
            entry_value = position['amount'] * position['entry_price']
            # Для шорту: прибуток коли ціна падає
            pnl = (position['entry_price'] - effective_price) * amount_to_close - costs['total_cost']
            pnl_pct = (pnl / entry_value) * 100 if entry_value > 0 else 0

            print(f" ЗАКРИТТЯ SHORT позиції ({reason.upper()}): {amount_to_close:.6f} BTC @ ${effective_price:.0f}")
            print(f"   🔧 ШТУЧНИЙ ШОРТ PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%)")

            # 🆕 ВАЖЛИВО: Захоплюємо PnL для логів
            capture_pnl_from_log(f"ШТУЧНИЙ ШОРТ PnL: ${pnl:+.0f} ({pnl_pct:+.2f}%)")

            # Просто додаємо/віднімаємо PnL від балансу
            self.balance += pnl

            self.trading_stats['total_fees_paid'] += costs['fee']
            self.trading_stats['total_slippage_cost'] += costs['slippage']
            self.trading_stats['short_cover_trades'] += 1
            self.trading_stats['total_trades'] += 1
            self.trading_stats['short_pnl'] += pnl

        # 🆕 ВАЖЛИВО: Створюємо запис про трейд з PnL
        trade_info = {
            'timestamp': timestamp,
            'signal': 'SELL' if position['type'] == 'BUY' else 'COVER',
            'signal_type': f'close_{reason}',
            'price': current_price,
            'predicted_price': current_price,
            'confidence': 1.0,
            'executed': True,
            'amount': amount_to_close,
            'total_cost': costs['total_cost'],
            'fee': costs['fee'],
            'slippage': costs['slippage'],
            'effective_price': effective_price,
            'balance_after': self.balance,
            'btc_after': self.btc_holdings,
            'portfolio_value': self.balance + (self.btc_holdings * current_price),
            'close_reason': reason,
            'position_type': position['type'],
            'entry_price': position['entry_price'],
            'pnl': pnl,  # ← Тепер безпечно записуємо PnL
            'pnl_pct': pnl_pct,  # ← Тепер безпечно записуємо PnL%
            'portfolio_change': pnl  # ← Також додаємо як portfolio_change
        }

        # 🆕 ВАЖЛИВО: Додаємо запис до історії торгівлі
        self.trade_history.append(trade_info)

        # Видаляємо позицію з активних
        self.positions.pop(close_info['index'])

        # 🆕 ПОВЕРТАЄМО ІНФОРМАЦІЮ ПРО ЗАКРИТТЯ (для використання в бектесті)
        return trade_info

    def generate_aggressive_signal(self, features, current_price, technical_indicators):
        """🔥 Генерація сигналів зі штучними шортами"""
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
        short_score = 0

        # Перевірка лімітів позицій
        long_positions = len([p for p in self.positions if p['type'] == 'BUY'])
        short_positions = len([p for p in self.positions if p['type'] == 'SHORT'])

        if len(self.positions) >= self.risk_params['max_open_positions']:
            return 'HOLD', predicted_price, price_change_pct, 0, 'max_positions_reached'

        if short_positions >= self.risk_params['max_short_positions'] and self.enable_shorts:
            pass  # Можна відкривати лонги

        # ML прогноз
        ml_threshold = self.trading_params['base_confidence_threshold']
        if price_change_pct > ml_threshold:
            buy_score += 0.3 * min(2.0, price_change_pct / ml_threshold)
        elif price_change_pct < -ml_threshold:
            sell_score += 0.3 * min(2.0, abs(price_change_pct) / ml_threshold)
            # Шорт скор для негативних прогнозів
            if self.enable_shorts and price_change_pct < -ml_threshold * self.trading_params['short_threshold_multiplier']:
                short_score += 0.4 * min(2.0, abs(price_change_pct) / ml_threshold)

        # RSI індикатори
        if rsi < self.trading_params['rsi_oversold']:
            buy_score += 0.2 * (self.trading_params['rsi_oversold'] - rsi) / self.trading_params['rsi_oversold']
        elif rsi > self.trading_params['rsi_overbought']:
            sell_score += 0.2 * (rsi - self.trading_params['rsi_overbought']) / (
                    100 - self.trading_params['rsi_overbought'])
            # Шорт скор для перекупленості
            if self.enable_shorts and rsi > 75:
                short_score += 0.3 * (rsi - 75) / 25

        # MACD
        if macd > macd_signal:
            buy_score += 0.15
        elif macd < macd_signal:
            sell_score += 0.15
            # Шорт скор для негативного MACD
            if self.enable_shorts and (macd_signal - macd) > 50:
                short_score += 0.2

        # Швидкість ціни і моментум
        if self.market_conditions['price_velocity'] > 0:
            buy_score += 0.2 * min(1.0, abs(self.market_conditions['price_velocity']) / 100)
        elif self.market_conditions['price_velocity'] < 0:
            sell_score += 0.2 * min(1.0, abs(self.market_conditions['price_velocity']) / 100)
            # Шорт скор для швидкого падіння
            if self.enable_shorts and self.market_conditions['price_velocity'] < -50:
                short_score += 0.25

        # Ведмежий ринок індикатор для шортів
        # 🔧 ПОСИЛЕНИЙ ведмежий ринок індикатор для шортів
        if self.enable_shorts and self.market_conditions['bear_market_indicator'] > 0.2:  # Знижений поріг
            short_score += 0.4 * self.market_conditions['bear_market_indicator']

        # 🔧 ДОДАТКОВИЙ шорт скор при спадаючому тренді
        if self.enable_shorts and self.market_conditions['trend_momentum'] < -1.0:  # Спадаючий тренд
            short_score += 0.3 * abs(self.market_conditions['trend_momentum']) / 5.0

        # Логіка SELL сигналів
        if long_positions > 0:
            for position in self.positions:
                if position['type'] == 'BUY':
                    profit_pct = (current_price - position['entry_price']) / position['entry_price'] * 100

                    # Продаж при хорошому прибутку та негативних сигналах
                    if profit_pct > 2.0 and sell_score > 0.2:
                        sell_score += 0.4

                    # Продаж при ослабленні тренду
                    if profit_pct > 1.0 and rsi > 65 and macd < macd_signal:
                        sell_score += 0.3

                    # Продаж при перекупленості
                    if rsi > 75 and profit_pct > 0.5:
                        sell_score += 0.2

        # Об'єм
        if volume_ratio > self.market_conditions['volume_spike_multiplier']:
            volume_boost = min(0.15, (volume_ratio - 1) * 0.3)
            buy_score += volume_boost if buy_score > max(sell_score, short_score) else 0
            sell_score += volume_boost if sell_score > max(buy_score, short_score) else 0
            short_score += volume_boost if short_score > max(buy_score, sell_score) else 0

        # Скальпінг
        if self.trading_params['enable_scalping']:
            scalping_threshold = self.trading_params['scalping_threshold']
            if abs(price_change_pct) > scalping_threshold and abs(price_change_pct) < ml_threshold:
                if price_change_pct > 0 and rsi < 60:
                    buy_score += 0.4
                elif price_change_pct < 0 and rsi > 40:
                    sell_score += 0.4
                    if self.enable_shorts and price_change_pct < -scalping_threshold:
                        short_score += 0.3
        if self.market_conditions['trend_momentum'] < -1.5:  # Сильний спадаючий тренд
            buy_score *= 0.4  # Сильно зменшуємо схильність до покупок
            short_score *= 1.3  # Збільшуємо схильність до шортів
        elif self.market_conditions['trend_momentum'] < -0.5:  # Помірний спадаючий тренд
            buy_score *= 0.7  # Помірно зменшуємо покупки
            short_score *= 1.15  # Помірно збільшуємо шорти
        # Генерація сигналу
        signal = 'HOLD'
        confidence = 0
        signal_type = 'regular'
        min_signal_threshold = 0.4

        # Визначаємо найсильніший сигнал
        max_score = max(buy_score, sell_score, short_score)

        if max_score > min_signal_threshold:
            if buy_score == max_score and buy_score > max(sell_score, short_score) * 1.1:
                signal = 'BUY'
                confidence = min(1.0, buy_score)
                if buy_score > 0.8:
                    signal_type = 'strong'
                elif self.trading_params['enable_scalping'] and abs(price_change_pct) < self.trading_params[
                    'scalping_threshold']:
                    signal_type = 'scalping'

            elif self.enable_shorts and short_score == max_score and short_score > max(buy_score,
                                                                                       sell_score) * 1.1 and short_positions < \
                    self.risk_params['max_short_positions']:
                signal = 'SHORT'
                confidence = min(1.0, short_score)
                if short_score > 0.8:
                    signal_type = 'strong_short'
                else:
                    signal_type = 'short'

            elif sell_score == max_score and sell_score > max(buy_score, short_score) * 1.1:
                signal = 'SELL'
                confidence = min(1.0, sell_score)
                if long_positions > 0:
                    signal_type = 'close_long'
                else:
                    signal_type = 'market_sell'

        return signal, predicted_price, price_change_pct, confidence, signal_type

    def calculate_position_size(self, signal, confidence, signal_type, current_price):
        """📊 ВИПРАВЛЕНЕ ризик-базоване позиціонування"""

        # position_size_base як відсоток ризику від початкового капіталу
        risk_percentage = self.trading_params['position_size_base']  # 0.1 = 10%
        max_risk_per_trade = self.initial_balance * risk_percentage  # $100,000 * 0.1 = $10,000

        #print(f"🎯 Ризик-базоване позиціонування: {risk_percentage:.1%} від початкового = ${max_risk_per_trade:.0f}")

        # Визначаємо стоп-лос відстань
        if signal == 'SHORT':
            stop_loss_distance_pct = self.risk_params['short_stop_loss_pct']  # 1.5%
        else:  # BUY
            stop_loss_distance_pct = self.risk_params['stop_loss_pct']  # 1.5%

        # Розраховуємо ризик на одиницю
        risk_per_unit = current_price * stop_loss_distance_pct

        # Додаємо комісії
        total_trading_cost_pct = (self.trading_costs['taker_fee'] +
                                  self.trading_costs['slippage_pct'] +
                                  self.trading_costs['spread_impact'])
        additional_cost_per_unit = current_price * total_trading_cost_pct

        # Загальний ризик на одиницю
        total_risk_per_unit = risk_per_unit + additional_cost_per_unit

        # 🔧 ГОЛОВНЕ ВИПРАВЛЕННЯ: Розраховуємо максимальну кількість BTC
        max_btc_amount = max_risk_per_trade / total_risk_per_unit

        # 🔧 ДОДАЄМО ПЕРЕВІРКУ НА ДОСТУПНИЙ БАЛАНС
        max_investment_by_balance = self.balance * 0.95  # 95% від доступного балансу
        max_btc_by_balance = max_investment_by_balance / current_price

        # Беремо менше значення
        actual_btc_amount = min(max_btc_amount, max_btc_by_balance)
        actual_position_value = actual_btc_amount * current_price

        # Перераховуємо фактичний ризик
        actual_risk = actual_btc_amount * total_risk_per_unit

        print(
            f"   📊 {signal}: SL {stop_loss_distance_pct * 100:.1f}% = ${risk_per_unit:.2f}/BTC + комісії ${additional_cost_per_unit:.2f}/BTC")
        #print(f"   💰 Ідеальна позиція: {max_btc_amount:.6f} BTC (${max_btc_amount * current_price:.0f})")
        #print(f"   🏦 Обмеження балансом: {max_btc_by_balance:.6f} BTC (${max_investment_by_balance:.0f})")
        print(f"   Фактична позиція: {actual_btc_amount:.6f} BTC (${actual_position_value:.0f})")
        print(f"    ризик: ${actual_risk:.0f}")

        return actual_btc_amount, actual_position_value, actual_risk

    def execute_trade_with_costs(self, signal, current_price, timestamp, predicted_price,
                                 confidence, signal_type):
        """💰 Виконання торгівлі зі штучними шортами"""
        high_price = getattr(self, '_current_high', current_price)
        low_price = getattr(self, '_current_low', current_price)

        # Спочатку перевіряємо стоп-лос/тейк-профіт
        closed_positions = self.check_stop_loss_take_profit(
            current_price, timestamp, high_price, low_price
        )
        if closed_positions > 0:
            print(f"🔒 Закрито {closed_positions} позицій через стоп-лос/тейк-профіт")

        # Початковий розрахунок
        initial_portfolio_value = self.get_portfolio_value(current_price)

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
            # ЛОНГ позиція з виправленим ризик-базованим розрахунком
            btc_amount, position_value, planned_risk = self.calculate_position_size(signal, confidence, signal_type,
                                                                                    current_price)

            # Розраховуємо комісії
            costs = self.calculate_trading_costs(btc_amount, current_price, 'taker')
            effective_price = costs['effective_price_buy']
            total_needed = position_value + costs['total_cost']

            if total_needed <= self.balance:
                final_btc_amount = (position_value - costs['total_cost']) / effective_price

                if final_btc_amount > 0:
                    self.btc_holdings += final_btc_amount
                    self.balance -= total_needed

                    # Розраховуємо стоп-лос/тейк-профіт
                    stop_loss_price = effective_price * (1 - self.risk_params['stop_loss_pct'])
                    take_profit_price = effective_price * (1 + self.risk_params['take_profit_pct'])

                    # Розраховуємо РЕАЛЬНИЙ фактичний ризик
                    actual_risk = (effective_price - stop_loss_price) * final_btc_amount + costs['total_cost']

                    position = {
                        'type': 'BUY',
                        'entry_price': effective_price,
                        'amount': final_btc_amount,
                        'timestamp': timestamp,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'stop_loss_price': stop_loss_price,
                        'take_profit_price': take_profit_price,
                        'highest_price': effective_price,
                        'planned_risk': planned_risk,
                        'actual_risk': actual_risk
                    }
                    self.positions.append(position)

                    # Оновлюємо статистику
                    self.trading_stats['total_fees_paid'] += costs['fee']
                    self.trading_stats['total_slippage_cost'] += costs['slippage']
                    self.trading_stats['total_trades'] += 1
                    self.trading_stats['buy_trades'] += 1

                    trade_info.update({
                        'executed': True,
                        'amount': final_btc_amount,
                        'total_cost': costs['total_cost'],
                        'fee': costs['fee'],
                        'slippage': costs['slippage'],
                        'effective_price': effective_price,
                        'amount_invested': position_value,
                        'stop_loss_price': stop_loss_price,
                        'take_profit_price': take_profit_price,
                        'planned_risk': planned_risk,
                        'actual_risk': actual_risk
                    })

                    executed_this_step = True
                    print(f"🟢 LONG: ${position_value:.0f} → {final_btc_amount:.6f} BTC @ ${effective_price:.0f}")
                    print(f"    SL: ${stop_loss_price:.0f} | 🎯 TP: ${take_profit_price:.0f}")
                    print(f"    Ризик: ${actual_risk:.0f} (план: ${planned_risk:.0f})")
                    print(f"    Balance: ${self.balance:.0f}, BTC: {self.btc_holdings:.6f}")

        elif signal == 'SHORT' and self.enable_shorts and self.balance > self.trading_costs['min_trade_amount']:
            # ШТУЧНИЙ ШОРТ з виправленим розрахунком
            btc_amount, position_value, planned_risk = self.calculate_position_size(signal, confidence, signal_type,
                                                                                    current_price)

            # Для штучного шорту потрібні тільки комісії
            costs = self.calculate_trading_costs(btc_amount, current_price, 'taker')

            if costs['total_cost'] <= self.balance * 0.5:  # Максимум 50% балансу на комісії
                effective_price = costs['effective_price_sell']

                # Віднімаємо комісії
                self.balance -= costs['total_cost']

                # Розраховуємо стоп-лос/тейк-профіт для шорту
                stop_loss_price = effective_price * (1 + self.risk_params['short_stop_loss_pct'])
                take_profit_price = effective_price * (1 - self.risk_params['short_take_profit_pct'])

                # РЕАЛЬНИЙ ризик для шорту
                actual_risk = (stop_loss_price - effective_price) * btc_amount + costs['total_cost']

                position = {
                    'type': 'SHORT',
                    'entry_price': effective_price,
                    'amount': btc_amount,
                    'timestamp': timestamp,
                    'signal_type': signal_type,
                    'confidence': confidence,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'lowest_price': effective_price,
                    'planned_risk': planned_risk,
                    'actual_risk': actual_risk
                }
                self.positions.append(position)

                # Оновлюємо статистику
                self.trading_stats['total_fees_paid'] += costs['fee']
                self.trading_stats['total_slippage_cost'] += costs['slippage']
                self.trading_stats['total_trades'] += 1
                self.trading_stats['short_trades'] += 1

                trade_info.update({
                    'executed': True,
                    'amount': btc_amount,
                    'total_cost': costs['total_cost'],
                    'fee': costs['fee'],
                    'slippage': costs['slippage'],
                    'effective_price': effective_price,
                    'stop_loss_price': stop_loss_price,
                    'take_profit_price': take_profit_price,
                    'planned_risk': planned_risk,
                    'actual_risk': actual_risk
                })

                executed_this_step = True
                print(f"🔴 SHORT: {btc_amount:.6f} BTC @ ${effective_price:.0f}")
                print(f"   🛡️ SL: ${stop_loss_price:.0f} | 🎯 TP: ${take_profit_price:.0f}")
                print(f"   💀 Ризик: ${actual_risk:.0f} (план: ${planned_risk:.0f})")
                print(f"   💰 Balance: ${self.balance:.0f} (штучний шорт)")
            else:
                    trade_info[
                        'reason'] = f"Not enough balance for fees: need ${costs['total_cost']:.0f}, have ${self.balance:.0f}"
        elif signal == 'SELL' and self.btc_holdings > 0:
            # ПРОДАЖ BTC
            position_size = self.calculate_position_size(signal, confidence, signal_type, current_price)
            btc_holdings = float(self.btc_holdings)
            max_btc_to_sell = btc_holdings * position_size
            btc_to_sell = min(btc_holdings, max_btc_to_sell)
            trade_value = btc_to_sell * current_price

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
        final_portfolio_value = self.get_portfolio_value(current_price)

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
        """📊 Статистика торгівлі зі штучними шортами"""
        if not self.trade_history:
            initial_stats = self._get_initial_trading_stats()
            initial_stats.update({
                'avg_confidence': 0,
                'portfolio_volatility': 0,
                'cost_ratio': 0,
                'total_trading_costs': 0,
                'open_positions': 0,
                'stop_loss_rate': 0,
                'take_profit_rate': 0,
                'open_long_positions': 0,
                'open_short_positions': 0,
                'short_success_rate': 0,
                'long_success_rate': 0,
            })
            return initial_stats

        executed_trades_history = [t for t in self.trade_history if t.get('executed', False)]
        portfolio_values = [t['portfolio_value'] for t in self.trade_history if 'portfolio_value' in t]

        # Розділення позицій за типами
        long_positions = len([p for p in self.positions if p['type'] == 'BUY'])
        short_positions = len([p for p in self.positions if p['type'] == 'SHORT'])

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

            # Метрики стоп-лосу/тейк-профіту
            'open_positions': len(self.positions),
            'stop_loss_triggered': self.trading_stats['stop_loss_triggered'],
            'take_profit_triggered': self.trading_stats['take_profit_triggered'],
            'trailing_stop_triggered': self.trading_stats['trailing_stop_triggered'],
            'position_timeouts': self.trading_stats['position_timeouts'],

            # Метрики для штучних шортів
            'short_trades': self.trading_stats['short_trades'],
            'short_cover_trades': self.trading_stats['short_cover_trades'],
            'short_pnl': self.trading_stats['short_pnl'],
            'long_pnl': self.trading_stats['long_pnl'],
            'open_long_positions': long_positions,
            'open_short_positions': short_positions,
        }

        # Розрахунок відсотків спрацьовування
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

        # Розрахунок успішності шортів та лонгів
        if stats['short_pnl'] != 0:
            stats['short_success_rate'] = (stats['short_pnl'] > 0) * 100
        else:
            stats['short_success_rate'] = 0

        if stats['long_pnl'] != 0:
            stats['long_success_rate'] = (stats['long_pnl'] > 0) * 100
        else:
            stats['long_success_rate'] = 0

        # Розрахунок cost_ratio (без borrowing fees)
        total_costs = stats['total_trading_costs']
        current_or_initial_balance = self.balance if self.balance > 0 else self.initial_balance
        if current_or_initial_balance > 0:
            stats['cost_ratio'] = (total_costs / current_or_initial_balance) * 100
        else:
            stats['cost_ratio'] = 0

        return stats

    def show_trading_predictions_summary(self, historical_data, feature_names, n_examples=8):
        """📊 Показ прогнозування зі штучними шортами"""
        print(f"\n📊 ПРИКЛАДИ ПРОГНОЗУВАННЯ (ШТУЧНІ ШОРТИ):")
        print("=" * 80)

        sample_indices = np.linspace(0, len(historical_data) - 1, n_examples, dtype=int)

        print(f"{'Час':<12} {'Поточна':<9} {'Прогноз':<9} {'Зміна%':<7} {'Сигнал':<8} {'Тип':<12} {'Впевненість':<10}")
        print("-" * 80)

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

            # Кольорове кодування для сигналів
            if signal == 'SHORT':
                signal_display = f"🔴{signal}"
            elif signal == 'BUY':
                signal_display = f"🟢{signal}"
            elif signal == 'SELL':
                signal_display = f"🟡{signal}"
            else:
                signal_display = signal

            print(f"{time_str:<12} ${current_price:<8.0f} ${predicted_price:<8.0f} "
                  f"{price_change_pct:+6.2f}% {signal_display:<8} {signal_type:<12} {confidence:<9.2f}")

        print("-" * 80)

        # Інформація про налаштування
        print(f"🔧 НАЛАШТУВАННЯ ШТУЧНИХ ШОРТІВ:")
        print(f"   Поріг впевненості: {self.trading_params['base_confidence_threshold']}")
        print(f"   Max позицій: {self.risk_params['max_open_positions']}")
        print(f"   Timeout: {self.risk_params['position_timeout_hours']} годин")
        if self.enable_shorts:
            print(f"🔴 Штучні шорт позиції: УВІМКНЕНІ")
            print(f"   Max шорт позицій: {self.risk_params['max_short_positions']}")
            print(f"   Шорт поріг: {self.trading_params['short_threshold_multiplier']}")
        else:
            print("🔴 Шорт позиції ВИМКНЕНІ")
        print("-" * 80)