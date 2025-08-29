import pandas as pd
import numpy as np


class AggressiveCryptoTradingStrategy:
    """
    Агресивна торгова стратегія з більшою кількістю трейдів, комісіями та проскальзуванням
    """

    def __init__(self, prediction_model, initial_balance=100000):
        self.model = prediction_model
        self.balance = initial_balance
        self.btc_holdings = 0
        self.positions = []
        self.trade_history = []

        # 🔥 АГРЕСИВНІ ПАРАМЕТРИ для більшої кількості трейдів
        self.trading_params = {
            'base_confidence_threshold': 0.4,  # Знижено з 0.8 до 0.3%
            'min_confidence_threshold': 0.2,  # Мінімальний поріг
            'position_size_base': 0.25,  # Збільшено з 0.15 до 0.25
            'max_position_size': 0.4,  # Збільшено максимум
            'min_position_size': 0.08,
            'rsi_oversold': 42,  # Розширено з 35 до 45
            'rsi_overbought': 58,  # Звужено з 65 до 55
            'enable_scalping': True,  # Скальпінг для коротких трейдів
            'scalping_threshold': 0.3,  # Дуже низький поріг для скальпінгу
        }

        # 💰 КОМІСІЇ ТА ПРОСКАЛЬЗУВАННЯ
        self.trading_costs = {
            'maker_fee': 0.0004,  # 0.1% комісія мейкера
            'taker_fee': 0.0006,  # 0.15% комісія тейкера
            'slippage_pct': 0.0005,  # 0.05% проскальзування
            'min_trade_amount': 300,  # Мінімальна сума трейду
            'spread_impact': 0.0002,  # 0.02% вплив спреду
        }

        # 📊 СТАТИСТИКА ТОРГІВЛІ
        self.trading_stats = {
            'total_fees_paid': 0,
            'total_slippage_cost': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'scalping_trades': 0,
        }

        # 🎯 ДОДАТКОВІ ІНДИКАТОРИ для агресивної стратегії
        self.market_conditions = {
            'volatility_threshold': 0.015,  # 1.5% волатільність для активної торгівлі
            'volume_spike_multiplier': 1.2,  # Сплеск об'єму
            'trend_momentum': 0,
            'recent_prices': [],
            'price_velocity': 0,
        }

    def calculate_trading_costs(self, amount, price, trade_type='taker'):
        """
        💰 Розрахунок реальних торгових витрат
        """
        trade_value = amount * price

        # Комісія (залежить від типу ордера)
        if trade_type == 'maker':
            fee = trade_value * self.trading_costs['maker_fee']
        else:
            fee = trade_value * self.trading_costs['taker_fee']

        # Проскальзування (завжди працює проти трейдера)
        slippage = trade_value * self.trading_costs['slippage_pct']

        # Спред (половина спреду)
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
                        1 - self.trading_costs['slipgage_pct'] - self.trading_costs['spread_impact'])
        }

    def update_market_conditions(self, current_price, volume=None, technical_indicators=None):
        """
        🎯 Оновлення ринкових умов для агресивної торгівлі
        """
        # Додаємо поточну ціну до історії
        self.market_conditions['recent_prices'].append(current_price)
        if len(self.market_conditions['recent_prices']) > 20:
            self.market_conditions['recent_prices'].pop(0)

        if len(self.market_conditions['recent_prices']) >= 5:
            prices = np.array(self.market_conditions['recent_prices'])

            # Швидкість зміни ціни (momentum)
            price_changes = np.diff(prices)
            self.market_conditions['price_velocity'] = np.mean(price_changes[-3:])

            # Trend momentum
            if len(prices) >= 10:
                short_avg = np.mean(prices[-5:])
                long_avg = np.mean(prices[-10:])
                self.market_conditions['trend_momentum'] = (short_avg - long_avg) / long_avg * 100

    def generate_aggressive_signal(self, features, current_price, technical_indicators):
        """
        🔥 Агресивна генерація сигналів для більшої кількості трейдів
        """
        # Оновлюємо ринкові умови
        self.update_market_conditions(current_price, technical_indicators=technical_indicators)

        # Базовий прогноз від моделі
        try:
            predicted_price = self.model.predict(features)[0]
            price_change_pct = (predicted_price - current_price) / current_price * 100
        except:
            predicted_price = current_price
            price_change_pct = 0

        # Технічні індикатори
        rsi = technical_indicators.get('rsi', 50)
        macd = technical_indicators.get('macd', 0)
        macd_signal = technical_indicators.get('macd_signal', 0)
        adx = technical_indicators.get('adx', 20)
        volume_ratio = technical_indicators.get('volume_ratio_20', 1.0)

        # 🎯 БАГАТОФАКТОРНА СИСТЕМА СКОРИНГУ
        buy_score = 0
        sell_score = 0

        # 1️⃣ ML прогноз (вага: 30% - знижено для більшої агресивності)
        ml_threshold = self.trading_params['base_confidence_threshold']
        if price_change_pct > ml_threshold:
            buy_score += 0.3 * min(2.0, price_change_pct / ml_threshold)
        elif price_change_pct < -ml_threshold:
            sell_score += 0.3 * min(2.0, abs(price_change_pct) / ml_threshold)

        # 2️⃣ Технічні індикатори (вага: 35%)
        # RSI (більш агресивні пороги)
        if rsi < self.trading_params['rsi_oversold']:
            buy_score += 0.2 * (self.trading_params['rsi_oversold'] - rsi) / self.trading_params['rsi_oversold']
        elif rsi > self.trading_params['rsi_overbought']:
            sell_score += 0.2 * (rsi - self.trading_params['rsi_overbought']) / (
                        100 - self.trading_params['rsi_overbought'])

        # MACD
        if macd > macd_signal:
            buy_score += 0.15
        elif macd < macd_signal:
            sell_score += 0.15

        # 3️⃣ Momentum та velocity (вага: 20%)
        if self.market_conditions['price_velocity'] > 0:
            buy_score += 0.2 * min(1.0, abs(self.market_conditions['price_velocity']) / 100)
        elif self.market_conditions['price_velocity'] < 0:
            sell_score += 0.2 * min(1.0, abs(self.market_conditions['price_velocity']) / 100)

        # 4️⃣ Об'єм (вага: 15%)
        if volume_ratio > self.market_conditions['volume_spike_multiplier']:
            # Високий об'єм підсилює сигнали
            volume_boost = min(0.15, (volume_ratio - 1) * 0.3)
            buy_score += volume_boost if buy_score > sell_score else 0
            sell_score += volume_boost if sell_score > buy_score else 0

        # 🔥 СКАЛЬПІНГ режим для дуже коротких трейдів
        if self.trading_params['enable_scalping']:
            scalping_threshold = self.trading_params['scalping_threshold']

            # Скальпінг на малих рухах з високою впевненістю
            if abs(price_change_pct) > scalping_threshold and abs(price_change_pct) < ml_threshold:
                if price_change_pct > 0 and rsi < 60:
                    buy_score += 0.4  # Бонус для скальпінгу
                elif price_change_pct < 0 and rsi > 40:
                    sell_score += 0.4

        # 📈 ГЕНЕРАЦІЯ СИГНАЛУ з нижчими порогами
        signal = 'HOLD'
        confidence = 0
        signal_type = 'regular'

        # Знижені пороги для більшої кількості трейдів
        min_signal_threshold = 0.35  # Знижено з 0.6

        if buy_score > min_signal_threshold and buy_score > sell_score * 1.1:
            signal = 'BUY'
            confidence = min(1.0, buy_score)
            if buy_score > 0.8:
                signal_type = 'strong'
            elif abs(price_change_pct) < self.trading_params['scalping_threshold']:
                signal_type = 'scalping'

        elif sell_score > min_signal_threshold and sell_score > buy_score * 1.1:
            signal = 'SELL'
            confidence = min(1.0, sell_score)
            if sell_score > 0.8:
                signal_type = 'strong'
            elif abs(price_change_pct) < self.trading_params['scalping_threshold']:
                signal_type = 'scalping'

        return signal, predicted_price, price_change_pct, confidence, signal_type

    def calculate_position_size(self, signal, confidence, signal_type, current_price):
        """
        📊 Розрахунок розміру позиції з урахуванням типу сигналу
        """
        base_size = self.trading_params['position_size_base']

        # Адаптація до впевненості
        confidence_multiplier = 0.5 + (confidence * 1.5)  # 0.5-2.0

        # Адаптація до типу сигналу
        if signal_type == 'strong':
            type_multiplier = 1.3
        elif signal_type == 'scalping':
            type_multiplier = 0.6  # Менші позиції для скальпінгу
        else:
            type_multiplier = 1.0

        # Адаптація до поточного балансу портфеля
        portfolio_value = self.balance + (self.btc_holdings * current_price)
        if portfolio_value < self.balance * 0.8:  # Якщо втратили >20%
            balance_multiplier = 0.7
        else:
            balance_multiplier = 1.0

        # Фінальний розрахунок
        position_size = base_size * confidence_multiplier * type_multiplier * balance_multiplier

        # Обмеження
        position_size = max(self.trading_params['min_position_size'],
                            min(self.trading_params['max_position_size'], position_size))

        return position_size

    def execute_trade_with_costs(self, signal, current_price, timestamp, predicted_price,
                                 confidence, signal_type):
        """
        💰 Виконання торгівлі з урахуванням комісій та проскальзування
        """
        trade_info = {
            'timestamp': timestamp,
            'signal': signal,
            'signal_type': signal_type,
            'price': current_price,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'balance_before': self.balance,
            'btc_before': self.btc_holdings,
            'total_cost': 0,
            'fee': 0,
            'slippage': 0,
            'effective_price': current_price
        }

        if (signal == 'BUY' and
                self.balance > self.trading_costs['min_trade_amount']):

            # Розрахунок розміру позиції
            position_size = self.calculate_position_size(signal, confidence, signal_type, current_price)
            amount_to_invest = self.balance * position_size

            # Перевірка мінімальної суми
            if amount_to_invest < self.trading_costs['min_trade_amount']:
                trade_info['reason'] = 'Amount too small'
                return trade_info

            # Розрахунок торгових витрат
            costs = self.calculate_trading_costs(
                amount_to_invest / current_price, current_price, 'taker'
            )

            # Ефективна ціна покупки (з урахуванням проскальзування)
            effective_price = costs['effective_price_buy']
            btc_to_buy = (amount_to_invest - costs['total_cost']) / effective_price

            # Перевірка, чи достатньо коштів
            total_needed = amount_to_invest + costs['total_cost']
            if total_needed <= self.balance:
                self.btc_holdings += btc_to_buy
                self.balance -= total_needed

                # Оновлення статистики
                self.trading_stats['total_fees_paid'] += costs['fee']
                self.trading_stats['total_slippage_cost'] += costs['slippage']
                if signal_type == 'scalping':
                    self.trading_stats['scalping_trades'] += 1

                # Інформація про трейд
                trade_info.update({
                    'executed': True,
                    'amount': btc_to_buy,
                    'total_cost': costs['total_cost'],
                    'fee': costs['fee'],
                    'slippage': costs['slippage'],
                    'effective_price': effective_price
                })

                self.positions.append({
                    'type': 'BUY',
                    'price': effective_price,
                    'amount': btc_to_buy,
                    'timestamp': timestamp,
                    'signal_type': signal_type,
                    'confidence': confidence
                })

        elif (signal == 'SELL' and
              self.btc_holdings > 0):

            # Розрахунок кількості для продажу
            position_size = self.calculate_position_size(signal, confidence, signal_type, current_price)
            btc_to_sell = min(self.btc_holdings, self.btc_holdings * position_size)

            # Перевірка мінімальної суми
            if btc_to_sell * current_price < self.trading_costs['min_trade_amount']:
                trade_info['reason'] = 'Amount too small'
                return trade_info

            # Розрахунок торгових витрат
            costs = self.calculate_trading_costs(btc_to_sell, current_price, 'taker')

            # Ефективна ціна продажу (з урахуванням проскальзування)
            effective_price = costs['effective_price_sell']
            proceeds = (btc_to_sell * effective_price) - costs['total_cost']

            self.balance += proceeds
            self.btc_holdings -= btc_to_sell

            # Оновлення статистики
            self.trading_stats['total_fees_paid'] += costs['fee']
            self.trading_stats['total_slippage_cost'] += costs['slippage']
            if signal_type == 'scalping':
                self.trading_stats['scalping_trades'] += 1

            # Інформація про трейд
            trade_info.update({
                'executed': True,
                'amount': btc_to_sell,
                'proceeds': proceeds,
                'total_cost': costs['total_cost'],
                'fee': costs['fee'],
                'slippage': costs['slippage'],
                'effective_price': effective_price
            })

            self.positions.append({
                'type': 'SELL',
                'price': effective_price,
                'amount': btc_to_sell,
                'timestamp': timestamp,
                'signal_type': signal_type,
                'confidence': confidence
            })

        # Фінальні розрахунки
        trade_info['balance_after'] = self.balance
        trade_info['btc_after'] = self.btc_holdings
        trade_info['portfolio_value'] = self.balance + (self.btc_holdings * current_price)

        self.trade_history.append(trade_info)
        return trade_info

    def get_trading_statistics(self):
        """
        📊 Детальна статистика торгівлі
        """
        if not self.trade_history:
            return {}

        executed_trades = [t for t in self.trade_history if t.get('executed', False)]
        buy_trades = [t for t in executed_trades if t['signal'] == 'BUY']
        sell_trades = [t for t in executed_trades if t['signal'] == 'SELL']
        scalping_trades = [t for t in executed_trades if t.get('signal_type') == 'scalping']

        portfolio_values = [t['portfolio_value'] for t in self.trade_history]

        return {
            'total_trades': len(executed_trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'scalping_trades': len(scalping_trades),
            'total_fees_paid': self.trading_stats['total_fees_paid'],
            'total_slippage_cost': self.trading_stats['total_slippage_cost'],
            'total_trading_costs': self.trading_stats['total_fees_paid'] + self.trading_stats['total_slippage_cost'],
            'avg_confidence': np.mean([t.get('confidence', 0) for t in executed_trades]) if executed_trades else 0,
            'portfolio_volatility': np.std(np.diff(portfolio_values)) if len(portfolio_values) > 1 else 0,
            'cost_ratio': (self.trading_stats['total_fees_paid'] + self.trading_stats[
                'total_slippage_cost']) / self.balance * 100
        }