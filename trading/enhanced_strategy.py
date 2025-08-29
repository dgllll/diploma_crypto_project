import pandas as pd
import numpy as np


class EnhancedCryptoTradingStrategy:
    """
    Покращена торгова стратегія з адаптивними параметрами та trend-following логікою
    """

    def __init__(self, prediction_model, initial_balance=10000):
        self.model = prediction_model
        self.balance = initial_balance
        self.btc_holdings = 0
        self.positions = []
        self.trade_history = []

        # 🆕 ПОКРАЩЕННЯ 1: Адаптивні параметри
        self.adaptive_params = {
            'base_confidence_threshold': 0.8,  # Знижено з 1.5%
            'volatility_adjustment': True,
            'trend_adjustment': True,
            'position_size_base': 0.15,  # Знижено з 0.2
            'max_position_size': 0.3,
            'min_position_size': 0.05
        }

        # 🆕 ПОКРАЩЕННЯ 2: Trend tracking
        self.trend_state = {
            'current_trend': 'NEUTRAL',  # BULLISH, BEARISH, NEUTRAL
            'trend_strength': 0,
            'trend_duration': 0,
            'recent_prices': []
        }

        # 🆕 ПОКРАЩЕННЯ 3: Risk management
        self.risk_params = {
            'max_drawdown_limit': 0.15,  # 15%
            'consecutive_losses_limit': 3,
            'stop_loss_pct': 0.05,  # 5%
            'take_profit_pct': 0.12,  # 12%
            'trailing_stop_pct': 0.03  # 3%
        }

        self.risk_state = {
            'consecutive_losses': 0,
            'peak_portfolio_value': initial_balance,
            'current_drawdown': 0
        }

    def calculate_adaptive_threshold(self, volatility, trend_strength):
        """
        🆕 ПОКРАЩЕННЯ 4: Динамічний розрахунок порогу впевненості
        """
        base_threshold = self.adaptive_params['base_confidence_threshold']

        # Адаптація до волатільності
        if self.adaptive_params['volatility_adjustment']:
            if volatility > 0.04:  # Висока волатільність
                threshold = base_threshold * 0.7  # Зменшуємо поріг
            elif volatility < 0.02:  # Низька волатільність
                threshold = base_threshold * 1.3  # Збільшуємо поріг
            else:
                threshold = base_threshold
        else:
            threshold = base_threshold

        # Адаптація до тренду
        if self.adaptive_params['trend_adjustment']:
            if abs(trend_strength) > 0.6:  # Сильний тренд
                threshold *= 0.8  # Легше відкривати позиції
            elif abs(trend_strength) < 0.3:  # Слабкий тренд
                threshold *= 1.2  # Складніше відкривати позиції

        return max(0.3, min(2.0, threshold))  # Обмежуємо діапазон

    def update_trend_state(self, current_price, technical_indicators):
        """
        🆕 ПОКРАЩЕННЯ 5: Відстеження стану тренду
        """
        # Додаємо поточну ціну до історії
        self.trend_state['recent_prices'].append(current_price)
        if len(self.trend_state['recent_prices']) > 20:
            self.trend_state['recent_prices'].pop(0)

        if len(self.trend_state['recent_prices']) < 10:
            return

        # Розрахунок тренду на основі EMA
        prices = np.array(self.trend_state['recent_prices'])
        short_ema = np.mean(prices[-5:])
        long_ema = np.mean(prices[-15:])

        # Визначення тренду
        trend_ratio = (short_ema - long_ema) / long_ema

        if trend_ratio > 0.01:  # 1% різниця
            new_trend = 'BULLISH'
            self.trend_state['trend_strength'] = min(1.0, trend_ratio * 50)
        elif trend_ratio < -0.01:
            new_trend = 'BEARISH'
            self.trend_state['trend_strength'] = max(-1.0, trend_ratio * 50)
        else:
            new_trend = 'NEUTRAL'
            self.trend_state['trend_strength'] = 0

        # Оновлення тривалості тренду
        if new_trend == self.trend_state['current_trend']:
            self.trend_state['trend_duration'] += 1
        else:
            self.trend_state['trend_duration'] = 1
            self.trend_state['current_trend'] = new_trend

    def calculate_position_size(self, signal, confidence, volatility, current_price):
        """
        🆕 ПОКРАЩЕННЯ 6: Адаптивний розмір позиції
        """
        base_size = self.adaptive_params['position_size_base']

        # Адаптація до впевненості
        confidence_multiplier = min(2.0, confidence / 0.5)

        # Адаптація до волатільності (менше при високій волатільності)
        volatility_multiplier = 1.0 / (1.0 + volatility * 10)

        # Адаптація до тренду
        trend_multiplier = 1.0
        if signal == 'BUY' and self.trend_state['current_trend'] == 'BULLISH':
            trend_multiplier = 1.3
        elif signal == 'SELL' and self.trend_state['current_trend'] == 'BEARISH':
            trend_multiplier = 1.3
        elif signal == 'BUY' and self.trend_state['current_trend'] == 'BEARISH':
            trend_multiplier = 0.6  # Обережніше проти тренду
        elif signal == 'SELL' and self.trend_state['current_trend'] == 'BULLISH':
            trend_multiplier = 0.6

        # Адаптація до drawdown
        drawdown_multiplier = 1.0
        if self.risk_state['current_drawdown'] > 0.05:  # Якщо drawdown > 5%
            drawdown_multiplier = 0.7

        # Фінальний розрахунок
        position_size = (base_size * confidence_multiplier *
                         volatility_multiplier * trend_multiplier *
                         drawdown_multiplier)

        # Обмеження
        position_size = max(self.adaptive_params['min_position_size'],
                            min(self.adaptive_params['max_position_size'], position_size))

        return position_size

    def generate_enhanced_signal(self, features, current_price, technical_indicators):
        """
        🆕 ПОКРАЩЕННЯ 7: Покращена логіка генерації сигналів
        """
        # Базовий прогноз від моделі
        predicted_price = self.model.predict(features)[0]
        price_change_pct = (predicted_price - current_price) / current_price * 100

        # Оновлення стану тренду
        self.update_trend_state(current_price, technical_indicators)

        # Розрахунок волатільності
        volatility = technical_indicators.get('volatility', 0.02)

        # Адаптивний поріг
        adaptive_threshold = self.calculate_adaptive_threshold(volatility,
                                                               self.trend_state['trend_strength'])

        # Отримання технічних індикаторів
        rsi = technical_indicators.get('rsi', 50)
        macd = technical_indicators.get('macd', 0)
        macd_signal = technical_indicators.get('macd_signal', 0)
        adx = technical_indicators.get('adx', 20)

        # 🆕 ПОКРАЩЕННЯ 8: Multi-factor scoring система
        buy_score = 0
        sell_score = 0

        # ML прогноз (вага: 40%)
        if price_change_pct > adaptive_threshold:
            buy_score += 0.4 * (price_change_pct / adaptive_threshold)
        elif price_change_pct < -adaptive_threshold:
            sell_score += 0.4 * (abs(price_change_pct) / adaptive_threshold)

        # Технічні індикатори (вага: 35%)
        # RSI
        if rsi < 35:
            buy_score += 0.15 * (35 - rsi) / 35
        elif rsi > 65:
            sell_score += 0.15 * (rsi - 65) / 35

        # MACD
        macd_signal_strength = abs(macd - macd_signal)
        if macd > macd_signal and macd_signal_strength > 50:
            buy_score += 0.2
        elif macd < macd_signal and macd_signal_strength > 50:
            sell_score += 0.2

        # Тренд (вага: 25%)
        if self.trend_state['current_trend'] == 'BULLISH':
            buy_score += 0.25 * self.trend_state['trend_strength']
        elif self.trend_state['current_trend'] == 'BEARISH':
            sell_score += 0.25 * abs(self.trend_state['trend_strength'])

        # 🆕 ПОКРАЩЕННЯ 9: Trend-following фільтри
        # Не продавати під час сильного зростання
        if sell_score > 0.6 and price_change_pct > 2.0:
            sell_score *= 0.3

        # Не купувати під час сильного падіння
        if buy_score > 0.6 and price_change_pct < -2.0:
            buy_score *= 0.3

        # Підсилення сигналів у напрямку тренду
        if (buy_score > 0.4 and self.trend_state['current_trend'] == 'BULLISH' and
                self.trend_state['trend_duration'] > 3):
            buy_score *= 1.3

        if (sell_score > 0.4 and self.trend_state['current_trend'] == 'BEARISH' and
                self.trend_state['trend_duration'] > 3):
            sell_score *= 1.3

        # Генерація сигналу
        signal = 'HOLD'
        confidence = 0

        if buy_score > 0.6 and buy_score > sell_score:
            signal = 'BUY'
            confidence = min(1.0, buy_score)
        elif sell_score > 0.6 and sell_score > buy_score:
            signal = 'SELL'
            confidence = min(1.0, sell_score)

        return signal, predicted_price, price_change_pct, confidence

    def check_risk_management(self, current_price):
        """
        🆕 ПОКРАЩЕННЯ 10: Розширений ризик-менеджмент
        """
        portfolio_value = self.balance + (self.btc_holdings * current_price)

        # Оновлення peak value і drawdown
        if portfolio_value > self.risk_state['peak_portfolio_value']:
            self.risk_state['peak_portfolio_value'] = portfolio_value

        self.risk_state['current_drawdown'] = (
                (self.risk_state['peak_portfolio_value'] - portfolio_value) /
                self.risk_state['peak_portfolio_value']
        )

        # Перевірка лімітів
        risk_override = False

        # Максимальна просадка
        if self.risk_state['current_drawdown'] > self.risk_params['max_drawdown_limit']:
            risk_override = True
            print(f"⚠️ Досягнуто ліміт drawdown: {self.risk_state['current_drawdown']:.2%}")

        # Послідовні збитки
        if self.risk_state['consecutive_losses'] >= self.risk_params['consecutive_losses_limit']:
            risk_override = True
            print(f"⚠️ Занадто багато послідовних збитків: {self.risk_state['consecutive_losses']}")

        return risk_override

    def execute_enhanced_trade(self, signal, current_price, timestamp, predicted_price, confidence):
        """
        🆕 ПОКРАЩЕННЯ 11: Покращене виконання угод з ризик-менеджментом
        """
        # Перевірка ризик-менеджменту
        if self.check_risk_management(current_price):
            signal = 'HOLD'  # Блокуємо торгівлю при високому ризику

        trade_info = {
            'timestamp': timestamp,
            'signal': signal,
            'price': current_price,
            'predicted_price': predicted_price,
            'confidence': confidence,
            'balance_before': self.balance,
            'btc_before': self.btc_holdings
        }

        if signal == 'BUY' and self.balance > 100:  # Мінімум $100 для торгівлі
            # Розрахунок адаптивного розміру позиції
            volatility = 0.02  # Заглушка, в реальності з technical_indicators
            position_size = self.calculate_position_size('BUY', confidence, volatility, current_price)

            # Розрахунок кількості BTC для покупки
            amount_to_invest = self.balance * position_size
            btc_to_buy = amount_to_invest / current_price

            self.btc_holdings += btc_to_buy
            self.balance -= amount_to_invest

            # Додаємо позицію з stop-loss та take-profit
            self.positions.append({
                'type': 'BUY',
                'price': current_price,
                'amount': btc_to_buy,
                'timestamp': timestamp,
                'stop_loss': current_price * (1 - self.risk_params['stop_loss_pct']),
                'take_profit': current_price * (1 + self.risk_params['take_profit_pct']),
                'confidence': confidence
            })

        elif signal == 'SELL' and self.btc_holdings > 0.001:  # Мінімум 0.001 BTC
            # Розрахунок кількості для продажу
            volatility = 0.02
            position_size = self.calculate_position_size('SELL', confidence, volatility, current_price)
            btc_to_sell = min(self.btc_holdings, self.btc_holdings * position_size)

            self.balance += btc_to_sell * current_price
            self.btc_holdings -= btc_to_sell

            self.positions.append({
                'type': 'SELL',
                'price': current_price,
                'amount': btc_to_sell,
                'timestamp': timestamp,
                'confidence': confidence
            })

        trade_info['balance_after'] = self.balance
        trade_info['btc_after'] = self.btc_holdings
        trade_info['portfolio_value'] = self.balance + (self.btc_holdings * current_price)

        # Оновлення статистики ризику
        if len(self.trade_history) > 0:
            prev_portfolio = self.trade_history[-1]['portfolio_value']
            current_portfolio = trade_info['portfolio_value']

            if current_portfolio < prev_portfolio:
                self.risk_state['consecutive_losses'] += 1
            else:
                self.risk_state['consecutive_losses'] = 0

        self.trade_history.append(trade_info)
        return trade_info

    def get_strategy_stats(self):
        """
        🆕 ПОКРАЩЕННЯ 12: Детальна статистика стратегії
        """
        if not self.trade_history:
            return {}

        portfolio_values = [trade['portfolio_value'] for trade in self.trade_history]

        return {
            'trend_state': self.trend_state['current_trend'],
            'trend_strength': self.trend_state['trend_strength'],
            'current_drawdown': self.risk_state['current_drawdown'],
            'consecutive_losses': self.risk_state['consecutive_losses'],
            'total_trades': len([t for t in self.trade_history if t['signal'] != 'HOLD']),
            'avg_confidence': np.mean([t.get('confidence', 0) for t in self.trade_history]),
            'portfolio_volatility': np.std(np.diff(portfolio_values)) if len(portfolio_values) > 1 else 0
        }