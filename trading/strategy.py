import pandas as pd
import numpy as np


class CryptoTradingStrategy:
    """
    Торгова стратегія на основі прогнозів ансамблевої моделі та технічних індикаторів
    """

    def __init__(self, prediction_model, confidence_threshold=0.02, position_size=0.1):
        """
        Ініціалізує торгову стратегію

        Parameters:
        -----------
        prediction_model : CryptoPricePredictionPipeline
            Навчена модель прогнозування
        confidence_threshold : float
            Поріг впевненості для здійснення угоди (% зміни)
        position_size : float
            Розмір позиції (частка капіталу)
        """
        self.model = prediction_model
        self.confidence_threshold = confidence_threshold
        self.position_size = position_size
        self.positions = []
        self.balance = 10000  # Початковий капітал
        self.btc_holdings = 0
        self.trade_history = []

    def generate_signal(self, features, current_price, technical_indicators):
        """
        Генерує торговий сигнал на основі прогнозу моделі та технічних індикаторів
        """
        # Прогноз ціни від моделі
        predicted_price = self.model.predict(features)[0]

        # Розрахунок прогнозованої зміни ціни
        price_change_pct = (predicted_price - current_price) / current_price * 100

        # Отримання технічних індикаторів
        rsi = technical_indicators.get('rsi', 50)
        macd = technical_indicators.get('macd', 0)
        macd_signal = technical_indicators.get('macd_signal', 0)
        adx = technical_indicators.get('adx', 20)

        # Гібридна логіка сигналу
        signal = 'HOLD'

        # Сигнал на покупку
        if (price_change_pct > self.confidence_threshold and  # Модель прогнозує зростання
                rsi < 70 and  # RSI не в зоні перекупленості
                macd > macd_signal and  # Позитивний MACD кросовер
                adx > 25):  # Сильний тренд
            signal = 'BUY'

        # Сигнал на продаж
        elif (price_change_pct < -self.confidence_threshold and  # Модель прогнозує падіння
              rsi > 30 and  # RSI не в зоні перепроданості
              macd < macd_signal and  # Негативний MACD кросовер
              adx > 25):  # Сильний тренд
            signal = 'SELL'

        return signal, predicted_price, price_change_pct

    def execute_trade(self, signal, current_price, timestamp, predicted_price):
        """
        Виконує торгівлю на основі сигналу
        """
        trade_info = {
            'timestamp': timestamp,
            'signal': signal,
            'price': current_price,
            'predicted_price': predicted_price,
            'balance_before': self.balance,
            'btc_before': self.btc_holdings
        }

        if signal == 'BUY' and self.balance > 0:
            # Розрахунок кількості BTC для покупки
            btc_to_buy = (self.balance * self.position_size) / current_price
            self.btc_holdings += btc_to_buy
            self.balance -= btc_to_buy * current_price

            # Додаємо позицію
            self.positions.append({
                'type': 'BUY',
                'price': current_price,
                'amount': btc_to_buy,
                'timestamp': timestamp
            })

        elif signal == 'SELL' and self.btc_holdings > 0:
            # Продаємо частину або всі BTC
            btc_to_sell = self.btc_holdings * self.position_size
            self.balance += btc_to_sell * current_price
            self.btc_holdings -= btc_to_sell

            # Додаємо позицію
            self.positions.append({
                'type': 'SELL',
                'price': current_price,
                'amount': btc_to_sell,
                'timestamp': timestamp
            })

        trade_info['balance_after'] = self.balance
        trade_info['btc_after'] = self.btc_holdings
        trade_info['portfolio_value'] = self.balance + (self.btc_holdings * current_price)

        # Додаємо інформацію про торгівлю до історії
        self.trade_history.append(trade_info)

        return trade_info