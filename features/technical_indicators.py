import pandas as pd
import numpy as np
from ta import trend, momentum, volatility, volume


def add_trend_indicators(df):
    """Додає індикатори тренду до датафрейму"""
    # Копіюємо датафрейм
    data = df.copy()

    # Індикатори тренду
    # ADX (Average Directional Index)
    try:
        adx = trend.ADXIndicator(data['high'], data['low'], data['close'], window=14)
        data['adx'] = adx.adx()
        data['adx_pos'] = adx.adx_pos()
        data['adx_neg'] = adx.adx_neg()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати ADX: {e}")
        data['adx'] = np.nan
        data['adx_pos'] = np.nan
        data['adx_neg'] = np.nan

    # MACD (Moving Average Convergence Divergence)
    for fast, slow in [(12, 26), (8, 21), (5, 35)]:
        try:
            macd = trend.MACD(data['close'], window_fast=fast, window_slow=slow, window_sign=9)
            data[f'macd_{fast}_{slow}'] = macd.macd()
            data[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
            data[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()
        except Exception as e:
            print(f"Попередження: Не вдалося розрахувати MACD ({fast}, {slow}): {e}")
            data[f'macd_{fast}_{slow}'] = np.nan
            data[f'macd_signal_{fast}_{slow}'] = np.nan
            data[f'macd_diff_{fast}_{slow}'] = np.nan

    # SMA (Simple Moving Average) та EMA (Exponential Moving Average)
    for window in [10, 12, 20, 26, 50, 100, 200]:  # Додали 12 та 26 для кросоверів
        try:
            data[f'sma_{window}'] = trend.SMAIndicator(data['close'], window=window).sma_indicator()
            data[f'ema_{window}'] = trend.EMAIndicator(data['close'], window=window).ema_indicator()
        except Exception as e:
            print(f"Попередження: Не вдалося розрахувати SMA/EMA ({window}): {e}")
            data[f'sma_{window}'] = np.nan
            data[f'ema_{window}'] = np.nan

    # SMA і EMA кросовери - тепер після створення всіх потрібних індикаторів
    try:
        data['sma_20_50_cross'] = np.where(data['sma_20'] > data['sma_50'], 1, -1)
        data['ema_12_26_cross'] = np.where(data['ema_12'] > data['ema_26'], 1, -1)
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати кросовери: {e}")
        data['sma_20_50_cross'] = np.nan
        data['ema_12_26_cross'] = np.nan

    # Ichimoku Cloud
    try:
        ichimoku = trend.IchimokuIndicator(data['high'], data['low'])
        data['ichimoku_a'] = ichimoku.ichimoku_a()
        data['ichimoku_b'] = ichimoku.ichimoku_b()
        data['ichimoku_base'] = ichimoku.ichimoku_base_line()
        data['ichimoku_conv'] = ichimoku.ichimoku_conversion_line()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати Ichimoku: {e}")
        data['ichimoku_a'] = np.nan
        data['ichimoku_b'] = np.nan
        data['ichimoku_base'] = np.nan
        data['ichimoku_conv'] = np.nan

    return data


def add_momentum_indicators(df):
    """Додає індикатори моментуму до датафрейму"""
    data = df.copy()

    # RSI (Relative Strength Index)
    try:
        data['rsi'] = momentum.RSIIndicator(data['close'], window=14).rsi()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати RSI: {e}")
        data['rsi'] = np.nan

    # Стохастичний осцилятор
    try:
        stoch = momentum.StochasticOscillator(data['high'], data['low'], data['close'])
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати Stochastic: {e}")
        data['stoch_k'] = np.nan
        data['stoch_d'] = np.nan

    # CCI (Commodity Channel Index)
    try:
        data['cci'] = trend.CCIIndicator(data['high'], data['low'], data['close']).cci()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати CCI: {e}")
        data['cci'] = np.nan

    # TSI (True Strength Index)
    try:
        data['tsi'] = momentum.TSIIndicator(data['close']).tsi()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати TSI: {e}")
        data['tsi'] = np.nan

    # Williams %R
    try:
        data['williams_r'] = momentum.WilliamsRIndicator(data['high'], data['low'], data['close']).williams_r()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати Williams %R: {e}")
        data['williams_r'] = np.nan

    # MFI (Money Flow Index) з обсягом
    try:
        data['mfi'] = volume.MFIIndicator(data['high'], data['low'], data['close'],
                                             data['volume']).money_flow_index()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати MFI: {e}")
        data['mfi'] = np.nan

    return data


def add_volatility_indicators(df):
    """Додає індикатори волатільності до датафрейму"""
    data = df.copy()

    # Bollinger Bands
    for window in [10, 20, 50]:
        try:
            bollinger = volatility.BollingerBands(data['close'], window=window, window_dev=2)
            data[f'bb_upper_{window}'] = bollinger.bollinger_hband()
            data[f'bb_lower_{window}'] = bollinger.bollinger_lband()
            data[f'bb_mid_{window}'] = bollinger.bollinger_mavg()

            # Перевіряємо, чи mid не дорівнює нулю перед діленням
            bb_mid = data[f'bb_mid_{window}']
            bb_width = (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'])
            data[f'bb_width_{window}'] = bb_width / bb_mid.replace(0, np.nan)

            bb_range = (data[f'bb_upper_{window}'] - data[f'bb_lower_{window}'])
            data[f'bb_pct_{window}'] = (data['close'] - data[f'bb_lower_{window}']) / bb_range.replace(0, np.nan)

        except Exception as e:
            print(f"Попередження: Не вдалося розрахувати Bollinger Bands ({window}): {e}")
            data[f'bb_upper_{window}'] = np.nan
            data[f'bb_lower_{window}'] = np.nan
            data[f'bb_mid_{window}'] = np.nan
            data[f'bb_width_{window}'] = np.nan
            data[f'bb_pct_{window}'] = np.nan

    # ATR (Average True Range)
    try:
        atr = volatility.AverageTrueRange(data['high'], data['low'], data['close'])
        data['atr'] = atr.average_true_range()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати ATR: {e}")
        data['atr'] = np.nan

    # Keltner Channel
    try:
        keltner = volatility.KeltnerChannel(data['high'], data['low'], data['close'])
        data['keltner_upper'] = keltner.keltner_channel_hband()
        data['keltner_lower'] = keltner.keltner_channel_lband()
        data['keltner_mid'] = keltner.keltner_channel_mband()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати Keltner Channel: {e}")
        data['keltner_upper'] = np.nan
        data['keltner_lower'] = np.nan
        data['keltner_mid'] = np.nan

    # Історична волатільність
    for window in [5, 10, 20, 50]:
        try:
            returns = data['close'].pct_change()
            data[f'volatility_{window}'] = returns.rolling(window=window).std() * np.sqrt(window)
        except Exception as e:
            print(f"Попередження: Не вдалося розрахувати волатільність ({window}): {e}")
            data[f'volatility_{window}'] = np.nan

    return data


def add_volume_indicators(df):
    """Додає індикатори об'єму до датафрейму"""
    data = df.copy()

    # OBV (On-Balance Volume)
    try:
        data['obv'] = volume.OnBalanceVolumeIndicator(data['close'], data['volume']).on_balance_volume()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати OBV: {e}")
        data['obv'] = np.nan

    # Volume EMA
    for window in [10, 20, 50]:
        try:
            data[f'volume_ema_{window}'] = data['volume'].ewm(span=window, adjust=False).mean()
            volume_ema = data[f'volume_ema_{window}']
            data[f'volume_ratio_{window}'] = data['volume'] / volume_ema.replace(0, np.nan)
        except Exception as e:
            print(f"Попередження: Не вдалося розрахувати Volume EMA ({window}): {e}")
            data[f'volume_ema_{window}'] = np.nan
            data[f'volume_ratio_{window}'] = np.nan

    # Chaikin Money Flow
    try:
        data['cmf'] = volume.ChaikinMoneyFlowIndicator(data['high'], data['low'], data['close'],
                                                          data['volume']).chaikin_money_flow()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати CMF: {e}")
        data['cmf'] = np.nan

    # Ease of Movement
    try:
        data['eom'] = volume.EaseOfMovementIndicator(data['high'], data['low'], data['volume']).ease_of_movement()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати EOM: {e}")
        data['eom'] = np.nan

    # Force Index
    try:
        data['force_index'] = volume.ForceIndexIndicator(data['close'], data['volume']).force_index()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати Force Index: {e}")
        data['force_index'] = np.nan

    # Volume Price Trend
    try:
        data['vpt'] = volume.VolumePriceTrendIndicator(data['close'], data['volume']).volume_price_trend()
    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати VPT: {e}")
        data['vpt'] = np.nan

    return data


def add_cycle_indicators(df):
    """Додає часові/циклічні ознаки"""
    data = df.copy()

    try:
        # Переконуємося, що timestamp у форматі datetime
        if 'timestamp' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            data['timestamp'] = pd.to_datetime(data['timestamp'])

        # Години дня (циклічне представлення)
        data['hour_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.hour / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.hour / 24)

        # День тижня (циклічне представлення)
        data['day_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.dayofweek / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.dayofweek / 7)

        # Місяць року (циклічне представлення)
        data['month_sin'] = np.sin(2 * np.pi * data['timestamp'].dt.month / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['timestamp'].dt.month / 12)

    except Exception as e:
        print(f"Попередження: Не вдалося розрахувати циклічні індикатори: {e}")
        for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']:
            data[col] = np.nan

    return data