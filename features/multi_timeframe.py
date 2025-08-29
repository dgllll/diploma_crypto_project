import pandas as pd
import numpy as np
import ta


def resample_ohlcv(df, timeframe):
    """Ресемплінг OHLCV даних до бажаного таймфрейму"""
    # Переконуємося, що timestamp є DatetimeIndex
    if 'timestamp' in df.columns:
        df_resampled = df.set_index('timestamp')
    else:
        df_resampled = df.copy()

    # Ресемплінг
    resampled = df_resampled.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    return resampled


def create_multitimeframe_features(df, base_timeframe='1H', higher_timeframes=[]):
    """Створює ознаки з різних таймфреймів"""
    features = df.copy()

    # Якщо немає вищих таймфреймів, повертаємо без змін
    if not higher_timeframes:
        print("Мультитаймфрейм ознаки не створюються - не вказано таймфрейми")
        return features

    print(f"Створення мультитаймфрейм ознак для: {higher_timeframes}")

    # Встановлюємо timestamp як індекс, якщо потрібно
    if 'timestamp' in features.columns:
        features_with_ts = features.set_index('timestamp')
        original_index = features.index  # Зберігаємо оригінальний індекс
    else:
        features_with_ts = features.copy()
        original_index = features.index

    # Для кожного вищого таймфрейму
    for tf in higher_timeframes:
        print(f"Обробка таймфрейму {tf}...")

        try:
            # Ресемплінг до вищого таймфрейму
            resampled = resample_ohlcv(features, tf)

            if resampled.empty:
                print(f"Попередження: Пустий результат ресемплінгу для {tf}")
                continue

            print(f"Ресемплінг {tf}: {len(resampled)} рядків")

            # Розрахунок базових індикаторів на вищому таймфреймі
            indicators = {}

            # RSI
            try:
                indicators['rsi'] = ta.momentum.RSIIndicator(resampled['close']).rsi()
            except Exception as e:
                print(f"Помилка розрахунку RSI для {tf}: {e}")
                indicators['rsi'] = pd.Series(index=resampled.index, dtype=float)

            # MACD
            try:
                macd = ta.trend.MACD(resampled['close'])
                indicators['macd'] = macd.macd()
                indicators['macd_signal'] = macd.macd_signal()
            except Exception as e:
                print(f"Помилка розрахунку MACD для {tf}: {e}")
                indicators['macd'] = pd.Series(index=resampled.index, dtype=float)
                indicators['macd_signal'] = pd.Series(index=resampled.index, dtype=float)

            # ADX
            try:
                adx_ind = ta.trend.ADXIndicator(resampled['high'], resampled['low'], resampled['close'])
                indicators['adx'] = adx_ind.adx()
            except Exception as e:
                print(f"Помилка розрахунку ADX для {tf}: {e}")
                indicators['adx'] = pd.Series(index=resampled.index, dtype=float)

            # Bollinger Bands width
            try:
                bollinger = ta.volatility.BollingerBands(resampled['close'])
                bb_upper = bollinger.bollinger_hband()
                bb_lower = bollinger.bollinger_lband()
                bb_middle = bollinger.bollinger_mavg()
                indicators['bb_width'] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
            except Exception as e:
                print(f"Помилка розрахунку Bollinger Bands для {tf}: {e}")
                indicators['bb_width'] = pd.Series(index=resampled.index, dtype=float)

            # Повертаємо до базового таймфрейму через передискретизацію
            for indicator_name, indicator_series in indicators.items():
                col_name = f"{tf}_{indicator_name}"

                # Створюємо серію з правильним індексом
                if 'timestamp' in features.columns:
                    # Використовуємо forward fill для заповнення значень
                    reindexed_series = indicator_series.reindex(
                        features_with_ts.index, method='ffill'
                    )

                    # Додаємо до основного датафрейму
                    features[col_name] = reindexed_series.values
                else:
                    # Якщо немає timestamp, просто заповнюємо NaN
                    features[col_name] = np.nan

                # Перевіряємо кількість валідних значень
                valid_count = features[col_name].notna().sum()
                print(f"  {col_name}: {valid_count} валідних значень з {len(features)}")

        except Exception as e:
            print(f"Помилка при обробці таймфрейму {tf}: {e}")
            # Додаємо колонки з NaN при помилці
            for indicator_name in ['rsi', 'macd', 'macd_signal', 'adx', 'bb_width']:
                col_name = f"{tf}_{indicator_name}"
                features[col_name] = np.nan

    # Додавання взаємодій між таймфреймами (тільки якщо є валідні дані)
    try:
        for tf in higher_timeframes:
            # RSI дивергенції між таймфреймами
            if 'rsi' in features.columns and f'{tf}_rsi' in features.columns:
                rsi_base = features['rsi']
                rsi_higher = features[f'{tf}_rsi']
                if rsi_base.notna().sum() > 0 and rsi_higher.notna().sum() > 0:
                    features[f'rsi_divergence_{tf}'] = rsi_base - rsi_higher
                else:
                    features[f'rsi_divergence_{tf}'] = np.nan

            # MACD дивергенції
            if 'macd_12_26' in features.columns and f'{tf}_macd' in features.columns:
                macd_base = features['macd_12_26']
                macd_higher = features[f'{tf}_macd']
                if macd_base.notna().sum() > 0 and macd_higher.notna().sum() > 0:
                    features[f'macd_divergence_{tf}'] = macd_base - macd_higher
                else:
                    features[f'macd_divergence_{tf}'] = np.nan

    except Exception as e:
        print(f"Помилка при створенні дивергенцій: {e}")

    return features


def create_lag_features(df, lag_list, columns=None):
    """Створює лагові ознаки для вказаних колонок"""
    features = df.copy()

    # Якщо колонки не вказані, використовуємо лише 'close'
    if columns is None:
        columns = ['close']

    # Фільтруємо тільки існуючі колонки
    existing_columns = [col for col in columns if col in features.columns]

    if not existing_columns:
        print("Попередження: Жодна з вказаних колонок для лагових ознак не існує")
        return features

    print(f"Створення лагових ознак для колонок: {existing_columns}")

    # Для кожної колонки та лагу створюємо нову ознаку
    for col in existing_columns:
        for lag in lag_list:
            try:
                lag_col_name = f'{col}_lag_{lag}'
                features[lag_col_name] = features[col].shift(lag)

                # Перевіряємо кількість валідних значень
                valid_count = features[lag_col_name].notna().sum()
                if valid_count == 0:
                    print(f"Попередження: {lag_col_name} містить тільки NaN")

            except Exception as e:
                print(f"Помилка при створенні лагової ознаки {col}_lag_{lag}: {e}")
                features[f'{col}_lag_{lag}'] = np.nan

    return features