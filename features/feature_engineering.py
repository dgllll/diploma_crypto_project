import pandas as pd
import numpy as np
from features.technical_indicators import (
    add_trend_indicators, add_momentum_indicators,
    add_volatility_indicators, add_volume_indicators,
    add_cycle_indicators
)
from features.multi_timeframe import  create_lag_features


def build_advanced_features(df, timeframes=['1H', '4H', '1D']):
    """
    Створює розширений набір ознак з технічних індикаторів та різних часових масштабів

    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм з OHLCV даними (timestamp, open, high, low, close, volume)
    timeframes : list
        Список часових масштабів для агрегації

    Returns:
    --------
    pandas.DataFrame
        Датафрейм з розширеними ознаками
    """
    print(f"Початкова форма даних: {df.shape}")

    # Копіюємо базовий датафрейм
    features = df.copy()

    try:
        # Додаємо індикатори поступово з обробкою помилок
        print("Додавання індикаторів тренду...")
        features = add_trend_indicators(features)

        print("Додавання індикаторів моментуму...")
        features = add_momentum_indicators(features)

        print("Додавання індикаторів волатільності...")
        features = add_volatility_indicators(features)

        print("Додавання індикаторів об'єму...")
        features = add_volume_indicators(features)

        print("Додавання циклічних індикаторів...")
        features = add_cycle_indicators(features)

        print(f"Після додавання базових індикаторів: {features.shape}")

        # Додаємо лагові ознаки на основі послідовності Фібоначчі
        print("Додавання лагових ознак...")
        fib_lags = [1, 2, 3, 5, 8, 13, 21, 34]

        # Вибираємо тільки ті колонки, які існують
        lag_columns = []
        for col in ['close', 'volume', 'rsi', 'adx', 'volatility_20']:
            if col in features.columns:
                lag_columns.append(col)

        if lag_columns:
            features = create_lag_features(features, fib_lags, columns=lag_columns)
            print(f"Після додавання лагових ознак: {features.shape}")

        # Додаємо похідні та взаємодії ознак
        print("Створення похідних ознак...")

        # Моментум ціни для різних періодів
        for lag in fib_lags:
            if f'close_lag_{lag}' in features.columns:
                close_lag = features[f'close_lag_{lag}']
                features[f'momentum_{lag}'] = features['close'] / close_lag.replace(0, np.nan) - 1

        # Взаємодія між технічними індикаторами
        # RSI та MACD
        if 'rsi' in features.columns and 'macd_12_26' in features.columns:
            features['rsi_macd_interaction'] = features['rsi'] * features['macd_12_26']

        # Відношення волатільності до ADX (сила тренду)
        if 'volatility_20' in features.columns and 'adx' in features.columns:
            adx_safe = features['adx'].replace(0, np.nan)  # Замінюємо 0 на NaN для безпечного ділення
            features['vol_trend_ratio'] = features['volatility_20'] / adx_safe

        # Відношення ціни до різних МА для виявлення відхилень
        for window in [20, 50, 200]:
            if f'sma_{window}' in features.columns:
                sma_safe = features[f'sma_{window}'].replace(0, np.nan)
                features[f'close_to_sma_{window}_ratio'] = features['close'] / sma_safe

        # Виявлення фракталів (локальних максимумів/мінімумів)
        print("Створення фрактальних ознак...")
        try:
            features['fractal_high'] = (features['high'] > features['high'].shift(1)) & \
                                       (features['high'] > features['high'].shift(2)) & \
                                       (features['high'] > features['high'].shift(-1)) & \
                                       (features['high'] > features['high'].shift(-2))

            features['fractal_low'] = (features['low'] < features['low'].shift(1)) & \
                                      (features['low'] < features['low'].shift(2)) & \
                                      (features['low'] < features['low'].shift(-1)) & \
                                      (features['low'] < features['low'].shift(-2))

            # Кількість фракталів у вікні (показник зміни тренду)
            features['fractal_count_10'] = features['fractal_high'].rolling(10).sum() + features['fractal_low'].rolling(
                10).sum()

        except Exception as e:
            print(f"Попередження: Не вдалося створити фрактальні ознаки: {e}")

        print(f"Фінальна форма даних з ознаками: {features.shape}")

        # Очищення від NaN та безкінечних значень
        print("Очищення даних...")
        features = features.replace([np.inf, -np.inf], np.nan)

        # Виводимо інформацію про NaN значення
        nan_counts = features.isnull().sum()
        if nan_counts.sum() > 0:
            print(f"Кількість NaN значень по колонках:")
            nan_counts_nonzero = nan_counts[nan_counts > 0]
            if len(nan_counts_nonzero) > 10:
                print(f"Показуємо топ-10 колонок з найбільшою кількістю NaN:")
                print(nan_counts_nonzero.sort_values(ascending=False).head(10))
            else:
                print(nan_counts_nonzero)

        return features

    except Exception as e:
        print(f"Критична помилка при створенні ознак: {e}")
        print("Повертаємо базовий датафрейм...")
        return df


def prepare_features_targets(df, target_col='close', forecast_horizon=24, dropna=True):
    """
    Підготовка ознак та цільових змінних для моделі прогнозування

    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм з ознаками
    target_col : str
        Назва колонки для прогнозування
    forecast_horizon : int
        Горизонт прогнозування (кількість періодів вперед)
    dropna : bool
        Чи видаляти рядки з NaN

    Returns:
    --------
    tuple
        (X, y) - ознаки та цільові змінні
    """
    # Копіюємо датафрейм
    data = df.copy()

    # Створюємо цільову змінну - ціна через n періодів
    data[f'target_{target_col}_{forecast_horizon}'] = data[target_col].shift(-forecast_horizon)

    # Відокремлюємо ознаки від цільової змінної
    columns_to_drop = [f'target_{target_col}_{forecast_horizon}', 'timestamp', 'open', 'high', 'low', 'close', 'volume']

    # Видаляємо тільки ті колонки, які дійсно існують
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    X = data.drop(existing_columns_to_drop, axis=1, errors='ignore')
    y = data[f'target_{target_col}_{forecast_horizon}']

    print(f"Форма ознак до очищення: {X.shape}")
    print(f"Форма цільової змінної до очищення: {y.shape}")

    # Видаляємо рядки з NaN, якщо необхідно
    if dropna:
        # Знаходимо індекси без NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        print(f"Форма після видалення NaN - ознаки: {X.shape}, цільова змінна: {y.shape}")

    return X, y


def select_features(X_train, y_train, method='rf', n_features=50):
    """
    Відбір найважливіших ознак

    Parameters:
    -----------
    X_train : pandas.DataFrame
        Тренувальні ознаки
    y_train : pandas.Series
        Цільова змінна для тренування
    method : str
        Метод відбору ознак ('rf', 'xgb', 'mutual_info')
    n_features : int
        Кількість ознак для відбору

    Returns:
    --------
    tuple
        (selected_features, importances) - відібрані ознаки та їх важливості
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.feature_selection import SelectFromModel, mutual_info_regression
    import xgboost as xgb

    try:
        if method == 'rf':
            # Використання Random Forest для відбору ознак
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            importances = model.feature_importances_

        elif method == 'xgb':
            # Використання XGBoost для відбору ознак
            model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            importances = model.feature_importances_

        elif method == 'mutual_info':
            # Використання Mutual Information для відбору ознак
            importances = mutual_info_regression(X_train, y_train)

        else:
            raise ValueError("Непідтримуваний метод відбору ознак")

        # Створюємо словник "ознака: важливість"
        feature_importance = {feature: importance for feature, importance in zip(X_train.columns, importances)}

        # Сортуємо ознаки за важливістю
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Відбираємо top-n ознак
        selected_features = [feature for feature, _ in sorted_features[:n_features]]
        selected_importances = [importance for _, importance in sorted_features[:n_features]]

        return selected_features, selected_importances

    except Exception as e:
        print(f"Помилка при відборі ознак: {e}")
        # Повертаємо всі ознаки, якщо відбір не вдався
        return list(X_train.columns), [1.0] * len(X_train.columns)


def prepare_features_targets_robust(df, target_col='close', forecast_horizon=24,
                                    nan_threshold=0.5, min_samples=100):
    """
    Підготовка ознак та цільових змінних з покращеною обробкою NaN

    Parameters:
    -----------
    df : pandas.DataFrame
        Датафрейм з ознаками
    target_col : str
        Назва колонки для прогнозування
    forecast_horizon : int
        Горизонт прогнозування (кількість періодів вперед)
    nan_threshold : float
        Поріг NaN - колонки з більшою часткою NaN будуть видалені
    min_samples : int
        Мінімальна кількість зразків після очищення

    Returns:
    --------
    tuple
        (X, y) - ознаки та цільові змінні
    """
    print(f"Розпочинаємо підготовку ознак з порогом NaN: {nan_threshold}")

    # Копіюємо датафрейм
    data = df.copy()

    # Створюємо цільову змінну - ціна через n періодів
    data[f'target_{target_col}_{forecast_horizon}'] = data[target_col].shift(-forecast_horizon)

    # Відокремлюємо ознаки від цільової змінної
    columns_to_drop = [f'target_{target_col}_{forecast_horizon}', 'timestamp',
                       'open', 'high', 'low', 'close', 'volume']

    # Видаляємо тільки ті колонки, які дійсно існують
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    X = data.drop(existing_columns_to_drop, axis=1, errors='ignore')
    y = data[f'target_{target_col}_{forecast_horizon}']

    print(f"Початкова форма ознак: {X.shape}")
    print(f"Початкова форма цільової змінної: {y.shape}")

    # Аналіз NaN значень по колонках
    nan_percentages = X.isnull().sum() / len(X)

    print(f"Колонки з NaN > {nan_threshold * 100}%:")
    high_nan_cols = nan_percentages[nan_percentages > nan_threshold]
    if len(high_nan_cols) > 0:
        print(high_nan_cols.sort_values(ascending=False).head(10))

        # Видаляємо колонки з високим відсотком NaN
        cols_to_remove = high_nan_cols.index.tolist()
        X = X.drop(columns=cols_to_remove)
        print(f"Видалено {len(cols_to_remove)} колонок з високим відсотком NaN")
        print(f"Форма після видалення колонок: {X.shape}")

    # Тепер видаляємо рядки з NaN у цільовій змінній
    valid_target_idx = ~y.isna()
    X = X.loc[valid_target_idx]
    y = y.loc[valid_target_idx]

    print(f"Після видалення рядків з NaN у цільовій змінній: {X.shape}")

    # Видаляємо рядки, де занадто багато NaN в ознаках
    # Залишаємо рядки, де не більше 20% ознак є NaN
    max_nan_per_row = int(X.shape[1] * 0.2)
    nan_counts_per_row = X.isnull().sum(axis=1)
    valid_rows = nan_counts_per_row <= max_nan_per_row

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print(f"Після видалення рядків з занадто багатьма NaN: {X.shape}")

    # Заповнюємо решту NaN значень
    if X.isnull().sum().sum() > 0:
        print("Заповнення решти NaN значень...")

        # Для числових колонок використовуємо медіану
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                median_value = X[col].median()
                if pd.isna(median_value):
                    # Якщо медіана теж NaN, використовуємо 0
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(median_value)

        # Для boolean колонок використовуємо False
        bool_cols = X.select_dtypes(include=[bool]).columns
        for col in bool_cols:
            X[col] = X[col].fillna(False)

    print(f"Фінальна форма ознак: {X.shape}")
    print(f"Фінальна форма цільової змінної: {y.shape}")

    # Перевірка на мінімальну кількість зразків
    if len(X) < min_samples:
        print(f"Попередження: Мало зразків після очищення ({len(X)} < {min_samples})")
        if len(X) == 0:
            raise ValueError("Не залишилося жодного валідного зразка після очищення даних!")

    # Фінальна перевірка на NaN
    remaining_nan = X.isnull().sum().sum()
    if remaining_nan > 0:
        print(f"Попередження: Залишилося {remaining_nan} NaN значень")
    else:
        print("✓ Всі NaN значення успішно оброблені")

    return X, y