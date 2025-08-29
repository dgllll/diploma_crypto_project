import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
import datetime
import time
import requests
from typing import Optional, List
import warnings

warnings.filterwarnings('ignore')


def load_bitcoin_data_yahoo(period='2y', interval='1h', max_retries=3):
    """Завантажує дані Bitcoin з Yahoo Finance з повторними спробами"""
    print(f"Спроба завантаження даних Bitcoin з Yahoo Finance...")
    print(f"Період: {period}, Інтервал: {interval}")

    for attempt in range(max_retries):
        try:
            print(f"Спроба {attempt + 1} з {max_retries}")

            # Створюємо тікер
            btc = yf.Ticker("BTC-USD")

            # Завантажуємо дані
            df = btc.history(period=period, interval=interval, auto_adjust=True, prepost=True)

            if df.empty:
                print(f"Отримано пустий датафрейм на спробі {attempt + 1}")
                if attempt < max_retries - 1:
                    print("Очікування 5 секунд перед наступною спробою...")
                    time.sleep(5)
                continue

            # Перевіряємо структуру даних
            print(f"Завантажено {len(df)} рядків даних")
            print(f"Колонки: {df.columns.tolist()}")
            print(f"Перші 5 рядків:")
            print(df.head())

            # Перетворюємо індекс в колонку timestamp
            df = df.reset_index()

            # Стандартизуємо назви колонок
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

            # Перевіряємо, чи є дані за останні кілька днів
            latest_date = df['timestamp'].max()
            days_ago = (datetime.datetime.now(latest_date.tz) - latest_date).days

            if days_ago > 7:
                print(f"Увага: Останні дані датовані {days_ago} днями тому")

            print(f"Успішно завантажено дані з Yahoo Finance!")
            print(f"Період даних: з {df['timestamp'].min()} до {df['timestamp'].max()}")

            return df

        except Exception as e:
            print(f"Помилка на спробі {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                print("Очікування 10 секунд перед наступною спробою...")
                time.sleep(10)
            else:
                print("Всі спроби завантаження з Yahoo Finance невдалі")

    return pd.DataFrame()


def load_bitcoin_data_coinbase():
    """Завантажує дані Bitcoin з Coinbase Pro API"""
    print("Спроба завантаження даних з Coinbase Pro...")

    try:
        # URL для Coinbase Pro API
        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

        # Параметри запиту (за останні 365 днів, 1-годинні свічки)
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=365)

        params = {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'granularity': 3600  # 1 година в секундах
        }

        # Виконуємо запит
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if not data:
            print("Отримано пусті дані з Coinbase Pro")
            return pd.DataFrame()

        # Перетворюємо в датафрейм
        # Coinbase повертає дані у форматі: [timestamp, low, high, open, close, volume]
        df = pd.DataFrame(data, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])

        # Перетворюємо timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

        # Змінюємо порядок колонок для стандартизації
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()

        # Сортуємо за часом
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Успішно завантажено {len(df)} рядків з Coinbase Pro")
        print(f"Період даних: з {df['timestamp'].min()} до {df['timestamp'].max()}")

        return df

    except Exception as e:
        print(f"Помилка завантаження з Coinbase Pro: {str(e)}")
        return pd.DataFrame()


def load_bitcoin_data_binance():
    """Завантажує дані Bitcoin з Binance через ccxt"""
    print("Спроба завантаження даних з Binance...")

    try:
        # Створюємо з'єднання з Binance
        exchange = ccxt.binance({
            'apiKey': None,  # Публічні дані не потребують API ключа
            'secret': None,
            'timeout': 30000,
            'enableRateLimit': True,
        })

        # Символ для торгівлі
        symbol = 'BTC/USDT'
        timeframe = '1h'

        # Розраховуємо timestamp для початку (365 днів тому)
        since = int((datetime.datetime.now() - datetime.timedelta(days=365)).timestamp() * 1000)

        # Завантажуємо дані частинами (Binance має ліміт на кількість свічок)
        all_data = []
        limit = 1000  # Максимум свічок за один запит

        current_since = since

        while True:
            print(f"Завантаження даних з {datetime.datetime.fromtimestamp(current_since / 1000)}")

            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, current_since, limit)

            if not ohlcv:
                break

            all_data.extend(ohlcv)

            # Перевіряємо, чи потрібно завантажувати ще дані
            last_timestamp = ohlcv[-1][0]
            current_since = last_timestamp + 1

            # Якщо дійшли до поточного часу, зупиняємося
            if last_timestamp >= int(datetime.datetime.now().timestamp() * 1000):
                break

            # Пауза для дотримання rate limit
            time.sleep(1)

        if not all_data:
            print("Не вдалося завантажити дані з Binance")
            return pd.DataFrame()

        # Перетворюємо в датафрейм
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Видаляємо дублікати та сортуємо
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        print(f"Успішно завантажено {len(df)} рядків з Binance")
        print(f"Період даних: з {df['timestamp'].min()} до {df['timestamp'].max()}")

        return df

    except Exception as e:
        print(f"Помилка завантаження з Binance: {str(e)}")
        return pd.DataFrame()


def create_sample_data():
    """Створює зразкові дані для тестування (якщо всі інші джерела не працюють)"""
    print("Створення зразкових даних для тестування...")

    # Створюємо синтетичні дані на основі випадкового блукання з трендом
    np.random.seed(42)

    # Параметри
    n_days = 365
    n_hours = n_days * 24

    # Часові мітки
    start_date = datetime.datetime.now() - datetime.timedelta(days=n_days)
    timestamps = [start_date + datetime.timedelta(hours=i) for i in range(n_hours)]

    # Початкова ціна Bitcoin (приблизно)
    initial_price = 30000

    # Генеруємо ціни з випадковим блуканням та трендом
    prices = [initial_price]

    for i in range(1, n_hours):
        # Випадкова зміна цін (-2% до +2%)
        change = np.random.normal(0, 0.02)

        # Невеликий позитивний тренд
        trend = 0.0001

        # Нова ціна
        new_price = prices[-1] * (1 + change + trend)
        prices.append(max(new_price, 1))  # Ціна не може бути від'ємною

    # Створюємо OHLCV дані
    data = []
    for i in range(len(timestamps)):
        close_price = prices[i]

        # Open - ціна попередньої години (або близька до неї)
        open_price = prices[i - 1] if i > 0 else close_price
        open_price *= (1 + np.random.normal(0, 0.005))

        # High та Low на основі волатільності
        volatility = np.random.uniform(0.01, 0.05)
        high_price = max(open_price, close_price) * (1 + volatility)
        low_price = min(open_price, close_price) * (1 - volatility)

        # Volume (випадковий)
        volume = np.random.uniform(1000, 10000)

        data.append({
            'timestamp': timestamps[i],
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': round(volume, 2)
        })

    df = pd.DataFrame(data)

    print(f"Створено {len(df)} рядків синтетичних даних")
    print(f"Період: з {df['timestamp'].min()} до {df['timestamp'].max()}")
    print(f"Діапазон цін: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def load_bitcoin_data(prefer_source='yahoo', period='2y', interval='1h'):
    """
    Головна функція для завантаження даних Bitcoin з автоматичним вибором джерела

    Parameters:
    -----------
    prefer_source : str
        Переважне джерело даних ('yahoo', 'coinbase', 'binance')
    period : str
        Період для Yahoo Finance
    interval : str
        Інтервал для Yahoo Finance

    Returns:
    --------
    pandas.DataFrame
        Датафрейм з OHLCV даними
    """

    sources = {
        'yahoo': lambda: load_bitcoin_data_yahoo(period, interval),
        'coinbase': load_bitcoin_data_coinbase,
        'binance': load_bitcoin_data_binance
    }

    # Спочатку спробуємо переважне джерело
    if prefer_source in sources:
        print(f"Спроба завантаження з переважного джерела: {prefer_source}")
        df = sources[prefer_source]()

        if not df.empty:
            return df

        print(f"Переважне джерело {prefer_source} не працює, пробуємо інші...")

    # Спробуємо всі інші джерела
    for source_name, source_func in sources.items():
        if source_name == prefer_source:
            continue  # Вже спробували

        print(f"Спроба завантаження з {source_name}...")
        df = source_func()

        if not df.empty:
            return df

    # Якщо всі джерела не працюють, створюємо синтетичні дані
    print("Всі джерела реальних даних недоступні. Створюємо синтетичні дані...")
    df = create_sample_data()

    return df


def validate_data(df):
    """Перевіряє якість завантажених даних"""
    if df.empty:
        return False, "Датафрейм пустий"

    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, f"Відсутні колонки: {missing_columns}"

    # Перевіряємо на NaN значення
    nan_counts = df[required_columns].isnull().sum()
    if nan_counts.sum() > 0:
        return False, f"Знайдено NaN значення: {nan_counts.to_dict()}"

    # Перевіряємо логічність OHLC даних
    invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | (df['high'] < df['close']) | \
                   (df['low'] > df['open']) | (df['low'] > df['close'])

    if invalid_ohlc.sum() > 0:
        return False, f"Знайдено {invalid_ohlc.sum()} рядків з некоректними OHLC даними"

    # Перевіряємо на від'ємні ціни
    negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
    if negative_prices.sum() > 0:
        return False, f"Знайдено {negative_prices.sum()} рядків з від'ємними цінами"

    return True, "Дані валідні"


def save_data_to_csv(df, filename):
    """Зберігає дані у CSV файл"""
    try:
        df.to_csv(filename, index=False)
        print(f"Дані збережено в {filename}")
        return True
    except Exception as e:
        print(f"Помилка збереження даних: {str(e)}")
        return False


def load_data_from_csv(filename):
    """Завантажує дані з CSV файлу"""
    try:
        df = pd.read_csv(filename)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Дані завантажено з {filename}")
        return df
    except Exception as e:
        print(f"Помилка завантаження даних з файлу: {str(e)}")
        return pd.DataFrame()


# Тестова функція
def test_data_loading():
    """Тестує всі методи завантаження даних"""
    print("=== Тестування завантаження даних ===")

    # Тестуємо кожне джерело
    sources = ['yahoo', 'coinbase', 'binance']

    for source in sources:
        print(f"\n--- Тестування {source} ---")
        df = load_bitcoin_data(prefer_source=source)

        if not df.empty:
            is_valid, message = validate_data(df)
            print(f"Результат валідації: {message}")

            if is_valid:
                print(f"✓ {source}: Успіх! Завантажено {len(df)} рядків")
                print(f"  Період: {df['timestamp'].min()} - {df['timestamp'].max()}")
                print(f"  Ціновий діапазон: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
                return df
            else:
                print(f"✗ {source}: Дані невалідні - {message}")
        else:
            print(f"✗ {source}: Не вдалося завантажити дані")

    print("\n--- Використання синтетичних даних ---")
    df = create_sample_data()
    is_valid, message = validate_data(df)
    print(f"Результат валідації синтетичних даних: {message}")

    return df


if __name__ == "__main__":
    # Запускаємо тест, якщо файл викликається напряму
    test_data_loading()