# custom_data_loader.py - Завантаження даних тільки з Binance за конкретні періоди

import pandas as pd
import ccxt
import time
from datetime import datetime, timedelta
import numpy as np


def load_bitcoin_binance_dates(start_date, end_date, interval='1h', symbol='BTC/USDT'):
    """
    🎯 Завантажує дані Bitcoin з Binance за конкретний період

    Parameters:
    -----------
    start_date : str
        Початкова дата у форматі 'YYYY-MM-DD'
    end_date : str
        Кінцева дата у форматі 'YYYY-MM-DD'
    interval : str
        Інтервал даних ('1h', '4h', '1d', '5m', тощо)
    symbol : str
        Торгова пара (за замовчуванням 'BTC/USDT')

    Examples:
    ---------
    # Тільки 2024 рік
    df = load_bitcoin_binance_dates('2024-01-01', '2024-12-31')

    # Конкретний період з низькими цінами
    df = load_bitcoin_binance_dates('2023-01-01', '2023-06-30')

    # Період падіння ринку
    df = load_bitcoin_binance_dates('2022-01-01', '2022-12-31')
    """
    print(f"🔄 Завантаження даних {symbol} з Binance: {start_date} - {end_date}")

    try:
        # Створюємо з'єднання з Binance
        exchange = ccxt.binance({
            'apiKey': None,  # Публічні дані не потребують API ключа
            'secret': None,
            'timeout': 30000,
            'enableRateLimit': True,
            'sandbox': False,
            'options': {
                'defaultType': 'spot'  # Спот торгівля
            }
        })

        # Перетворюємо дати в timestamp (мілісекунди)
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        print(f"📅 Період в timestamp: {start_ts} - {end_ts}")

        # Збираємо всі дані частинами
        all_data = []
        limit = 1000  # Максимум свічок за один запит
        current_ts = start_ts

        request_count = 0
        max_requests = 100  # Обмеження для безпеки

        while current_ts < end_ts and request_count < max_requests:
            try:
                print(f"🔄 Запит #{request_count + 1}: {datetime.fromtimestamp(current_ts / 1000)}")

                # Завантажуємо дані
                ohlcv = exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=interval,
                    since=current_ts,
                    limit=limit
                )

                if not ohlcv:
                    print("⚠️ Отримано пустий відповідь, припиняємо")
                    break

                print(f"✅ Отримано {len(ohlcv)} свічок")
                all_data.extend(ohlcv)

                # Оновлюємо timestamp для наступного запиту
                last_timestamp = ohlcv[-1][0]
                current_ts = last_timestamp + 1

                # Якщо дійшли до кінцевої дати, зупиняємося
                if last_timestamp >= end_ts:
                    print("✅ Досягнуто кінцеву дату")
                    break

                request_count += 1

                # Пауза для дотримання rate limit
                time.sleep(exchange.rateLimit / 1000)

            except ccxt.RateLimitExceeded:
                print("⏳ Rate limit перевищено, чекаємо 60 секунд...")
                time.sleep(60)
                continue

            except Exception as e:
                print(f"⚠️ Помилка в запиті #{request_count + 1}: {e}")
                time.sleep(5)
                continue

        if not all_data:
            print("❌ Не вдалося завантажити жодних даних")
            return pd.DataFrame()

        print(f"📊 Всього завантажено {len(all_data)} свічок за {request_count} запитів")

        # Перетворюємо в DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Перетворюємо timestamp з мілісекунд в datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Фільтруємо за потрібний період (на всякий випадок)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # До кінця дня

        df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] < end_dt)]

        # Видаляємо дублікати та сортуємо
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

        # Перетворюємо типи даних
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Видаляємо рядки з NaN
        df = df.dropna()

        if df.empty:
            print("❌ Після обробки дані стали пустими")
            return pd.DataFrame()

        # Статистика завантажених даних
        print(f"✅ Успішно оброблено {len(df)} рядків")
        print(f"📅 Фактичний період: {df['timestamp'].min()} - {df['timestamp'].max()}")
        print(f"💰 Діапазон цін: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
        print(f"📊 Середня ціна: ${df['close'].mean():.0f}")
        print(f"📈 Волатільність: {df['close'].std() / df['close'].mean() * 100:.1f}%")

        return df

    except ccxt.NetworkError as e:
        print(f"🌐 Мережева помилка: {e}")
        return pd.DataFrame()

    except ccxt.ExchangeError as e:
        print(f"🏛️ Помилка біржі: {e}")
        return pd.DataFrame()

    except Exception as e:
        print(f"❌ Непередбачена помилка: {e}")
        return pd.DataFrame()


# 🎯 ПРЕДЕФІНОВАНІ ПОПУЛЯРНІ ПЕРІОДИ
POPULAR_PERIODS = {
    'btc_2024': {
        'start': '2024-01-01',
        'end': '2024-12-31',
        'description': 'Весь 2024 рік - помірні ціни'
    },

    'btc_2023': {
        'start': '2023-01-01',
        'end': '2023-12-31',
        'description': '2023 рік - відносно низькі ціни'
    },

    'btc_bear_2022': {
        'start': '2022-01-01',
        'end': '2022-12-31',
        'description': '2022 - ведмежий ринок, низькі ціни'
    },

    'btc_bull_2021': {
        'start': '2021-01-01',
        'end': '2021-12-31',
        'description': '2021 - бичачий ринок, високі ціни'
    },

    'btc_recent_6m': {
        'start': (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'),
        'end': datetime.now().strftime('%Y-%m-%d'),
        'description': 'Останні 6 місяців'
    },

    'btc_corona_crash': {
        'start': '2020-01-01',
        'end': '2020-12-31',
        'description': '2020 - COVID кріза та відновлення'
    },

    'btc_mid_2024': {
        'start': '2024-04-01',
        'end': '2024-09-30',
        'description': 'Середина 2024 року - активний період'
    },

    'btc_q1_2024': {
        'start': '2024-01-01',
        'end': '2024-03-31',
        'description': 'Q1 2024 - початок року'
    }
}


def load_popular_period_binance(period_name, interval='1h'):
    """
    🎯 Завантажує дані з Binance за популярним періодом

    Examples:
    ---------
    df = load_popular_period_binance('btc_2024')
    df = load_popular_period_binance('btc_bear_2022')
    df = load_popular_period_binance('btc_recent_6m')
    """
    if period_name not in POPULAR_PERIODS:
        print(f"❌ Період '{period_name}' не знайдено")
        print(f"Доступні періоди: {list(POPULAR_PERIODS.keys())}")
        return pd.DataFrame()

    period = POPULAR_PERIODS[period_name]
    print(f"🎯 Завантаження з Binance: {period['description']}")

    return load_bitcoin_binance_dates(
        period['start'],
        period['end'],
        interval
    )


def compare_periods_binance():
    """
    📊 Порівняння різних періодів з Binance
    """
    print("📊 ПОРІВНЯННЯ ПЕРІОДІВ BTC (BINANCE):")
    print("=" * 80)
    print(f"{'Période':15} | {'Середня':>8} | {'Мін-Макс':>15} | {'Волат':>6} | {'Рядків':>7}")
    print("-" * 80)

    for name, period in POPULAR_PERIODS.items():
        try:
            df = load_bitcoin_binance_dates(period['start'], period['end'], '1d')

            if not df.empty:
                min_price = df['close'].min()
                max_price = df['close'].max()
                avg_price = df['close'].mean()
                volatility = df['close'].std() / avg_price * 100
                rows = len(df)

                print(
                    f"{name:15} | ${avg_price:7.0f} | ${min_price:6.0f}-${max_price:6.0f} | {volatility:5.1f}% | {rows:6d}")
            else:
                print(f"{name:15} | Помилка завантаження")

        except Exception as e:
            print(f"{name:15} | Помилка: {str(e)[:30]}...")


def test_binance_connection():
    """
    🧪 Тест з'єднання з Binance
    """
    print("🧪 Тестування з'єднання з Binance...")

    try:
        exchange = ccxt.binance({
            'timeout': 10000,
            'enableRateLimit': True,
        })

        # Тестовий запит - останні 10 свічок
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1h', limit=10)

        if ohlcv:
            print(f"✅ З'єднання успішне! Отримано {len(ohlcv)} тестових свічок")

            # Показуємо останню ціну
            last_price = ohlcv[-1][4]  # close price
            last_time = datetime.fromtimestamp(ohlcv[-1][0] / 1000)
            print(f"📊 Остання ціна BTC: ${last_price:,.0f} на {last_time}")

            return True
        else:
            print("❌ Отримано пустий відповідь")
            return False

    except Exception as e:
        print(f"❌ Помилка з'єднання: {e}")
        return False


# 🔧 ІНТЕГРАЦІЯ З ВАШИМ ПРОЕКТОМ
def modify_data_preparation_for_binance(start_date, end_date):
    """
    Модифікована версія run_data_preparation() для Binance з кастомними датами
    """
    print("=" * 50)
    print(f"КРОК 1: Завантаження даних з BINANCE за період {start_date} - {end_date}")
    print("=" * 50)

    try:
        # Тест з'єднання
        if not test_binance_connection():
            raise Exception("Не вдалося підключитися до Binance")

        # Завантажуємо дані за вказаний період
        df = load_bitcoin_binance_dates(start_date, end_date, '1h')

        if df.empty:
            raise ValueError("Не вдалося завантажити дані за вказаний період!")

        # Валідація даних
        from data.data_loader import validate_data
        is_valid, message = validate_data(df)

        if not is_valid:
            print(f"⚠️ Попередження валідації: {message}")
            # Але продовжуємо, якщо дані не критично пошкоджені

        # Стандартна обробка
        from data.data_processor import preprocess_data, split_data

        print("Попередня обробка даних...")
        df = preprocess_data(df)

        if df.empty:
            raise ValueError("Дані стали пустими після попередньої обробки!")

        print("Розділення на набори...")
        train_df, val_df, test_df = split_data(
            df,
            test_size=0.3,  # Більший тестовий набір для кращого бектесту
            validation_size=0.1
        )

        print(f"✅ Тренувальний набір: {len(train_df)} рядків")
        print(f"✅ Валідаційний набір: {len(val_df)} рядків")
        print(f"✅ Тестовий набір: {len(test_df)} рядків")

        return train_df, val_df, test_df

    except Exception as e:
        print(f"❌ Помилка: {e}")
        raise


if __name__ == "__main__":
    # Тестування функцій
    print("🧪 Тестування Binance завантажувача...")

    # Тест з'єднання
    test_binance_connection()

    # Порівняння періодів
    print("\n" + "=" * 50)
    compare_periods_binance()

    # Тест завантаження невеликого періоду
    print("\n" + "=" * 50)
    print("🧪 Тест завантаження тижня даних...")
    test_df = load_bitcoin_binance_dates('2024-01-01', '2024-01-07', '1h')

    if not test_df.empty:
        print(f"✅ Тест пройшов! Завантажено {len(test_df)} рядків")
    else:
        print("❌ Тест не пройшов")