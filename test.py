"""
Тестовий скрипт для перевірки завантаження даних
"""

from data.data_loader import test_data_loading, load_bitcoin_data, validate_data
import pandas as pd


def main():
    print("🔍 Тестування завантаження даних Bitcoin")
    print("=" * 50)

    # Тестуємо всі джерела
    df = test_data_loading()

    if not df.empty:
        print("\n📊 Базова статистика завантажених даних:")
        print(f"Кількість рядків: {len(df)}")
        print(f"Кількість колонок: {len(df.columns)}")
        print(f"Колонки: {list(df.columns)}")
        print(f"Період даних: {df['timestamp'].min()} - {df['timestamp'].max()}")
        print(f"Статистика ціни закриття:")
        print(df['close'].describe())

        print("\n🔍 Перші 5 рядків:")
        print(df.head())

        print("\n✅ Тест завершено успішно!")
    else:
        print("\n❌ Не вдалося завантажити дані")


if __name__ == "__main__":
    main()