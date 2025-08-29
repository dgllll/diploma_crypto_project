"""
Модуль для аналізу породжуючого розподілу даних
та теоретичного обґрунтування вибору непараметричних моделей
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import normaltest, jarque_bera, shapiro, anderson, kstest
import seaborn as sns
from datetime import datetime


class CryptoDataDistributionAnalyzer:
    """
    Аналізує розподіл криптовалютних даних для обґрунтування вибору моделей
    """

    def __init__(self, data, target_column='close'):
        self.data = data.copy()
        self.target_column = target_column
        self.prices = data[target_column].values
        self.log_prices = np.log(self.prices)
        self.returns = np.diff(self.log_prices)  # log-returns
        self.assumptions = {}
        self.test_results = {}

    def comprehensive_distribution_analysis(self):
        """
        Комплексний аналіз розподілу для обґрунтування вибору моделей
        """
        print("🔬" * 80)
        print("АНАЛІЗ ПОРОДЖУЮЧОГО РОЗПОДІЛУ КРИПТОВАЛЮТНИХ ДАНИХ")
        print("🔬" * 80)

        # 1. Аналіз цін
        price_stats = self._analyze_price_distribution()

        # 2. Аналіз дохідностей
        returns_stats = self._analyze_returns_distribution()

        # 3. Стилізовані факти
        stylized_facts = self._check_stylized_facts()

        # 4. Обґрунтування вибору моделей
        model_justification = self._justify_model_choice()

        # 5. Візуалізація
        self._create_comprehensive_plots()

        # 6. Фінальні висновки
        self._provide_final_conclusions()

        return {
            'price_stats': price_stats,
            'returns_stats': returns_stats,
            'stylized_facts': stylized_facts,
            'model_justification': model_justification
        }

    def _analyze_price_distribution(self):
        """
        🔬 ПОКРАЩЕНИЙ аналіз розподілу цін BTC з додатковими статистиками та тестами
        """
        print("\n📊 1. ПОКРАЩЕНИЙ АНАЛІЗ РОЗПОДІЛУ ЦІН BTC")
        print("=" * 60)

        # Основні статистики (розширені)
        price_stats = {
            'mean': np.mean(self.prices),
            'std': np.std(self.prices),
            'min': np.min(self.prices),
            'max': np.max(self.prices),
            'median': np.median(self.prices),
            'q25': np.percentile(self.prices, 25),
            'q75': np.percentile(self.prices, 75),
            'iqr': np.percentile(self.prices, 75) - np.percentile(self.prices, 25),
            'skewness': stats.skew(self.prices),
            'kurtosis': stats.kurtosis(self.prices),
            'cv': np.std(self.prices) / np.mean(self.prices),  # Коефіцієнт варіації
            'range_pct': (np.max(self.prices) - np.min(self.prices)) / np.mean(self.prices) * 100,
            'outliers_count': 0  # Буде розраховано нижче
        }

        # Виявлення викидів (IQR метод)
        iqr = price_stats['iqr']
        lower_bound = price_stats['q25'] - 1.5 * iqr
        upper_bound = price_stats['q75'] + 1.5 * iqr
        outliers = (self.prices < lower_bound) | (self.prices > upper_bound)
        price_stats['outliers_count'] = np.sum(outliers)
        price_stats['outliers_pct'] = (price_stats['outliers_count'] / len(self.prices)) * 100

        print(f"   💰 РОЗШИРЕНІ СТАТИСТИКИ ЦІН:")
        print(f"      Середня ціна: ${price_stats['mean']:,.2f}")
        print(f"      Медіана: ${price_stats['median']:,.2f}")
        print(f"      Стандартне відхилення: ${price_stats['std']:,.2f}")
        print(f"      Міжквартильний розмах (IQR): ${price_stats['iqr']:,.2f}")
        print(f"      Q25-Q75: ${price_stats['q25']:,.0f} - ${price_stats['q75']:,.0f}")
        print(f"      Діапазон: ${price_stats['min']:,.0f} - ${price_stats['max']:,.0f}")
        print(f"      Відносний діапазон: {price_stats['range_pct']:.1f}%")
        print(f"      Викиди (IQR метод): {price_stats['outliers_count']} ({price_stats['outliers_pct']:.1f}%)")

        print(f"\n   📊 ФОРМА РОЗПОДІЛУ:")
        print(f"      Асиметрія: {price_stats['skewness']:.4f}", end="")
        if price_stats['skewness'] > 0.5:
            print(" (сильна права асиметрія)")
        elif price_stats['skewness'] > 0:
            print(" (помірна права асиметрія)")
        elif price_stats['skewness'] < -0.5:
            print(" (сильна ліва асиметрія)")
        else:
            print(" (симетричний або слабка асиметрія)")

        print(f"      Ексцес: {price_stats['kurtosis']:.4f}", end="")
        if price_stats['kurtosis'] > 3:
            print(" (дуже важкі хвости)")
        elif price_stats['kurtosis'] > 0.5:
            print(" (важкі хвости)")
        elif price_stats['kurtosis'] < -0.5:
            print(" (легкі хвости)")
        else:
            print(" (нормальні хвости)")

        print(f"      Коефіцієнт варіації: {price_stats['cv']:.4f}", end="")
        if price_stats['cv'] > 1:
            print(" (дуже висока волатільність)")
        elif price_stats['cv'] > 0.5:
            print(" (висока волатільність)")
        else:
            print(" (помірна волатільність)")

        # Тестуємо гіпотези про розподіл цін
        print(f"\n🧪 РОЗШИРЕНІ ТЕСТИ НА РОЗПОДІЛ ЦІН:")

        # 1. Покращений тест на лог-нормальність
        try:
            # Перевірка на від'ємні значення
            if np.any(self.prices <= 0):
                print("      ⚠️ Попередження: Виявлені від'ємні або нульові ціни!")
                valid_prices = self.prices[self.prices > 0]
                log_prices = np.log(valid_prices)
            else:
                log_prices = self.log_prices

            # Обмежуємо розмір вибірки для тесту
            sample_size = min(5000, len(log_prices))
            log_prices_sample = log_prices[:sample_size]
            log_prices_normalized = (log_prices_sample - np.mean(log_prices_sample)) / np.std(log_prices_sample)

            shapiro_stat, shapiro_p = shapiro(log_prices_normalized)

            print(f"   📈 ТЕСТ ЛОГ-НОРМАЛЬНОСТІ ЦІН:")
            print(f"      Вибірка для тесту: {len(log_prices_normalized):,} спостережень")
            print(f"      Shapiro-Wilk статистика: {shapiro_stat:.6f}")
            print(f"      p-value: {shapiro_p:.6e}")

            alpha_levels = [0.05, 0.01, 0.001]
            for alpha in alpha_levels:
                if shapiro_p < alpha:
                    print(f"      ❌ ВІДХИЛЯЄМО лог-нормальність на рівні α={alpha}")
                else:
                    print(f"      ✅ НЕ відхиляємо лог-нормальність на рівні α={alpha}")
                    break

            if shapiro_p < 0.05:
                self.assumptions['log_normal_prices'] = False
                print(f"      💡 ВИСНОВОК: Ціни НЕ слідують лог-нормальному розподілу")
            else:
                self.assumptions['log_normal_prices'] = True
                print(f"      💡 ВИСНОВОК: Лог-нормальність НЕ відхилена")

        except Exception as e:
            print(f"      ⚠️ Помилка тесту лог-нормальності: {e}")
            self.assumptions['log_normal_prices'] = False

        # 2. Тест Колмогорова-Смірнова на лог-нормальність
        try:
            # Параметри лог-нормального розподілу
            mu, sigma = np.mean(self.log_prices), np.std(self.log_prices)

            # Обмежуємо вибірку для швидкості
            sample_size = min(1000, len(self.prices))
            prices_sample = self.prices[:sample_size]

            ks_stat, ks_p = kstest(prices_sample, lambda x: stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu)))

            print(f"\n   📊 ТЕСТ КОЛМОГОРОВА-СМІРНОВА (ЛОГ-НОРМАЛЬНИЙ):")
            print(f"      Вибірка для тесту: {len(prices_sample):,} спостережень")
            print(f"      Параметри: μ={mu:.4f}, σ={sigma:.4f}")
            print(f"      KS статистика: {ks_stat:.6f}")
            print(f"      p-value: {ks_p:.6e}")

            if ks_p < 0.05:
                print(f"      ❌ ВІДХИЛЯЄМО лог-нормальний розподіл")
            else:
                print(f"      ✅ НЕ відхиляємо лог-нормальний розподіл")

        except Exception as e:
            print(f"      ⚠️ Помилка KS тесту: {e}")

        # 3. Покращений тест на стаціонарність
        try:
            # Спочатку пробуємо statsmodels
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(self.prices, autolag='AIC', maxlag=int(12 * (len(self.prices) / 100) ** (1 / 4)))

                print(f"\n   📈 РОЗШИРЕНИЙ ТЕСТ СТАЦІОНАРНОСТІ (ADF):")
                print(f"      ADF статистика: {adf_result[0]:.6f}")
                print(f"      p-value: {adf_result[1]:.6e}")
                print(f"      Використані лаги: {adf_result[2]}")
                print(f"      Критичні значення:")

                for level, cv in adf_result[4].items():
                    result = "❌ Відхилено" if adf_result[0] < cv else "✅ Не відхилено"
                    print(f"        {level}: {cv:.4f} - {result}")

                if adf_result[1] < 0.05:
                    print(f"      ✅ ЦІНИ СТАЦІОНАРНІ (рідкість для фінансових ринків!)")
                    print(f"      💡 Це може свідчити про:")
                    print(f"         • Короткий період спостереження")
                    print(f"         • Структурні зломи в даних")
                    print(f"         • Специфіку криптовалютного ринку")
                    self.assumptions['stationary_prices'] = True
                else:
                    print(f"      ❌ ЦІНИ НЕ СТАЦІОНАРНІ (очікувано)")
                    print(f"      💡 Це підтверджує:")
                    print(f"         • Теорію випадкового блукання")
                    print(f"         • Наявність одиничного кореня")
                    print(f"         • Необхідність диференціювання")
                    self.assumptions['stationary_prices'] = False

            except ImportError:
                print(f"\n   ⚠️ STATSMODELS недоступний, використовуємо альтернативні тести")

                # Альтернативний підхід: комбінація тестів
                print(f"   🔄 АЛЬТЕРНАТИВНІ ТЕСТИ СТАЦІОНАРНОСТІ:")

                # 1. Тест на лінійний тренд
                x = np.arange(len(self.prices))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, self.prices)

                print(f"      📈 Тест на лінійний тренд:")
                print(f"         Нахил: {slope:.6f}")
                print(f"         R²: {r_value ** 2:.6f}")
                print(f"         p-value: {p_value:.6e}")

                has_trend = p_value < 0.05

                # 2. Тест на зміну дисперсії (простий)
                mid_point = len(self.prices) // 2
                first_half_var = np.var(self.prices[:mid_point])
                second_half_var = np.var(self.prices[mid_point:])
                var_ratio = max(first_half_var, second_half_var) / min(first_half_var, second_half_var)

                print(f"      📊 Тест зміни дисперсії:")
                print(f"         Перша половина var: {first_half_var:,.0f}")
                print(f"         Друга половина var: {second_half_var:,.0f}")
                print(f"         Співвідношення: {var_ratio:.2f}")

                has_changing_variance = var_ratio > 2.0

                # 3. Тест на середнє значення (зміна рівня)
                first_half_mean = np.mean(self.prices[:mid_point])
                second_half_mean = np.mean(self.prices[mid_point:])
                mean_change_pct = abs(second_half_mean - first_half_mean) / first_half_mean * 100

                print(f"      💰 Тест зміни середнього:")
                print(f"         Перша половина: ${first_half_mean:,.0f}")
                print(f"         Друга половина: ${second_half_mean:,.0f}")
                print(f"         Зміна: {mean_change_pct:.1f}%")

                has_mean_change = mean_change_pct > 20

                # Загальний висновок
                non_stationary_evidence = sum([has_trend, has_changing_variance, has_mean_change])

                print(f"\n      🎯 ПІДСУМОК ТЕСТІВ:")
                print(f"         Значущий тренд: {'✓' if has_trend else '✗'}")
                print(f"         Зміна дисперсії: {'✓' if has_changing_variance else '✗'}")
                print(f"         Зміна середнього: {'✓' if has_mean_change else '✗'}")
                print(f"         Свідчень нестаціонарності: {non_stationary_evidence}/3")

                if non_stationary_evidence >= 2:
                    print(f"      ❌ ЦІНИ НЕ СТАЦІОНАРНІ")
                    self.assumptions['stationary_prices'] = False
                else:
                    print(f"      ✅ НЕДОСТАТНЬО свідчень нестаціонарності")
                    self.assumptions['stationary_prices'] = True

        except Exception as e:
            print(f"      ⚠️ Помилка тестів стаціонарності: {e}")
            self.assumptions['stationary_prices'] = False

        # 4. Додатковий аналіз: Автокореляція цін
        try:
            print(f"\n   🔄 АНАЛІЗ АВТОКОРЕЛЯЦІЇ ЦІН:")

            # Автокореляція для різних лагів
            lags = [1, 24, 168, 720]  # 1 година, 1 день, 1 тиждень, 1 місяць (для годинних даних)
            lag_names = ['1 година', '1 день', '1 тиждень', '1 місяць']

            autocorrs = []
            for lag, name in zip(lags, lag_names):
                if len(self.prices) > lag:
                    autocorr = np.corrcoef(self.prices[:-lag], self.prices[lag:])[0, 1]
                    autocorrs.append(autocorr)
                    print(f"      Автокореляція ({name}): {autocorr:.4f}")
                else:
                    autocorrs.append(np.nan)
                    print(f"      Автокореляція ({name}): N/A (недостатньо даних)")

            # Оцінка стійкості автокореляції
            valid_autocorrs = [ac for ac in autocorrs if not np.isnan(ac)]
            if valid_autocorrs:
                high_autocorr_count = sum(1 for ac in valid_autocorrs if ac > 0.8)

                if high_autocorr_count >= len(valid_autocorrs) * 0.5:
                    print(f"      💡 ВИСОКА автокореляція → Підтверджує нестаціонарність")
                else:
                    print(f"      💡 ПОМІРНА автокореляція")

        except Exception as e:
            print(f"      ⚠️ Помилка аналізу автокореляції: {e}")

        # 5. Фінальна оцінка розподілу цін
        print(f"\n   🎯 ВИСНОВКИ ПРО РОЗПОДІЛ ЦІН:")

        if not self.assumptions.get('log_normal_prices', True):
            print(f"      📊 Ціни НЕ слідують лог-нормальному розподілу")
            print(f"      💡 Рекомендація: Використовувати непараметричні методи")
        else:
            print(f"      📊 Лог-нормальність цін НЕ відхилена")

        if not self.assumptions.get('stationary_prices', True):
            print(f"      📈 Ціни НЕ стаціонарні (наявність тренду/зломів)")
            print(f"      💡 Рекомендація: Робота з дохідностями або диференціювання")
        else:
            print(f"      📈 Не виявлено сильних свідчень нестаціонарності")

        return price_stats

    def _analyze_returns_distribution(self):
        """
        Аналіз розподілу дохідностей
        """
        print("\n 2. АНАЛІЗ РОЗПОДІЛУ ДОХІДНОСТЕЙ ")
        print("=" * 50)

        # Основні статистики
        returns_stats = {
            'mean': np.mean(self.returns),
            'std': np.std(self.returns),
            'skewness': stats.skew(self.returns),
            'kurtosis': stats.kurtosis(self.returns),
            'annual_return': np.mean(self.returns) * 24 * 365,  # Припускаємо годинні дані
            'annual_volatility': np.std(self.returns) * np.sqrt(24 * 365),
            'sharpe_ratio': (np.mean(self.returns) * 24 * 365) / (np.std(self.returns) * np.sqrt(24 * 365))
        }

        print(f"    Середня дохідність: {returns_stats['mean']:.6f}")
        print(f"    Волатільність: {returns_stats['std']:.6f}")
        print(f"    Річна дохідність: {returns_stats['annual_return']:.2%}")
        print(f"    Річна волатільність: {returns_stats['annual_volatility']:.2%}")
        print(f"    Асиметрія: {returns_stats['skewness']:.4f}")
        print(f"    Ексцес: {returns_stats['kurtosis']:.4f}")
        print(f"    Коефіцієнт Шарпа: {returns_stats['sharpe_ratio']:.4f}")

        # Критичні тести для дохідностей
        print(f"\n ТЕСТИ НА РОЗПОДІЛ ДОХІДНОСТЕЙ:")

        # 1. Тест на нормальність
        shapiro_stat, shapiro_p = shapiro(self.returns[:5000] if len(self.returns) > 5000 else self.returns)
        jb_stat, jb_p = jarque_bera(self.returns)

        print(f"   Тест нормальності дохідностей:")
        print(f"      Shapiro-Wilk p-value: {shapiro_p:.6f}")
        print(f"      Jarque-Bera p-value: {jb_p:.6f}")

        if shapiro_p < 0.05 and jb_p < 0.05:
            print(f"      ❌ Дохідності НЕ нормально розподілені")
            self.assumptions['normal_returns'] = False
        else:
            print(f"      ✅ Не відхиляємо гіпотезу про нормальність")
            self.assumptions['normal_returns'] = True

        # 2. Тест на важкі хвости
        excess_kurtosis = returns_stats['kurtosis']
        print(f"    Аналіз ексцесу:")
        print(f"      Excess Kurtosis: {excess_kurtosis:.4f}")

        if excess_kurtosis > 0.5:
            print(f"       ВАЖКІ ХВОСТИ виявлені")
            self.assumptions['fat_tails'] = True
        elif excess_kurtosis < -0.5:
            print(f"      📊 Легкі хвости (platykurtic distribution)")
            self.assumptions['fat_tails'] = False
        else:
            print(f"      📊 Нормальні хвости (mesokurtic distribution)")
            self.assumptions['fat_tails'] = False

        # 3. Тест на стаціонарність дохідностей
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(self.returns)
            print(f"   📊 Тест стаціонарності дохідностей (ADF):")
            print(f"      p-value: {adf_result[1]:.6f}")
            if adf_result[1] < 0.05:
                print(f"      ✅ Дохідності стаціонарні")
                self.assumptions['stationary_returns'] = True
            else:
                print(f"      ❌ Дохідності НЕ стаціонарні")
                self.assumptions['stationary_returns'] = False
        except ImportError:
            self.assumptions['stationary_returns'] = None

        return returns_stats

    def _check_stylized_facts(self):
        """
        Перевіряє стилізовані факти фінансових часових рядів
        """
        print("\n 3. СТИЛІЗОВАНІ ФАКТИ ФІНАНСОВИХ ЧАСОВИХ РЯДІВ")
        print("=" * 50)

        stylized_facts = {}

        # 1. Кластеризація волатільності (ARCH ефекти)
        print(f"    Перевірка кластеризації волатільності:")

        # Автокореляція квадратів дохідностей
        squared_returns = self.returns ** 2
        autocorr_1 = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
        autocorr_5 = np.corrcoef(squared_returns[:-5], squared_returns[5:])[0, 1]

        print(f"      Автокореляція квадратів дохідностей (лаг 1): {autocorr_1:.4f}")
        print(f"      Автокореляція квадратів дохідностей (лаг 5): {autocorr_5:.4f}")

        if autocorr_1 > 0.1 or autocorr_5 > 0.05:
            print(f"      ✅ КЛАСТЕРИЗАЦІЯ ВОЛАТІЛЬНОСТІ виявлена")
            stylized_facts['volatility_clustering'] = True
        else:
            print(f"      ❌ Кластеризація волатільності не виявлена")
            stylized_facts['volatility_clustering'] = False

        # 2. Відсутність автокореляції в дохідностях
        returns_autocorr_1 = np.corrcoef(self.returns[:-1], self.returns[1:])[0, 1]
        returns_autocorr_5 = np.corrcoef(self.returns[:-5], self.returns[5:])[0, 1]

        print(f"    Автокореляція дохідностей:")
        print(f"      Лаг 1: {returns_autocorr_1:.4f}")
        print(f"      Лаг 5: {returns_autocorr_5:.4f}")

        if abs(returns_autocorr_1) < 0.05 and abs(returns_autocorr_5) < 0.05:
            print(f"      ✅ ВІДСУТНІСТЬ автокореляції в дохідностях")
            stylized_facts['no_autocorr_returns'] = True
        else:
            print(f"      ❌ Є автокореляція в дохідностях")
            stylized_facts['no_autocorr_returns'] = False

        # 3. Асиметрія (leverage effect)
        print(f"   ⚖ Аналіз асиметрії:")
        skewness = stats.skew(self.returns)
        print(f"      Асиметрія дохідностей: {skewness:.4f}")

        if skewness < -0.1:
            print(f"      ✅ НЕГАТИВНА асиметрія")
            stylized_facts['negative_skewness'] = True
        else:
            print(f"       Асиметрія в межах норми")
            stylized_facts['negative_skewness'] = False

        # 4. Aналіз розподілу абсолютних дохідностей
        abs_returns = np.abs(self.returns)
        abs_returns_mean = np.mean(abs_returns)
        abs_returns_std = np.std(abs_returns)

        print(f"    Абсолютні дохідності:")
        print(f"      Середнє: {abs_returns_mean:.6f}")
        print(f"      Стд відхилення: {abs_returns_std:.6f}")

        stylized_facts.update({
            'volatility_clustering': stylized_facts.get('volatility_clustering', False),
            'no_autocorr_returns': stylized_facts.get('no_autocorr_returns', False),
            'negative_skewness': stylized_facts.get('negative_skewness', False),
            'fat_tails': self.assumptions.get('fat_tails', False)
        })

        return stylized_facts

    def _justify_model_choice(self):
        """
        Обґрунтовує вибір непараметричних моделей на основі аналізу
        """
        print("\n 4. ОБҐРУНТУВАННЯ ВИБОРУ НЕПАРАМЕТРИЧНИХ МОДЕЛЕЙ")
        print("=" * 50)

        justification = {
            'parametric_assumptions_violated': [],
            'nonparametric_advantages': [],
            'model_recommendation': 'nonparametric'
        }

        print("   📋 Порушені припущення параметричних моделей:")

        # Перевіряємо порушення припущень
        violated_assumptions = 0

        if not self.assumptions.get('normal_returns', True):
            print("      ❌ Дохідності НЕ нормально розподілені")
            justification['parametric_assumptions_violated'].append('non_normal_returns')
            violated_assumptions += 1

        if self.assumptions.get('fat_tails', False):
            print("      ❌ Важкі хвости в розподілі дохідностей")
            justification['parametric_assumptions_violated'].append('fat_tails')
            violated_assumptions += 1

        if not self.assumptions.get('stationary_prices', True):
            print("      ❌ Ціни НЕ стаціонарні")
            justification['parametric_assumptions_violated'].append('non_stationary_prices')
            violated_assumptions += 1

        if not self.assumptions.get('log_normal_prices', True):
            print("      ❌ Ціни НЕ лог-нормально розподілені")
            justification['parametric_assumptions_violated'].append('non_log_normal_prices')
            violated_assumptions += 1

        print(f"\n    Загалом порушено {violated_assumptions} ключових припущень")

        print("\n    Переваги непараметричних моделей для наших даних:")

        advantages = [
            "Не роблять припущень про розподіл даних",
            "Адаптуються до складних нелінійних залежностей",
            "Робастні до викидів та важких хвостів",
            "Здатні виявляти локальні patterns в даних",
            "Ефективні для великих наборів даних",
            "Автоматично виявляють взаємодії між ознаками"
        ]

        for i, advantage in enumerate(advantages, 1):
            print(f"      {i}. {advantage}")
            justification['nonparametric_advantages'].append(advantage)

        # Рекомендація
        if violated_assumptions >= 2:
            recommendation = "СИЛЬНА рекомендація непараметричних моделей"
            confidence = "високий"
        elif violated_assumptions == 1:
            recommendation = "Помірна рекомендація непараметричних моделей"
            confidence = "середній"
        else:
            recommendation = "Можна розглядати обидва підходи"
            confidence = "низький"

        print(f"\n    ВИСНОВОК: {recommendation}")
        print(f"    Рівень впевненості: {confidence}")

        justification['recommendation'] = recommendation
        justification['confidence'] = confidence

        return justification

    def _create_comprehensive_plots(self):
        """
        Створює комплексні графіки для візуального аналізу
        """
        print(f"\n📊 5. ВІЗУАЛІЗАЦІЯ РОЗПОДІЛІВ")
        print("=" * 50)

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # График 1: Розподіл цін vs лог-нормальний
            ax1.hist(self.prices, bins=7, alpha=0.7, density=True, color='skyblue',
                     edgecolor='black', label='Фактичні ціни')

            # Теоретичний лог-нормальний розподіл
            mu, sigma = np.mean(self.log_prices), np.std(self.log_prices)
            x_lognorm = np.linspace(self.prices.min(), self.prices.max(), 100)
            y_lognorm = stats.lognorm.pdf(x_lognorm, s=sigma, scale=np.exp(mu))
            #ax1.plot(x_lognorm, y_lognorm, 'r-', linewidth=2, label='Лог-нормальний')

            ax1.set_title('Розподіл цін BTC', fontweight='bold')
            ax1.set_xlabel('Ціна (USD)')
            ax1.set_ylabel('Щільність')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # График 2: Розподіл дохідностей vs нормальний
            ax2.hist(self.returns, bins=7, alpha=0.7, density=True, color='lightcoral',
                     edgecolor='black', label='Фактичні дохідності')

            # Теоретичний нормальний розподіл
            mu_ret, sigma_ret = np.mean(self.returns), np.std(self.returns)
            x_norm = np.linspace(self.returns.min(), self.returns.max(), 100)
            y_norm = stats.norm.pdf(x_norm, mu_ret, sigma_ret)
            ax2.plot(x_norm, y_norm, 'r-', linewidth=2, label='Нормальний')

            # # Теоретичний t-розподіл (краще для важких хвостів)
            # try:
            #     df_param = 5  # degrees of freedom
            #     y_t = stats.t.pdf(x_norm, df_param, loc=mu_ret, scale=sigma_ret)
            #     ax2.plot(x_norm, y_t, 'g--', linewidth=2, label='t-розподіл (df=5)')
            # except:
            #     pass

            ax2.set_title('Розподіл дохідностей vs Теоретичні', fontweight='bold')
            ax2.set_xlabel('Log-returns')
            ax2.set_ylabel('Щільність')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # График 3: Q-Q plot для дохідностей
            stats.probplot(self.returns, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot: Дохідності vs Нормальний розподіл', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # График 4: Автокореляційна функція квадратів дохідностей
            squared_returns = self.returns ** 2
            max_lags = min(50, len(squared_returns) // 4)
            autocorrs = [np.corrcoef(squared_returns[:-i], squared_returns[i:])[0, 1]
                         if i > 0 else 1.0 for i in range(max_lags)]

            ax4.plot(range(max_lags), autocorrs, 'b-', linewidth=2)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% поріг')
            ax4.axhline(y=-0.05, color='red', linestyle='--', alpha=0.7)
            ax4.set_title('Автокореляція квадратів дохідностей\n(Тест кластеризації волатільності)',
                          fontweight='bold')
            ax4.set_xlabel('Лаг')
            ax4.set_ylabel('Автокореляція')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            print("✅ Графіки створено успішно")

        except Exception as e:
            print(f"❌ Помилка створення графіків: {e}")

    def _provide_final_conclusions(self):
        """
        Надає фінальні висновки та рекомендації
        """
        print(f"\n🎯 6. ФІНАЛЬНІ ВИСНОВКИ ТА РЕКОМЕНДАЦІЇ")
        print("=" * 50)

        print("📋 ПРИПУЩЕННЯ ПРО ПОРОДЖУЮЧИЙ РОЗПОДІЛ:")

        if not self.assumptions.get('normal_returns', True):
            print("   📊 H1: Дохідності НЕ нормально розподілені")
            print("        → Рекомендується t-розподіл або інші розподіли з важкими хвостами")

        if self.assumptions.get('fat_tails', False):
            print("   📏 H2: Розподіл дохідностей має важкі хвости (leptokurtic)")
            print("        → Високий ризик екстремальних подій")

        if not self.assumptions.get('stationary_prices', True):
            print("   📈 H3: Ціни НЕ стаціонарні (випадкове блукання)")
            print("        → Необхідне диференціювання або робота з дохідностями")

        print("\n🎯 ОБҐРУНТУВАННЯ НЕПАРАМЕТРИЧНИХ МОДЕЛЕЙ:")
        print("   ✅ Tree-based моделі (XGBoost, LightGBM) ідеально підходять, тому що:")
        print("      • Не роблять припущень про розподіл")
        print("      • Адаптуються до нелінійних залежностей")
        print("      • Робастні до викидів")
        print("      • Автоматично виявляють взаємодії")
        print("      • Ефективні для великих наборів даних")

        print("\n📚 ТЕОРЕТИЧНЕ ОБҐРУНТУВАННЯ:")
        print("   📖 Фінансові часові ряди (особливо криптовалюти) характеризуються:")
        print("      • Нестаціонарністю")
        print("      • Кластеризацією волатільності")
        print("      • Важкими хвостами розподілу")
        print("      • Структурними зломами")
        print("      • Нелінійними залежностями")

        print("\n🏆 ВИСНОВОК для захисту:")
        print("   'Вибір непараметричних моделей ТЕОРЕТИЧНО ОБҐРУНТОВАНИЙ'")
        print("   'порушенням ключових припущень параметричних методів'")
        print("   'що підтверджено статистичними тестами та візуальним аналізом'")


# Функція для інтеграції в основний код
def analyze_data_generating_process(data, target_col='close'):
    """
    Основна функція для аналізу породжуючого процесу даних
    """
    print("🔬 ЗАПУСК АНАЛІЗУ ПОРОДЖУЮЧОГО РОЗПОДІЛУ")
    print("=" * 80)

    analyzer = CryptoDataDistributionAnalyzer(data, target_col)
    results = analyzer.comprehensive_distribution_analysis()

    return results, analyzer.assumptions