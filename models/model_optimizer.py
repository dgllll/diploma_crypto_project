# 🛠️ ВИПРАВЛЕННЯ NUMPY СУМІСНОСТІ (ДОДАТИ НА ПОЧАТОК ФАЙЛУ!)
import numpy as np
import warnings
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

# 🔧 Workaround для scikit-optimize з новими версіями numpy
def fix_numpy_compatibility():
    """Відновлює застарілі numpy алиаси для сумісності з scikit-optimize"""
    if not hasattr(np, 'int'):
        np.int = int
    if not hasattr(np, 'float'):
        np.float = float
    if not hasattr(np, 'bool'):
        np.bool = bool
    if not hasattr(np, 'complex'):
        np.complex = complex


# Застосовуємо виправлення ПЕРЕД імпортом scikit-optimize
fix_numpy_compatibility()

# Придушуємо deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.int.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.float.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.int.*")

# Тепер можна безпечно імпортувати
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import randint, uniform

# 🚀 Баєсова оптимізація (тепер без помилок!)
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer

    BAYESIAN_AVAILABLE = True
    print("✅ BayesSearchCV успішно завантажено!")
except ImportError as e:
    print(f"⚠️ BayesSearchCV недоступний: {e}")
    BAYESIAN_AVAILABLE = False


def optimize_xgboost_bayesian(X_train, y_train, cv=5, n_iter=25):
    """
    🎯 Виправлена баєсова оптимізація XGBoost
    """
    print("🎯 Баєсова оптимізація XGBoost (виправлена версія)...")

    if not BAYESIAN_AVAILABLE:
        print("❌ BayesSearchCV недоступний, використовуємо fallback")
        return fallback_xgboost(X_train, y_train)

    # Простір пошуку (менші діапазони для стабільності)
    search_spaces = {
        'n_estimators': Integer(200, 600),  # Зменшений діапазон
        'max_depth': Integer(4, 8),  # Зменшений діапазон
        'learning_rate': Real(0.02, 0.15, prior='log-uniform'),
        'subsample': Real(0.7, 0.95),
        'colsample_bytree': Real(0.7, 0.95),
        'min_child_weight': Integer(1, 6),
        'gamma': Real(0, 2),  # Зменшений діапазон
        'reg_alpha': Real(0, 5),
        'reg_lambda': Real(1, 8)
    }

    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        random_state=42,
        n_jobs=-1
    )

    try:
        bayes_search = BayesSearchCV(
            estimator=xgb_model,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0  # Вимикаємо verbose для чистого виводу
        )

        print(f"🔄 Запуск {n_iter} ітерацій оптимізації...")
        bayes_search.fit(X_train, y_train)

        return bayes_search.best_estimator_

    except Exception as e:
        print(f"❌ Помилка в BayesSearchCV: {e}")
        print("🔧 Перехід на fallback...")
        return fallback_xgboost(X_train, y_train)


def optimize_lightgbm_bayesian(X_train, y_train, cv=5, n_iter=25):
    """
    🎯 Виправлена баєсова оптимізація LightGBM
    """
    print("🎯 Баєсова оптимізація LightGBM (виправлена версія)...")

    if not BAYESIAN_AVAILABLE:
        print("❌ BayesSearchCV недоступний, використовуємо fallback")
        return fallback_lightgbm(X_train, y_train)

    # Простір пошуку для LightGBM (оптимізовані діапазони)
    search_spaces = {
        'n_estimators': Integer(200, 600),
        'num_leaves': Integer(25, 80),  # Оптимальні значення
        'learning_rate': Real(0.02, 0.15, prior='log-uniform'),
        'subsample': Real(0.7, 0.95),
        'colsample_bytree': Real(0.7, 0.95),
        'min_child_samples': Integer(8, 25),  # Оптимальні значення
        'reg_alpha': Real(0, 5),
        'reg_lambda': Real(0, 5)
    }

    lgb_model = lgb.LGBMRegressor(
        objective='regression',
        boosting_type='gbdt',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    try:
        bayes_search = BayesSearchCV(
            estimator=lgb_model,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        print(f"🔄 Запуск {n_iter} ітерацій оптимізації...")
        bayes_search.fit(X_train, y_train)


        return bayes_search.best_estimator_

    except Exception as e:
        print(f"❌ Помилка в BayesSearchCV: {e}")
        print("🔧 Перехід на fallback...")
        return fallback_lightgbm(X_train, y_train)


def optimize_histgb_bayesian(X_train, y_train, cv=5, n_iter=25):
    """
    🎯 Баєсова оптимізація HistGradientBoostingRegressor (scikit-learn)
    """
    print("🎯 Баєсова оптимізація HistGradientBoosting (scikit-learn)...")

    if not BAYESIAN_AVAILABLE:
        print("❌ BayesSearchCV недоступний, використовуємо fallback")
        return fallback_histgb(X_train, y_train)

    # Простір пошуку для HistGradientBoosting
    search_spaces = {
        'max_iter': Integer(100, 500),  # Кількість boosting ітерацій
        'max_depth': Integer(3, 15),  # Максимальна глибина дерев
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),  # Швидкість навчання
        'max_leaf_nodes': Integer(15, 100),  # Максимум листя на дереві
        'min_samples_leaf': Integer(5, 50),  # Мінімум зразків у листі
        'l2_regularization': Real(0, 10),  # L2 регуляризація
        'max_bins': Integer(128, 255),  # Кількість bins для гістограм
        'early_stopping': [True, False],  # Раннє зупинення
        'validation_fraction': Real(0.1, 0.2),  # Частка для валідації
    }

    histgb_model = HistGradientBoostingRegressor(
        random_state=42,
        scoring='loss',  # Використовувати loss для раннього зупинення
        n_iter_no_change=10  # Кількість ітерацій без покращення
    )

    try:
        bayes_search = BayesSearchCV(
            estimator=histgb_model,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        print(f"🔄 Запуск {n_iter} ітерацій оптимізації...")
        bayes_search.fit(X_train, y_train)


        return bayes_search.best_estimator_

    except Exception as e:
        print(f"❌ Помилка в BayesSearchCV: {e}")
        print("🔧 Перехід на fallback...")
        return fallback_histgb(X_train, y_train)


# 🛡️ ПОКРАЩЕНІ FALLBACK функції
def fallback_xgboost(X_train, y_train):
    """Покращена fallback XGBoost модель"""
    print("🔧 Fallback XGBoost з ще кращими параметрами...")
    model = xgb.XGBRegressor(
        n_estimators=500,  # Більше дерев
        max_depth=6,
        learning_rate=0.04,  # Нижча для стабільності
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=1.5,  # Трохи більше регуляризації
        reg_lambda=4,
        random_state=42,
        n_jobs=-1,
        objective='reg:squarederror',
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    print("✅ XGBoost fallback навчено")
    return model


def fallback_lightgbm(X_train, y_train):
    """Покращена fallback LightGBM модель"""
    print("🔧 Fallback LightGBM з ще кращими параметрами...")
    model = lgb.LGBMRegressor(
        n_estimators=400,  # Зменшити з 500
        num_leaves=25,  # Зменшити з 45
        learning_rate=0.025,  # Зменшити з 0.04
        subsample=0.75,  # Зменшити з 0.85
        colsample_bytree=0.75,  # Зменшити з 0.85
        min_child_samples=20,  # Збільшити з 12
        reg_alpha=3.0,  # Збільшити з 1.5
        reg_lambda=3.0,
        random_state=42,
        objective='regression',
        boosting_type='gbdt',
        verbose=-1
    )
    model.fit(X_train, y_train)
    print("✅ LightGBM fallback навчено")
    return model


def fallback_histgb(X_train, y_train):
    """Покращена fallback HistGradientBoosting модель"""
    print("🔧 Fallback HistGradientBoosting з оптимальними параметрами...")

    model = HistGradientBoostingRegressor(
        max_iter=200,  # Зменшити з 300
        max_depth=6,  # Зменшити з 8
        learning_rate=0.04,  # Зменшити з 0.05
        max_leaf_nodes=35,  # Зменшити з 50
        min_samples_leaf=15,  # Збільшити з 10
        l2_regularization=3.0,  # L2 регуляризація
        max_bins=200,  # Хороша деталізація гістограм
        early_stopping=True,  # Раннє зупинення
        validation_fraction=0.15,  # 15% для валідації
        n_iter_no_change=15,  # Терпіння для раннього зупинення
        random_state=42,
        scoring='loss'  # Оптимізація loss функції
    )

    model.fit(X_train, y_train)
    print("✅ HistGradientBoosting fallback навчено")
    return model


# 🚀 ГОЛОВНА ФУНКЦІЯ з покращеною обробкою помилок
def bayesian_optimize_models(X_train, y_train, n_iter=25):
    """
    🎯 Виправлена баєсова оптимізація з HistGradientBoosting замість RandomForest
    """
    print(f"📊 Дані: {len(X_train)} зразків, {X_train.shape[1]} ознак")
    print(f"🔧 Numpy fix: {'✅' if hasattr(np, 'int') else '❌'}")
    print(f"🎯 BayesSearchCV: {'✅' if BAYESIAN_AVAILABLE else '❌'}")

    models = {}

    # XGBoost
    print("\n" + "=" * 50)
    print("🎯 XGBOOST ОПТИМІЗАЦІЯ")
    print("=" * 50)
    try:
        models['xgboost'] = optimize_xgboost_bayesian(X_train, y_train, n_iter=n_iter)
        print("✅ XGBoost готовий")
    except Exception as e:
        print(f"❌ Критична помилка XGBoost: {e}")
        models['xgboost'] = fallback_xgboost(X_train, y_train)

    # LightGBM
    print("\n" + "=" * 50)
    print("🎯 LIGHTGBM ОПТИМІЗАЦІЯ")
    print("=" * 50)
    try:
        models['lightgbm'] = optimize_lightgbm_bayesian(X_train, y_train, n_iter=n_iter)
        print("✅ LightGBM готовий")
    except Exception as e:
        print(f"❌ Критична помилка LightGBM: {e}")
        models['lightgbm'] = fallback_lightgbm(X_train, y_train)

    # 🆕 HistGradientBoosting замість RandomForest
    print("\n" + "=" * 50)
    print("🎯 HISTGRADIENTBOOSTING ОПТИМІЗАЦІЯ")
    print("=" * 50)
    try:
        models['histgb'] = optimize_histgb_bayesian(X_train, y_train, n_iter=n_iter // 2)
        print("✅ HistGradientBoosting готовий")
    except Exception as e:
        print(f"❌ Критична помилка HistGradientBoosting: {e}")
        models['histgb'] = fallback_histgb(X_train, y_train)

    print(f"\n🎉 Оптимізація завершена! Навчано {len(models)} моделей.")
    return models



# ⚡ Інші функції (оновлені)
def quick_bayesian_optimize(X_train, y_train):
    """⚡ Швидка оптимізація"""
    print("⚡ Швидка баєсова оптимізація (10 ітерацій)...")
    return bayesian_optimize_models(X_train, y_train, n_iter=10)


def quick_optimize_models(X_train, y_train):
    """🏃‍♂️ Без оптимізації - тільки fallback моделі"""
    print("🏃‍♂️ Швидке навчання без оптимізації...")

    models = {}
    models['xgboost'] = fallback_xgboost(X_train, y_train)
    models['lightgbm'] = fallback_lightgbm(X_train, y_train)
    models['histgb'] = fallback_histgb(X_train, y_train)
    return models


def get_model_predictions(models, X_test):
    """📊 Отримання прогнозів"""
    print(f"📊 Отримання прогнозів від {len(models)} моделей...")
    predictions = {}

    for name, model in models.items():
        try:
            if model is None:
                print(f"⚠️ Модель {name} пуста")
                continue

            pred = model.predict(X_test)
            predictions[name] = pred
            print(f"✅ {name}: {len(pred)} прогнозів")
        except Exception as e:
            print(f"❌ Помилка {name}: {e}")

    return predictions


# 🔧 LEGACY підтримка (оновлені)
def optimize_xgboost(X_train, y_train, cv=5, n_iter=15):
    """Legacy функція"""
    return optimize_xgboost_bayesian(X_train, y_train, cv, n_iter)


def optimize_lightgbm(X_train, y_train, cv=5, n_iter=15):
    """Legacy функція"""
    return optimize_lightgbm_bayesian(X_train, y_train, cv, n_iter)


def optimize_histgb(X_train, y_train, cv=5, n_iter=10):
    """Legacy функція для Random Forest (НОВИЙ!)"""
    return optimize_histgb_bayesian(X_train, y_train, cv, n_iter)