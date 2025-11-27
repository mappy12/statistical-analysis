import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import shapiro, anderson, normaltest, chi2, norm, chisquare, t
import seaborn as sns
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

penguins = sns.load_dataset('penguins')

species = 'Chinstrap'
gender = 'Female'
variable = 'bill_depth_mm'
alpha = 0.035

data = penguins[
    (penguins['species'] == species) &
    (penguins['sex'] == gender)
][variable].dropna().reset_index(drop=True)

print(f"Объем выборки: {len(data)}")
print(f"Первые 5 значений:\n{data.head()}")

print("\n...............#1...............\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle(f'Визуальный анализ распределения {variable}\n({species}, {gender})', fontsize=14)

# 1.1 Гистограмма с кривой плотности и нормальным распределением
axes[0, 0].hist(data, bins=10, density=True, alpha=0.7, color='skyblue', edgecolor='black')
data.plot(kind='density', ax=axes[0, 0], color='red', linewidth=2)

# Нормальное распределение с параметрами выборки
xmin, xmax = axes[0, 0].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, data.mean(), data.std())
axes[0, 0].plot(x, p, 'k--', linewidth=1.5, label='Нормальное распр.')
axes[0, 0].set_title('Гистограмма и кривые плотности')
axes[0, 0].set_xlabel(variable)
axes[0, 0].set_ylabel('Плотность')
axes[0, 0].legend()

# 1.2 Box-plot
axes[0, 1].boxplot(data, vert=True)
axes[0, 1].set_title('Box-plot')
axes[0, 1].set_ylabel(variable)

# 1.3 Q-Q plot
from scipy.stats import probplot
probplot(data, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q plot')

# 1.4 Эмпирическая функция распределения
axes[1, 1].ecdf(data, label='Эмпирическая ФР')
# Теоретическая нормальная ФР
x = np.linspace(data.min(), data.max(), 100)
y = norm.cdf(x, data.mean(), data.std())
axes[1, 1].plot(x, y, 'r--', label='Нормальная ФР')
axes[1, 1].set_title('Функции распределения')
axes[1, 1].set_xlabel(variable)
axes[1, 1].set_ylabel('F(x)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("\n...............#2...............\n")

stats_dict = {
    'Объем выборки (n)': len(data),
    'Среднее': data.mean(),
    'Стандартное отклонение': data.std(),
    'Минимум': data.min(),
    'Максимум': data.max(),
    'Q1 (25%)': data.quantile(0.25),
    'Медиана (Q2, 50%)': data.median(),
    'Q3 (75%)': data.quantile(0.75),
    'Асимметрия (skewness)': data.skew(),
    'Эксцесс (kurtosis)': data.kurtosis()
}

print("\n=== ОПИСАТЕЛЬНАЯ СТАТИСТИКА ===")
for key, value in stats_dict.items():
    print(f"{key}: {value:.4f}")

print("\n...............#3...............\n")

print(f"Переменная: {variable}")
print(f"Вид: {species}, Пол: {gender}")
print(f"Уровень значимости α = {alpha}")
print(f"Объем выборки: n = {len(data)}")
print()

# 3.1 Критерий Шапиро-Уилка
print("1. КРИТЕРИЙ ШАПИРО-УИЛКА")
print("   H₀: Распределение является нормальным")
print("   H₁: Распределение не является нормальным")

stat_sw, p_sw = shapiro(data)
print(f"   Статистика: W = {stat_sw:.4f}")
print(f"   p-value: {p_sw:.4f}")

if p_sw < alpha:
    print(f"   ▶ ВЫВОД: ОТВЕРГАЕМ H₀ (p-value < α={alpha})")
    print("   Распределение НЕ является нормальным")
else:
    print(f"   ▶ ВЫВОД: НЕ ОТВЕРГАЕМ H₀ (p-value ≥ α={alpha})")
    print("   Нет оснований отвергать нормальность распределения")
print()

# 3.2 Критерий Андерсона-Дарлинга
print("2. КРИТЕРИЙ АНДЕРСОНА-ДАРЛИНГА")
print("   H₀: Распределение является нормальным с заданными параметрами")
print("   H₁: Распределение не является нормальным")

result_ad = anderson(data, dist='norm')
print(f"   Статистика: A² = {result_ad.statistic:.4f}")

# Критические значения
print("   Критические значения:")
for i in range(len(result_ad.significance_level)):
    print(f"   {result_ad.significance_level[i]*100}%: {result_ad.critical_values[i]:.4f}")

# Интерполяция для α=0.035
critical_value_035 = np.interp(alpha,
                              [0.15, 0.10, 0.05, 0.025, 0.01],
                              result_ad.critical_values)
print(f"   Критическое значение для α={alpha}: {critical_value_035:.4f}")

if result_ad.statistic > critical_value_035:
    print(f"   ▶ ВЫВОД: ОТВЕРГАЕМ H₀ (A² > критического значения)")
    print("   Распределение НЕ является нормальным")
else:
    print(f"   ▶ ВЫВОД: НЕ ОТВЕРГАЕМ H₀ (A² ≤ критического значения)")
    print("   Нет оснований отвергать нормальность распределения")
print()

# 3.3 Критерий Д'Агостино-Пирсона
print("3. КРИТЕРИЙ Д'АГОСТИНО-ПИРСОНА")
print("   H₀: Распределение является нормальным (на основе асимметрии и эксцесса)")
print("   H₁: Распределение не является нормальным")

stat_ag, p_ag = normaltest(data)
print(f"   Статистика: K² = {stat_ag:.4f}")
print(f"   p-value: {p_ag:.4f}")

if p_ag < alpha:
    print(f"   ▶ ВЫВОД: ОТВЕРГАЕМ H₀ (p-value < α={alpha})")
    print("   Распределение НЕ является нормальным")
else:
    print(f"   ▶ ВЫВОД: НЕ ОТВЕРГАЕМ H₀ (p-value ≥ α={alpha})")
    print("   Нет оснований отвергать нормальность распределения")
print()

# 3.4 Критерий Хи-квадрат
print("4. КРИТЕРИЙ ХИ-КВАДРАТ")
print("   H₀: Распределение соответствует нормальному")
print("   H₁: Распределение не соответствует нормальному")

# Группируем данные в интервалы (используем правило Стерджеса)
n_bins = max(3, int(1 + 3.322 * np.log10(len(data))))  # минимум 3 интервала
f_obs, bins = np.histogram(data, bins=n_bins)

print(f"   Количество интервалов: {n_bins}")

# Ожидаемые частоты для нормального распределения
mu, sigma = data.mean(), data.std()
f_exp = []
for i in range(len(bins) - 1):
    prob = norm.cdf(bins[i + 1], mu, sigma) - norm.cdf(bins[i], mu, sigma)
    f_exp.append(prob * len(data))

# НОРМАЛИЗУЕМ ожидаемые частоты, чтобы их сумма совпадала с суммой наблюдаемых
f_exp = np.array(f_exp)
f_obs_sum = np.sum(f_obs)
f_exp_sum = np.sum(f_exp)
f_exp = f_exp * (f_obs_sum / f_exp_sum)  # нормализация

print(f"   Сумма наблюдаемых частот: {f_obs_sum}")
print(f"   Сумма ожидаемых частот после нормализации: {np.sum(f_exp):.2f}")

# Проверяем, что все ожидаемые частоты >= 5 (условие применимости)
if np.any(f_exp < 5):
    print("   ⚠ Внимание: некоторые ожидаемые частоты < 5")
    print("   Результат может быть ненадежным")
    # Объединяем соседние интервалы с малыми частотами
    f_obs_combined = []
    f_exp_combined = []
    i = 0
    while i < len(f_obs):
        if f_exp[i] < 5 and i < len(f_obs) - 1:
            # Объединяем с соседним интервалом
            f_obs_combined.append(f_obs[i] + f_obs[i + 1])
            f_exp_combined.append(f_exp[i] + f_exp[i + 1])
            i += 2
        else:
            f_obs_combined.append(f_obs[i])
            f_exp_combined.append(f_exp[i])
            i += 1

    f_obs = np.array(f_obs_combined)
    f_exp = np.array(f_exp_combined)
    print(f"   После объединения интервалов: {len(f_obs)} интервалов")

# Критерий хи-квадрат
chi2_stat, chi2_p = chisquare(f_obs, f_exp)
print(f"   Статистика: χ² = {chi2_stat:.4f}")
print(f"   p-value: {chi2_p:.4f}")

if chi2_p < alpha:
    print(f"   ▶ ВЫВОД: ОТВЕРГАЕМ H₀ (p-value < α={alpha})")
    print("   Распределение НЕ соответствует нормальному")
else:
    print(f"   ▶ ВЫВОД: НЕ ОТВЕРГАЕМ H₀ (p-value ≥ α={alpha})")
    print("   Нет оснований отвергать соответствие нормальному распределению")
print()

ssl._create_default_https_context = ssl._create_unverified_context

# Загрузка данных
penguins = sns.load_dataset('penguins')

# Параметры варианта 6
species = 'Chinstrap'
gender = 'Female'
variable = 'bill_depth_mm'
alpha = 0.035

# Фильтрация данных для определения объема выборки
data = penguins[
    (penguins['species'] == species) &
    (penguins['sex'] == gender)
    ][variable].dropna()

n = len(data)  # объем выборки

print("\n...............#4...............\n")

print(f"Критерий K1: критерий Д'Агостино-Пирсона")
print(f"Альтернатива H1: t-распределение Стьюдента с k=25")
print(f"Уровень значимости α: {alpha}")
print(f"Объем выборки n: {n}")
print()

# Параметры моделирования
M = 1000  # количество экспериментов

# Оценка мощности критерия против альтернативы H1
rejections = 0

for i in range(M):
    # Генерируем выборку из t-распределения (альтернатива H1)
    sample = t.rvs(df=25, size=n)

    # Применяем критерий Д'Агостино-Пирсона
    _, p_value = normaltest(sample)

    # Считаем отвержения H₀
    if p_value < alpha:
        rejections += 1

power = rejections / M

print("РЕЗУЛЬТАТЫ:")
print(f"Количество экспериментов M: {M}")
print(f"Количество отвержений H₀: {rejections}")
print(f"Оценка мощности: {power:.4f}")
print()

# Интерпретация
print("ВЫВОД:")
print(f"Мощность критерия Д'Агостино-Пирсона против")
print(f"t-распределения с 25 степенями свободы составляет {power:.3f}")

if power > 0.8:
    print("Критерий обладает высокой мощностью против данной альтернативы")
elif power > 0.5:
    print("Критерий обладает умеренной мощностью против данной альтернативы")
else:
    print("Критерий обладает низкой мощностью против данной альтернативы")