import numpy as np
import matplotlib.pyplot as plt

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
    print(f"{key}: {value}")

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

# Группировка данных в интервалы (правило Стерджеса)
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

# Сколько раз была отвергнута H₀
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



print("\n\n////////////////////////////ТЕОРИЯ////////////////////////////\n\n")

print("Критерий Шапиро–Уилка (W):")
print("Идея: сравнивает упорядоченные значения выборки с теоретическими значениями нормального распределения.")
print("Формула: W = ( (Σ a_i * x_(n+1−i))² ) / Σ (x_i − mean)²")
print("  где:")
print("    x_i — упорядоченные значения выборки")
print("    x̄ — среднее выборки")
print("    a_i — коэффициенты, зависящие от мат. ожиданий порядковых статистик N(0,1)")
print("Интерпретация:")
print("  • W близко к 1 → распределение похоже на нормальное")
print("  • W сильно меньше 1 → отклонения от нормальности, хвосты или асимметрия")
print("  • Малое W → маленький p-value → отвергаем нормальность")

print("\nКритерий Д’Агостино–Пирсона (K²):")
print("Идея: проверяет нормальность через два момента — асимметрию (g1) и эксцесс (g2).")
print("Формула: K² = Z_g1² + Z_g2², где Z_g1 и Z_g2 — нормализованные g1 и g2")
print("Интерпретация:")
print("  • K² маленькое → g1 и g2 близки к нормальным значениям → распределение нормальное")
print("  • K² большое → сильная асимметрия и/или «тяжёлые» хвосты")
print("  • Большое K² → маленький p-value → отвергаем нормальность")

print("\nКритерий Андерсона–Дарлинга (A²):")
print("Идея: учитывает расхождения по всей распределительной функции, особенно в хвостах.")
print("Формула: A² = −n − (1/n) Σ (2i−1)[ln(F(x_i)) + ln(1−F(x_{n+1−i}))]")
print("  где:")
print("    x_i — упорядоченные значения выборки")
print("    F(x) — функция распределения нормального распределения")
print("    n — размер выборки")
print("  • A² маленькое → выборка хорошо совпадает с F(x) нормального распределения")
print("  • A² большое → большие различия в хвостах")
print("  • Большое A² → критическое A² превышено → отвергаем нормальность")

print("\nКритерий хи-квадрат согласия (χ²):")
print("Идея: сравнивает наблюдаемое распределение с теоретическим по построенным интервалам.")
print("Формула: χ² = Σ ( (O_i − E_i)² / E_i ) , где O_i — наблюдаемые, E_i — ожидаемые частоты")
print("\nКритерий хи-квадрат согласия (χ²):")

print("  • χ² маленькое → наблюдаемое распределение близко к теоретическому")
print("  • χ² большое → частоты сильно отличаются")
print("  • Большое χ² → маленький p-value → отвергаем нормальность")
print("  • Важно: E_i должны быть ≥ 5, иначе критерий работает плохо")
