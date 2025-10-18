import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from scipy.stats import norm
from scipy.stats import skew, kurtosis

print(".............#1.1.............", "\n")


a = -1
sigma = 2
n = 125

np.random.seed(60)
data = np.random.normal(a, sigma, n)

iqr = np.percentile(data, 75) - np.percentile(data, 25)
interval_width = 2 * iqr / (n ** (1/3))
intervals_num = int((np.max(data) - np.min(data)) / interval_width)

abs_freq, interval_edges = np.histogram(data, bins=intervals_num)

print("Интервальный ряд абсолютных частот:", "\n")
for i in range(len(abs_freq)):
    left_bound = interval_edges[i]
    right_bound = interval_edges[i + 1]
    freq = abs_freq[i]

    print(f"[{left_bound:.4f}, {right_bound:.4f}): {freq}")

print(f"\nЧисло интервалов по правилу Фридмана-Диакониса: {intervals_num}")


print("\n.............#1.2.............\n")


sum_abs_freq = np.sum(abs_freq)
print(f"Сумма абсолютных частот: {sum_abs_freq}")


print("\n.............#1.3.............\n")


rel_freq = abs_freq / n

print("Интервальный ряд относительных частот:", "\n")
for i in range(len(rel_freq)):
    left_bound = interval_edges[i]
    right_bound = interval_edges[i + 1]
    freq = rel_freq[i]

    print(f"[{left_bound:.4f}, {right_bound:.4f}): {freq}")


print("\n.............#1.4.............\n")


sum_rel_freq = np.sum(rel_freq)
print(f"Сумма относительных частот: {sum_rel_freq}")


print("\n.............#2.1.............\n")

bins_range1 = list(range(2, 11))
bins_range2 = list(range(15,26))
plt.figure(figsize=(14,8))

for i, bins in enumerate(bins_range1):
    plt.subplot(3, 3, i+1)
    plt.hist(data, bins=bins, density=True, alpha=0.5,
             label=f"{bins} интервала")
    plt.xlabel("Значения СВ X")
    plt.ylabel("Относительная частота")
    plt.legend()
    plt.grid(True, alpha=0.3)
plt.tight_layout(pad=3.0)


plt.show()


plt.figure(figsize=(14,8))

for i, bins in enumerate(bins_range2):
    plt.subplot(3, 4, i+1)
    plt.hist(data, bins=bins, density=True, alpha=0.5,
             label=f"{bins} интервала")
    plt.xlabel("Значения СВ X")
    plt.ylabel("Относительная частота")
    plt.legend()
    plt.grid(True, alpha=0.3)
plt.tight_layout(pad=3.0)


plt.show()


print("\n.............#2.2.............\n")


print("Гистограмма абсолютных частот")

plt.figure(figsize=(14,8))

plt.hist(data, bins=intervals_num, color='yellow', edgecolor='black',
         alpha=0.5)

plt.title(f"[2.2] Гистограмма абсолютных частот. {intervals_num} интервалов, согласно правилу Ф-Д")
plt.xlabel("Значения СВ X")
plt.ylabel("Абсолютная частота")
plt.grid(True, alpha=0.3)
plt.show()


print("\n.............#2.3.............\n")


print("Гистограмма относительных частот с теоретической кривой")

plt.figure(figsize=(14,8))

plt.hist(data, bins=intervals_num, density=True, color='orange',
         edgecolor='black', alpha=0.5, label='Относительные частоты')

x = np.linspace(np.min(data), np.max(data), 1000)

pdf = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - a)/sigma)**2)

plt.plot(x, pdf, 'p-', linewidth=2, label='Теоретическая плотность')

plt.title(f"[2.3] Гистограмма относительных частот и теоретическая кривая")
plt.xlabel("Значения СВ X")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("\n.............#2.4.............\n")


print("Эмпирическая и теоретическая функции распределения")

plt.figure(figsize=(14, 8))

counts, bins = np.histogram(data, bins=intervals_num)

cum_freq = np.cumsum(counts) / n

plt.step(bins[:-1], cum_freq, where='post',
         label='Эмпирическая функция распределения', linewidth=2)

x = np.linspace(np.min(data), np.max(data), 1000)
cdf = norm.cdf(x, a, sigma)
plt.plot(x, cdf, 'r-', label='Теоретическая функция распределения N(-1, 4)',
         linewidth=2)

plt.title("Эмпирическая и теоретическая функции распределения")
plt.xlabel("Значения СВ X")
plt.ylabel("Вероятность")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("\n.............#2.4.............\n")


plt.figure(figsize=(14, 8))

sns.boxplot(data=data)
plt.title('Бокс-плот распределения')
plt.show()

q1, q2, q3 = np.percentile(data, [25, 50, 75])
iqr = q3 - q1
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr

print(f"Первый квартиль (Q1): {q1:.4f}")
print(f"Медиана (Q2): {q2:.4f}")
print(f"Третий квартиль (Q3): {q3:.4f}")
print(f"Интерквартильный размах (IQR): {iqr:.4f}")
print(f"Нижняя граница усов: {lower_whisker:.4f}")
print(f"Верхняя граница усов: {upper_whisker:.4f}")

print("\n(D) Теоретически ожидаемые границы бокс-плота для N(-1, 2):")

q_25 = norm.ppf(0.25)  # -0.6745
q_75 = norm.ppf(0.75)

q1_theor = a + sigma * q_25
q3_theor = a + sigma * q_75
iqr_theor = q3_theor - q1_theor

lower_whisker_theor = q1_theor - 1.5 * iqr_theor
upper_whisker_theor = q3_theor + 1.5 * iqr_theor

print(f"Теоретический Q1: {q1_theor:.4f}")
print(f"Теоретический Q3: {q3_theor:.4f}")
print(f"Теоретический IQR: {iqr_theor:.4f}")
print(f"Теоретическая нижняя граница усов: {lower_whisker_theor:.4f}")
print(f"Теоретическая верхняя граница усов: {upper_whisker_theor:.4f}")

print("\n.............#3.1.............\n")


print("Точечные оценки параметров распределения:")
print("\nСпособ 1: Вручную через формулы\n")

mean_manual = np.sum(data) / n

sorted_data = np.sort(data)

median = sorted_data[n//2]

values, counts = np.unique(data, return_counts=True)
moda = values[np.argmax(counts)]

dispersion = np.sum((data - mean_manual)**2) / n

corrected_dispersion = np.sum((data - mean_manual)**2) / (n - 1)

std_manual = np.sqrt(dispersion)

std_corrected_manual = np.sqrt(corrected_dispersion)

asymmetry_coefficient = (np.sum((data - mean_manual)**3) / n) / (std_manual**3)

excess = (np.sum((data - mean_manual)**4) / n) / (std_manual**4) - 3

print(f"Среднее: {mean_manual:.6f}")
print(f"Медиана: {median:.6f}")
print(f"Мода: {moda:.6f}")
print(f"Дисперсия: {dispersion:.6f}")
print(f"Исправленная дисперсия: {corrected_dispersion:.6f}")
print(f"Стандартное отклонение: {std_manual:.6f}")
print(f"Исправленное стандартное отклонение: {std_corrected_manual:.6f}")
print(f"Коэффициент асимметрии: {asymmetry_coefficient:.6f}")
print(f"Эксцесс: {excess:.6f}")

print("\nСпособ 2: Встроенные функции\n")

mean_2 = np.mean(data)
median_2 = np.median(data)
moda_2 = stats.mode(data)[0]
dispersion_2 = np.var(data)
corrected_dispersion_2 = np.var(data, ddof=1)
std_manual_2 = np.std(data)
std_corrected_2 = np.std(data, ddof=1)
asymmetry_coefficient_2 = skew(data)
excess_2 = kurtosis(data, fisher=True)

print(f"Среднее: {mean_2:.6f}2")
print(f"Медиана: {median_2:.6f}")
print(f"Мода: {moda_2:.6f}")
print(f"Дисперсия: {dispersion_2:.6f}")
print(f"Исправленная дисперсия: {corrected_dispersion_2:.6f}")
print(f"Стандартное отклонение: {std_manual_2:.6f}")
print(f"Исправленное стандартное отклонение: {std_corrected_2:.6f}")
print(f"Коэффициент асимметрии: {asymmetry_coefficient_2:.6f}")
print(f"Эксцесс: {excess_2:.6f}")


print("\n.............#3.2.............\n")


print("Увеличение объема выборки в 60 раз")


n_large = n * 60
data_large = np.random.normal(a, sigma, n_large)


print("Оценки для большой выборки (n = 7500):")

mean_large = np.mean(data_large)
median_large = np.median(data_large)
values_large, counts_large = np.unique(data_large, return_counts=True)
moda_large = values_large[np.argmax(counts_large)]
dispersion_large = np.var(data_large)
corrected_dispersion_large = np.var(data_large, ddof=1)
std_large = np.std(data_large)
std_corrected_large = np.std(data_large, ddof=1)
asymmetry_coefficient_large = skew(data_large)
excess_large = kurtosis(data_large, fisher=True)

print(f"Среднее: {mean_large:.6f}2")
print(f"Медиана: {median_large:.6f}")
print(f"Мода: {moda_large:.6f}")
print(f"Дисперсия: {dispersion_large:.6f}")
print(f"Исправленная дисперсия: {corrected_dispersion_large:.6f}")
print(f"Стандартное отклонение: {std_large:.6f}")
print(f"Исправленное стандартное отклонение: {std_corrected_large:.6f}")
print(f"Коэффициент асимметрии: {asymmetry_coefficient_large:.6f}")
print(f"Эксцесс: {excess_large:.6f}")