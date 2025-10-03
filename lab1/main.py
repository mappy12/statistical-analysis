import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm

print(".............#1.1.............", "\n")


a = -1
sigma = 2
n = 125

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


plt.figure(figsize=(14,8))

for intervals_count in range(2, 11):
    plt.hist(data, bins=intervals_count, density=True, alpha=0.5,
             label=f"{intervals_count} интервала")

plt.title("Гистограммы относительных частот (2-10)")
plt.xlabel("Значения СВ X")
plt.ylabel("Относительная частота")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()


plt.figure(figsize=(14,8))

for intervals_count in range(15, 26):
    plt.hist(data, bins=intervals_count, density=True, alpha=0.5,
             label=f"{intervals_count} интервала")

plt.title("Гистограммы относительных частот (15-25)")
plt.xlabel("Значения СВ X")
plt.ylabel("Относительная частота")
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()


print("\n.............#2.2.............\n")


print("Гистограмма абсолютных частот")

plt.figure(figsize=(14,8))

plt.hist(data, bins=intervals_num, color='yellow', edgecolor='black',
         alpha=0.5)

plt.title(f"Гистограмма абсолютных частот. {intervals_num} интервалов, согласно правилу Ф-Д")
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

plt.title(f"Гистограмма относительных частот и теоретическая кривая")
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
