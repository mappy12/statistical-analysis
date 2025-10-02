import numpy as np


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