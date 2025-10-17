import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

a = -1
sigma_2 = 4
sigma = np.sqrt(sigma_2)
n = 8
N = 190
M = 650

np.random.seed(53)


print("/////////////////ЧАСТЬ 0/////////////////", "\n")


print(".............#0.1.............", "\n")

print("Моделирование 190 выборок из 8 независимых наблюдений", "\n")

data = np.random.normal(a, sigma, size=(N,n))

print("\n.............#0.2.............", "\n")

print("Вычисление выборочных средних по каждой выборке", "\n")

sample_means = np.mean(data, axis=1)

print("\n.............#0.3.............", "\n")

print("Построение кумуляты относительных частот", "\n")

plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
sorted_means = np.sort(sample_means)
empirical_cdf = np.arange(1, len(sorted_means) + 1) / len(sorted_means)
plt.step(sorted_means, empirical_cdf, where='post', label='Эмпирическая ФР',
         color='red')

x_range = np.linspace(a - 3*sigma/np.sqrt(n), a + 3*sigma/np.sqrt(n), 1000)
theoretical_cdf = stats.norm.cdf(x_range, a, sigma/np.sqrt(n))
plt.plot(x_range, theoretical_cdf, 'b-', linewidth=2,
         label='Теоретическая ФР', alpha=0.7)

plt.xlabel('Выборочное среднее X̄')
plt.ylabel('Вероятность P(X̄ ≤ x)')
plt.title('3. Кумулята и теоретическая ФР\nдля выборочного среднего')
plt.legend()
plt.grid(True, alpha=0.3)

print("\n.............#0.4.............", "\n")

print("Построение гистограммы относительных частот", "\n")


plt.subplot(1, 2, 2)

plt.hist(sample_means, bins=15, density=True, alpha=0.7,
         color='lightblue', label='Гистограмма X̄', edgecolor='black')

theoretical_pdf = stats.norm.pdf(x_range, a, sigma/np.sqrt(n))

plt.plot(x_range, theoretical_pdf, 'r-', linewidth=2,
         label='Теоретическая плотность X̄')

plt.xlabel('Выборочное среднее X̄')
plt.ylabel('Плотность вероятности')
plt.title('4. Гистограмма и теоретическая плотность\nдля выборочного среднего')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n.............#0.5.............", "\n")

print("Нахождение выборочных средней, медианы и дисперсии", "\n")

mean_X = np.mean(sample_means)
median_X = np.median(sample_means)
var_X = np.var(sample_means, ddof=1)

print(f"Выборочное среднее для X̄: {mean_X:.4f}")
print(f"Медиана для X̄: {median_X:.4f}")
print(f"Выборочная дисперсия для X̄: {var_X:.4f}")

print("\n.............#0.6.............", "\n")
print("Нахождение выборочных коэффициентов асимметрии и эксцесса", "\n")

skewness = stats.skew(sample_means)
kurtosis = stats.kurtosis(sample_means)

print(f"Коэффициент асимметрии: {skewness:.4f}")
print(f"Коэффициент эксцесса: {kurtosis:.4f}")

print(f"\nИнтерпретация:")
if abs(skewness) < 0.1:
    print("Асимметрия ≈ 0: распределение симметрично")
elif skewness > 0:
    print("Асимметрия > 0: правосторонняя асимметрия (длинный правый хвост)")
else:
    print("Асимметрия < 0: левосторонняя асимметрия (длинный левый хвост)")

if abs(kurtosis) < 0.1:
    print("Эксцесс ≈ 0: распределение близко к нормальному")
elif kurtosis > 0:
    print("Эксцесс > 0: более острый пик и тяжелые хвосты")
else:
    print("Эксцесс < 0: более плоский пик и легкие хвосты")


print("\n.............#0.7.............", "\n")
print("Вычисление, сколько раз значения выборочной средней X̄ оказались больше"
      " параметра 'a'", "\n")

count_greater_than_a = np.sum(sample_means > a)

percentage = count_greater_than_a / N * 100

print(f"Количество случаев, когда X̄ > a: {count_greater_than_a}")
print(f"Общее количество выборок: {N}")
print(f"Процент случаев: {percentage:.1f}%")
