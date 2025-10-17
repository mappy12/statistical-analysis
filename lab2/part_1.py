import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

a = -1
sigma_2 = 4
sigma = np.sqrt(sigma_2)
n = 8
N = 190
M = 650

np.random.seed(52)

print("/////////////////ЧАСТЬ I/////////////////", "\n")


print(".............#1.1.............", "\n")

print("Моделирование 190 выборок из 8 наблюдений", "\n")

data = np.random.normal(a, sigma, size=(N, n))

print(f"Размер данных: {data.shape}")
print(f"Первая выборка: {data[0]}")


print("\n.............#1.2.............", "\n")

print("Вычисление выборочной дисперсии Dв для каждой выборки", "\n")

sample_variances = np.var(data, axis=1)

print(f"Количество значений Dв: {len(sample_variances)}")
print(f"Первые 5 значений Dв: {sample_variances[:5]}")


print("\n.............#1.3.............", "\n")
print("Нахождение выборочных средней, медианы и дисперсии для Dв", "\n")

mean_Dv = np.mean(sample_variances)
median_Dv = np.median(sample_variances)
var_Dv = np.var(sample_variances, ddof=1)

print(f"Выборочное среднее для Dв: {mean_Dv:.4f}")
print(f"Медиана для Dв: {median_Dv:.4f}")
print(f"Выборочная дисперсия для Dв: {var_Dv:.4f}")


print("\n.............#1.4.............", "\n")
print("Построение кумуляты относительных частот для Dв", "\n")

plt.figure(figsize=(15, 8))

sorted_Dv = np.sort(sample_variances)

empirical_cdf_Dv = np.arange(1, len(sorted_Dv) + 1) / len(sorted_Dv)

plt.step(sorted_Dv, empirical_cdf_Dv, where='post',
         label='Эмпирическая ФР', color='red', linewidth=2)

plt.xlabel('Выборочная дисперсия Dв')
plt.ylabel('Вероятность P(Dв ≤ x)')
plt.title('4. Кумулята относительных частот\nдля выборочной дисперсии Dв')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\n.............#1.5.............", "\n")
print("Построение гистограммы относительных частот для Dв", "\n")

plt.figure(figsize=(15, 8))

plt.hist(sample_variances, bins=15, density=True, alpha=0.7,
         color='lightgreen', label='Гистограмма Dв', edgecolor='black')

plt.xlabel('Выборочная дисперсия Dв')
plt.ylabel('Плотность вероятности')
plt.title('5. Гистограмма относительных частот\nдля выборочной дисперсии Dв')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("\n.............#1.6.............", "\n")

count_less_than_sigma2 = np.sum(sample_variances < sigma_2)

print(f"Количество случаев, когда Dв < σ²: {count_less_than_sigma2}")
print(f"Общее количество выборок: {N}")
print(f"Процент случаев: {(count_less_than_sigma2 / N * 100):.1f}%")
print(f"Теоретическое значение σ² = {sigma_2}")


print("\n.............#1.7.............", "\n")
print("Вычисление Y = n * Dв / σ²", "\n")

Y = n * sample_variances / sigma_2

print(f"Количество значений Y: {len(Y)}")
print(f"Первые 10 значений Y: {Y[:10]}")
print(f"Минимальное значение Y: {np.min(Y):.4f}")
print(f"Максимальное значение Y: {np.max(Y):.4f}")
print(f"Среднее значение Y: {np.mean(Y):.4f}")


print("\n.............#1.8.............", "\n")
print("Нахождение выборочных характеристик для Y", "\n")

mean_Y = np.mean(Y)
median_Y = np.median(Y)
var_Y = np.var(Y, ddof=1)
skewness_Y = stats.skew(Y)
kurtosis_Y = stats.kurtosis(Y)

print(f"Выборочное среднее для Y: {mean_Y:.4f}")
print(f"Медиана для Y: {median_Y:.4f}")
print(f"Выборочная дисперсия для Y: {var_Y:.4f}")
print(f"Коэффициент асимметрии Y: {skewness_Y:.4f}")
print(f"Коэффициент эксцесса Y: {kurtosis_Y:.4f}")

print("\n.............#1.9.............", "\n")
print("Вычисление теоретических характеристик для Y", "\n")

df = n - 1
# Объяснение:
# - По теореме Фишера, для нормальной совокупности
# - Y = n * Dв / σ² имеет распределение хи-квадрат
# - Число степеней свободы = n - 1 = 8 - 1 = 7

theoretical_mean_Y = df
theoretical_var_Y = 2 * df
theoretical_median_Y = stats.chi2.median(df)
theoretical_skewness_Y = np.sqrt(8 / df)
theoretical_kurtosis_Y = 12 / df

print(f"Теоретические характеристики для Y")
print(f"Математическое ожидание M[Y] = n - 1 = {theoretical_mean_Y}")
print(f"Дисперсия D[Y] = 2(n - 1) = {theoretical_var_Y}")
print(f"Медиана Med[Y] = {theoretical_median_Y:.4f}")
print(f"Коэффициент асимметрии = √(8/(n-1)) = {theoretical_skewness_Y:.4f}")
print(f"Коэффициент эксцесса = 12/(n-1) = {theoretical_kurtosis_Y:.4f}")


print("\n.............#1.10.............", "\n")
print("Построение гистограммы относительных частот для Y", "\n")

plt.figure(figsize=(15, 8))

plt.hist(Y, bins=15, density=True, alpha=0.7,
         color='orange', label='Гистограмма Y', edgecolor='black')

y_range = np.linspace(0, np.max(Y)*1.2, 1000)
chi2_pdf = stats.chi2.pdf(y_range, df)
plt.plot(y_range, chi2_pdf, 'r-', linewidth=2,
         label=f'Теоретическая плотность χ²({df})')

plt.xlabel('Y = nDв/σ²')
plt.ylabel('Плотность вероятности')
plt.title('10. Гистограмма и теоретическая плотность\nдля случайной величины Y')

plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("\n.............#1.11.............", "\n")
print("Построение кумуляты относительных частот и теоретической ФР для Y", "\n")

plt.figure(figsize=(10, 6))

sorted_Y = np.sort(Y)
empirical_cdf_Y = np.arange(1, len(sorted_Y) + 1) / len(sorted_Y)
plt.step(sorted_Y, empirical_cdf_Y, where='post',
         label='Эмпирическая ФР Y', color='red', linewidth=2)

y_range = np.linspace(0, np.max(Y)*1.2, 1000)
chi2_cdf = stats.chi2.cdf(y_range, df)
plt.plot(y_range, chi2_cdf, 'b-', linewidth=2,
         label=f'Теоретическая ФР χ²({df})', alpha=0.7)

plt.xlabel('Y = nDв/σ²')
plt.ylabel('Вероятность P(Y ≤ x)')
plt.title('11. Кумулята относительных частот и теоретическая ФР\nдля случайной величины Y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()