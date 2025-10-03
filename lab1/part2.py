import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

n = 125
a, b = 1, 2
epsilon = 0.015


print("..........#1..........")


np.random.seed(70)
Y = np.random.uniform(a, b, n)


print("..........#2..........")


plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(Y, bins=10, density=True, alpha=0.7,
                           color='lightblue',
                           edgecolor='black',
                           label='Гистограмма')

x = np.linspace(a, b, 100)
plt.plot(x, stats.uniform.pdf(x, a, b - a), 'r-', lw=2, label='Теоретическая плотность')
plt.title('Гистограмма и теоретическая плотность распределения Y')
plt.xlabel('Y')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True)
plt.show()


print("..........#3..........")


plt.figure(figsize=(8, 5))
sns.boxplot(Y)
plt.title('Бокс-плот распределения Y')
plt.grid(True)
plt.show()

q1, q3 = np.percentile(Y, [25, 75])
iqr = q3 - q1
lower_whisker = q1 - 1.5 * iqr
upper_whisker = q3 + 1.5 * iqr
outliers = Y[(Y < lower_whisker) | (Y > upper_whisker)]

print(f"Q1: {q1:.3f}, Q3: {q3:.3f}, IQR: {iqr:.3f}")
print(f"Нижняя граница усов: {lower_whisker:.3f}, Верхняя граница усов: {upper_whisker:.3f}")
print(f"Выбросы: {outliers}")

print("Теоретически ожидаемое число выбросов: ~0, тк все значения [1,2]")


print("..........#4..........")


df = pd.DataFrame(Y, columns=['Y'])
desc = df.describe()


mean_manual = desc.loc['mean','Y']
median = desc.loc['50%','Y']
values, counts = np.unique(Y, return_counts=True)
moda = values[np.argmax(counts)]

std_manual = desc.loc['std','Y']
dispersion = std_manual**2
corrected_dispersion = dispersion * n / (n - 1)
std_corrected_manual = np.sqrt(corrected_dispersion)

asymmetry_coefficient = stats.skew(Y)
excess = stats.kurtosis(Y)

print("Точечные оценки параметров распределения случайной величины Y:")
print()

print("Выборочные среднее, медиана и мода:")
print(f"Среднее: {mean_manual:.4f}")
print(f"Медиана: {median:.4f}")
print(f"Мода: {moda:.4f}")
print()

print("Выборочные дисперсия и исправленная дисперсия:")
print(f"Дисперсия: {dispersion:.4f}")
print(f"Исправленная дисперсия: {corrected_dispersion:.4f}")
print()

print("Выборочные стандартное отклонение и исправленное стандартное отклонение:")
print(f"Стандартное отклонение: {std_manual:.4f}")
print(f"Исправленное стандартное отклонение: {std_corrected_manual:.4f}")
print()

print("Выборочные коэффициент асимметрии и эксцесс:")
print(f"Коэффициент асимметрии: {asymmetry_coefficient:.4f}")
print(f"Эксцесс: {excess:.4f}")


print("..........#5..........")


def max_deviation(n, a, b):
    sample = np.random.uniform(a, b, n)
    ecdf = np.sort(sample)
    theoretical_cdf = stats.uniform.cdf(ecdf, a, b - a)
    empirical_cdf = np.arange(1, n + 1) / n
    return np.max(np.abs(empirical_cdf - theoretical_cdf))


print("Подбор объема выборки для точности ε =", epsilon)

n_new = 125  # начинаем с исходного объема
max_iterations = 10000  # ограничение на число итераций
found = False

for i in range(max_iterations):
    dev = max_deviation(n_new, a, b)

    if i % 100 == 0:  # Выводим прогресс каждые 100 итераций
        print(f"n = {n_new}, отклонение = {dev:.6f}")

    if dev <= epsilon:
        found = True
        break
    n_new += 1

if found:
    print(f"\n✅ Требуемый объем выборки для ε={epsilon}: n = {n_new}")
    print(f"Фактическое отклонение: {dev:.6f}")
else:
    print(f"\n❌ Не удалось найти подходящий n за {max_iterations} итераций")
    n_new = 1000  # используем запасной вариант

# Построение гистограммы для найденного n
print(f"\nСтроим гистограмму для n = {n_new}...")

Y_new = np.random.uniform(a, b, n_new)

plt.figure(figsize=(12, 5))

# Гистограмма относительных частот и теоретическая плотность
plt.subplot(1, 2, 1)
counts, bins, patches = plt.hist(Y_new, bins=10, density=True, alpha=0.7,
                                 color='lightblue', edgecolor='black',
                                 label='Гистограмма относительных частот')

# Теоретическая плотность равномерного распределения
x = np.linspace(a - 0.1, b + 0.1, 1000)
plt.plot(x, stats.uniform.pdf(x, a, b - a), 'r-', lw=2,
         label='Теоретическая плотность U(1,2)')

plt.title(
    f'Гистограмма и теоретическая плотность\n(n = {n_new}, ε = {epsilon})')
plt.xlabel('Y')
plt.ylabel('Плотность вероятности')
plt.legend()
plt.grid(True, alpha=0.3)

# График эмпирической и теоретической функций распределения
plt.subplot(1, 2, 2)
sorted_Y = np.sort(Y_new)
empirical_cdf = np.arange(1, len(sorted_Y) + 1) / len(sorted_Y)
theoretical_cdf = stats.uniform.cdf(sorted_Y, a, b - a)

plt.plot(sorted_Y, empirical_cdf, 'b-', lw=2, alpha=0.7,
         label='Эмпирическая ФР')
plt.plot(sorted_Y, theoretical_cdf, 'r-', lw=2, alpha=0.7,
         label='Теоретическая ФР')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=1, color='k', linestyle='-', alpha=0.3)

# Показываем максимальное отклонение
max_dev = np.max(np.abs(empirical_cdf - theoretical_cdf))
max_idx = np.argmax(np.abs(empirical_cdf - theoretical_cdf))
plt.plot([sorted_Y[max_idx], sorted_Y[max_idx]],
         [empirical_cdf[max_idx], theoretical_cdf[max_idx]],
         'g--', lw=2, label=f'Макс. отклонение = {max_dev:.4f}')

plt.title(f'Эмпирическая и теоретическая ФР\n(отклонение ≤ {epsilon})')
plt.xlabel('Y')
plt.ylabel('Функция распределения')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Проверка точности
final_deviation = max_deviation(n_new, a, b)
print(
    f"\nПроверка: максимальное отклонение для n={n_new}: {final_deviation:.6f}")
print(f"Условие выполнено: {final_deviation <= epsilon}")