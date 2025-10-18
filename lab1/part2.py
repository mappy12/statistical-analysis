import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import math

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

confidence = 0.95
alpha = 1 - confidence
n = - 1/(2*epsilon**2) * math.log(alpha / 2)
min_n = int(np.ceil(n))
interval_y_theor = int(1 + math.log2(min_n))
print(min_n, interval_y_theor)

plt.figure(figsize=(14,8))
plt.hist(Y, bins=interval_y_theor, density=True, color='lightgreen', edgecolor='red')
plt.title(f'Гистограмма {(bins)} интервалов)')
plt.xlabel('Значение СВ')
plt.ylabel('Относительная частота')
plt.grid(True, alpha=0.3)
x = np.linspace(0, 3, 1000)

uniform = stats.uniform(1, 2)

pdf = uniform.pdf(x)

plt.plot(x, pdf, '-r', label='Теоретическая функция плотности вероятности')
plt.xlabel("Y")
plt.ylabel("Плотность вероятности")
plt.title("Нормальное распределение")
plt.legend()

plt.show()



