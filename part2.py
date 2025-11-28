import numpy as np
from scipy.stats import normaltest, shapiro, expon
import matplotlib.pyplot as plt

print("\n...............#1...............\n")

mu, sigma2 = -3, 7  # Y ~ N(-3, 7)
sigma = np.sqrt(sigma2)
alpha = 0.035
M = 7900
n = 80
K2 = "Д'Агостино-Пирсона"

print(f"Параметры нормального распределения: μ={mu}, σ²={sigma2}")
print(f"Уровень значимости: α={alpha}")
print(f"Количество экспериментов: M={M}")
print(f"Объем выборки: n={n}")
print(f"Критерий: {K2}")
print()

false_rejections = 0

for i in range(M):
    # Генерируем выборку из нормального распределения (H₀ верна)
    sample = np.random.normal(loc=mu, scale=sigma, size=n)

    # Применяем критерий Д'Агостино-Пирсона
    _, p_value = normaltest(sample)

    # Считаем ложные отвержения H₀
    if p_value < alpha:
        false_rejections += 1

false_positive_rate = false_rejections / M

print("РЕЗУЛЬТАТЫ:")
print(f"Количество экспериментов: {M}")
print(f"Ложные отвержения H₀: {false_rejections}")
print(f"Оценка ошибки первого рода: {false_positive_rate:.4f}")
print(f"Теоретический уровень значимости: {alpha}")

if abs(false_positive_rate - alpha) < 0.01:
    print("Критерий сохраняет заданный уровень значимости")
else:
    print("Критерий не сохраняет заданный уровень значимости")

print("\n...............#2...............\n")

# Параметры альтернативы H2
scale_params = [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.25, 4.0]
K2_name = "критерий Д'Агостино-Пирсона"
K3_name = "критерий Шапиро-Уилка"

print(f"Критерий K2: {K2_name}")
print(f"Критерий K3: {K3_name}")
print(f"Альтернатива H2: экспоненциальное распределение")
print(f"Математическое ожидание M(X): {scale_params}")
print()

power_K2 = []
power_K3 = []

for scale in scale_params:
    rejections_K2 = 0
    rejections_K3 = 0

    for i in range(M):
        # Генерируем выборку из экспоненциального распределения (H2)
        sample = expon.rvs(scale=scale, size=n)

        # Применяем критерий K2 (Д'Агостино-Пирсона)
        _, p_value_K2 = normaltest(sample)
        if p_value_K2 < alpha:
            rejections_K2 += 1

        # Применяем критерий K3 (Шапиро-Уилка)
        _, p_value_K3 = shapiro(sample)
        if p_value_K3 < alpha:
            rejections_K3 += 1

    power_K2.append(rejections_K2 / M)
    power_K3.append(rejections_K3 / M)

    print(f"M(X) = {scale:4.2f}: мощность K2 = {power_K2[-1]:.3f}, мощность K3 = {power_K3[-1]:.3f}")

print("\nРЕЗУЛЬТАТЫ ОЦЕНКИ МОЩНОСТИ:")
print("M(X)  | Мощность K2 | Мощность K3")
print("-" * 35)
for i, scale in enumerate(scale_params):
    print(f"{scale:4.2f} |    {power_K2[i]:.3f}     |    {power_K3[i]:.3f}")

print("\n...............#3...............\n")

plt.figure(figsize=(10, 6))
plt.plot(scale_params, power_K2, 'bo-', linewidth=2, markersize=6, label="K2: Д'Агостино-Пирсона")
plt.plot(scale_params, power_K3, 'ro-', linewidth=2, markersize=6, label='K3: Шапиро-Уилка')

plt.xlabel('Математическое ожидание M(X) экспоненциального распределения')
plt.ylabel('Мощность критерия')
plt.title('Зависимость мощности критериев от параметра альтернативы H2')
plt.grid(True, alpha=0.3)
plt.legend()

# Добавляем значения на график
for i, scale in enumerate(scale_params):
    plt.annotate(f'{power_K2[i]:.2f}', (scale, power_K2[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
    plt.annotate(f'{power_K3[i]:.2f}', (scale, power_K3[i]), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=8)

plt.tight_layout()
plt.show()


print("\n...............#4...............\n")

# Сравниваем среднюю мощность
mean_power_K2 = np.mean(power_K2)
mean_power_K3 = np.mean(power_K3)

print(f"Средняя мощность критерия K2: {mean_power_K2:.4f}")
print(f"Средняя мощность критерия K3: {mean_power_K3:.4f}")
print()

# Считаем, для каких параметров какой критерий мощнее
K2_better = 0
K3_better = 0
equal = 0

for i in range(len(scale_params)):
    if power_K2[i] > power_K3[i]:
        K2_better += 1
    elif power_K3[i] > power_K2[i]:
        K3_better += 1
    else:
        equal += 1

print("СРАВНЕНИЕ ПО ПАРАМЕТРАМ:")
print(f"Критерий K2 мощнее для {K2_better} параметров из {len(scale_params)}")
print(f"Критерий K3 мощнее для {K3_better} параметров из {len(scale_params)}")
if equal > 0:
    print(f"Мощность одинакова для {equal} параметров")

print("\nВЫВОД:")
if mean_power_K2 > mean_power_K3:
    print("Критерий K2 (Д'Агостино-Пирсона) оказался более мощным")
elif mean_power_K3 > mean_power_K2:
    print("Критерий K3 (Шапиро-Уилка) оказался более мощным")
else:
    print("Критерии имеют примерно одинаковую мощность")

# Дополнительный анализ
if K2_better > K3_better:
    print("Критерий K2 демонстрирует более стабильную высокую мощность")
elif K3_better > K2_better:
    print("Критерий K3 демонстрирует более стабильную высокую мощность")