import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

a = -1
sigma2 = 5
sigma = np.sqrt(sigma2)
gamma = 0.90
n = 16
K = 100
M = 2400

print("............#2.1............\n")

#   1. Смоделировать K выборок из n значений нормально распределенной случайной
#   величины X с параметрами (a,2).  По каждой из K выборок  с надежностью , указанной в
#   Вашем варианте, найти доверительный интервал для дисперсии случайной величины X.

print("\nМоделирование K выборок и построение ДИ для дисперсии...")

gamma_star_var = 0
confidence_intervals_var = []
sample_variances = []

for i in range(K):
    data_k = np.random.normal(a, sigma, n)

    s2_k = np.var(data_k, ddof=1)
    sample_variances.append(s2_k)

    alpha_var = (1 - gamma) / 2
    beta_var = (1 + gamma) / 2
    chi2_alpha = stats.chi2.ppf(alpha_var, df=n - 1)
    chi2_beta = stats.chi2.ppf(beta_var, df=n - 1)

    ci_lower = (n - 1) * s2_k / chi2_beta
    ci_upper = (n - 1) * s2_k / chi2_alpha
    ci_var = (ci_lower, ci_upper)
    confidence_intervals_var.append(ci_var)

    if ci_lower <= sigma2 <= ci_upper:
        gamma_star_var += 1

gamma_star_var_value = gamma_star_var / K

print("\nРЕЗУЛЬТАТЫ ПУНКТА 1:")
print(f"Надежность: γ = {gamma}")
print(f"Количество ДИ, покрывающих σ² = {sigma2}: {gamma_star_var} из {K}")
print(f"Точечная оценка надежности γ* = {gamma_star_var}/{K} = {gamma_star_var_value:.4f}")

print(f"\nПримеры первых 10 доверительных интервалов для дисперсии:")
print("№ | Нижняя граница | Верхняя граница | Покрывает σ²")
print("-" * 50)

for i in range(10):
    covers = "ДА" if confidence_intervals_var[i][0] <= sigma2 <= confidence_intervals_var[i][1] else "НЕТ"
    print(f"{i+1:2} | {confidence_intervals_var[i][0]:13.4f} | {confidence_intervals_var[i][1]:13.4f}   | {covers:11}")

print("....................#2.1....................\n")

#   2. Пункт 1 повторить M раз, сформировав таким образом массив из M  значений точечной
#   оценки  * надежности .

gamma_star_array = []

for j in range(M):
    cover_count_var = 0
    for i in range(K):
        data_k = np.random.normal(a, sigma, n)

        s2_k = np.var(data_k, ddof=1)

        alpha_var = (1 - gamma) / 2
        beta_var = (1 + gamma) / 2
        chi2_alpha = stats.chi2.ppf(alpha_var, df=n - 1)
        chi2_beta = stats.chi2.ppf(beta_var, df=n - 1)

        ci_lower = (n - 1) * s2_k / chi2_beta
        ci_upper = (n - 1) * s2_k / chi2_alpha

        if ci_lower <= sigma2 <= ci_upper:
            cover_count_var += 1

    gamma_star_array.append(cover_count_var / K)

print(f"Завершено! Обработано {M} итераций")

print(f"\nРЕЗУЛЬТАТЫ ПУНКТА 2:")
print(f"Количество повторений: M = {M}")
print(f"Количество выборок в каждом повторении: K = {K}")
print(f"Размер массива γ*: {len(gamma_star_array)}")
print(f"Диапазон значений γ*: от {min(gamma_star_array):.4f} до {max(gamma_star_array):.4f}")
print(f"Истинное значение γ: {gamma}")

# Первые 20 значений для примера
print(f"\nПервые 20 значений γ*:")
for i in range(20):
    print(f"γ*[{i+1:2d}] = {gamma_star_array[i]:.4f}")


print("....................#2.3....................\n")

#   3. Построить гистограмму относительных частот и теоретическую кривую распределения
#   точечной оценки  *, а также ящичковую диаграмму.

print("Построение гистограммы и диаграмм для распределения γ*...")

mean_gamma_star = np.mean(gamma_star_array)
std_gamma_star = np.std(gamma_star_array)

theoretical_mean = gamma
theoretical_std = np.sqrt(gamma * (1 - gamma) / K)

print(f"Параметры распределения γ*:")
print(f"Выборочное среднее: {mean_gamma_star:.4f}")
print(f"Выборочное стандартное отклонение: {std_gamma_star:.4f}")
print(f"Теоретическое среднее: {theoretical_mean:.4f}")
print(f"Теоретическое стандартное отклонение: {theoretical_std:.4f}")

plt.figure(figsize=(15, 5))

# 1. Гистограмма относительных частот с теоретической кривой
plt.subplot(1, 2, 1)
n_bins = 30

# Гистограмма
hist_values, bin_edges, patches = plt.hist(gamma_star_array, bins=n_bins,
                                          density=True, alpha=0.7,
                                          color='lightblue', edgecolor='black',
                                          label='Гистограмма γ*')

# Теоретическая нормальная кривая
x = np.linspace(theoretical_mean - 3*theoretical_std,
                theoretical_mean + 3*theoretical_std, 1000)
y_theoretical = stats.norm.pdf(x, theoretical_mean, theoretical_std)
plt.plot(x, y_theoretical, 'r-', linewidth=2,
         label=f'Теоретическое N({theoretical_mean:.3f}, {theoretical_std:.3f}²)')

plt.xlabel('γ*')
plt.ylabel('Плотность вероятности')
plt.title('Гистограмма и теоретическое распределение γ*')
plt.legend()
plt.grid(True, alpha=0.3)

plt.axvline(x=gamma, color='green', linestyle='--',
            label=f'Истинное γ = {gamma}')
plt.legend()

plt.subplot(1, 2, 2)
box_plot = plt.boxplot(gamma_star_array, patch_artist=True)

box_plot['boxes'][0].set_facecolor('lightyellow')
box_plot['whiskers'][0].set_color('black')
box_plot['whiskers'][1].set_color('black')
box_plot['caps'][0].set_color('black')
box_plot['caps'][1].set_color('black')
box_plot['medians'][0].set_color('red')

plt.ylabel('γ*')
plt.title('Ящичковая диаграмма γ*')
plt.grid(True, alpha=0.3)

plt.plot(1, gamma, 'go', markersize=8, label=f'γ = {gamma}')
plt.legend()

plt.tight_layout()
plt.show()

print(f"\nСтатистика распределения γ*:")
print(f"Минимум: {np.min(gamma_star_array):.4f}")
print(f"Максимум: {np.max(gamma_star_array):.4f}")
print(f"Квартили: Q1 = {np.percentile(gamma_star_array, 25):.4f}, "
      f"Q2 = {np.percentile(gamma_star_array, 50):.4f}, "
      f"Q3 = {np.percentile(gamma_star_array, 75):.4f}")


print("....................#2.4....................\n")
#   4. По полученной в пункте 3 выборке из M значений значений точечной оценки  *найти
#   выборочные числовые характеристики ее распределения: среднее, моду, медиану,
#   исправленную выборочную дисперсию, коэффициенты асимметрии и эксцесса.

print("Вычисление выборочных числовых характеристик распределения γ*...")

mean_val = np.mean(gamma_star_array)
median_val = np.median(gamma_star_array)
variance_val = np.var(gamma_star_array, ddof=1)
std_val = np.std(gamma_star_array, ddof=1)
skewness_val = stats.skew(gamma_star_array)
kurtosis_val = stats.kurtosis(gamma_star_array)

#мода
values, counts = np.unique(np.round(gamma_star_array, 3), return_counts=True)
mode_val = values[np.argmax(counts)]
mode_count = counts[np.argmax(counts)]

print("ВЫБОРОЧНЫЕ ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ:")
print("=" * 50)
print(f"Среднее арифметическое: {mean_val:.6f}")
print(f"Медиана:                {median_val:.6f}")
print(f"Мода:                   {mode_val:.3f} (встречается {mode_count} раз)")
print(f"Исправленная дисперсия: {variance_val:.8f}")
print(f"Стандартное отклонение: {std_val:.6f}")
print(f"Коэффициент асимметрии: {skewness_val:.6f}")
print(f"Коэффициент эксцесса:   {kurtosis_val:.6f}")


print("....................#2.5....................\n")

#   5. Смоделировать M выборок из n значений нормально распределенной случайной
#   величины X с параметрами (a,σ2).

#   По каждой из M выборок найти наблюдаемое значение случайной величины Z (описание
#   случайной величины Z приведено в Вашем варианте)

print("Моделирование M выборок для случайной величины Z...")

Z_values = []

print(f"Моделируем {M} выборок объема n = {n}...")
print(f"Формула для Z: Z = (X̄ - a)√n / S")
print(f"где X̄ - выборочное среднее, S - исправленное стандартное отклонение")

for i in range(M):
    sample = np.random.normal(a, sigma, n)

    sample_mean = np.mean(sample)

    s = np.std(sample, ddof=1)

    # Вычисляем случайную величину Z по формуле варианта 6
    # Z = (X̄ - a)√n / S
    Z = (sample_mean - a) * np.sqrt(n) / s
    Z_values.append(Z)

print(f"\nПервые 15 наблюдаемых значений случайной величины Z:")
print("№  |      Z значение   |  Выборочное среднее X̄ |  Стандартное отклонение S")
print("-" * 80)

for i in range(15):
    sample = np.random.normal(a, sigma, n)
    sample_mean = np.mean(sample)
    s = np.std(sample, ddof=1)
    Z_demo = (sample_mean - a) * np.sqrt(n) / s

    print(f"{i + 1:2} | {Z_demo:15.6f} | {sample_mean:20.6f} | {s:23.6f}")


print("....................#2.6....................\n")

#   6. По полученной в пункте 5 выборке из M значений случайной величины Z найти
#   выборочные числовые характеристики ее распределения: среднее, моду, медиану,
#   исправленную выборочную дисперсию, коэффициенты асимметрии и эксцесса.

print("Вычисление выборочных числовых характеристик распределения Z...")

mean_Z = np.mean(Z_values)
median_Z = np.median(Z_values)
variance_Z = np.var(Z_values, ddof=1)
std_Z = np.std(Z_values, ddof=1)
skewness_Z = stats.skew(Z_values)
kurtosis_Z = stats.kurtosis(Z_values)

#мода
Z_values_rounded = np.round(Z_values, 2)
values_Z, counts_Z = np.unique(Z_values_rounded, return_counts=True)
mode_Z = values_Z[np.argmax(counts_Z)]
mode_count_Z = counts_Z[np.argmax(counts_Z)]

print("ВЫБОРОЧНЫЕ ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ СЛУЧАЙНОЙ ВЕЛИЧИНЫ Z:")
print("=" * 60)
print(f"Среднее арифметическое: {mean_Z:.6f}")
print(f"Медиана:                {median_Z:.6f}")
print(f"Мода:                   {mode_Z:.3f} (встречается {mode_count_Z} раз)")
print(f"Исправленная дисперсия: {variance_Z:.6f}")
print(f"Стандартное отклонение: {std_Z:.6f}")
print(f"Коэффициент асимметрии: {skewness_Z:.6f}")
print(f"Коэффициент эксцесса:   {kurtosis_Z:.6f}")


print("....................#2.7....................\n")

#   7. Построить гистограмму относительных частот и теоретическую кривую распределения
#   случайной величины Z, а также ящичковую диаграмму.

print("Построение гистограммы и диаграмм для распределения Z...")

df = n - 1

plt.figure(figsize=(15, 5))

# 1. Гистограмма относительных частот с теоретической кривой
plt.subplot(1, 2, 1)
n_bins = 30

# Гистограмма
hist_values, bin_edges, patches = plt.hist(Z_values, bins=n_bins,
                                          density=True, alpha=0.7,
                                          color='lightgreen', edgecolor='black',
                                          label='Гистограмма Z')

# Теоретическая кривая t-распределения
x = np.linspace(stats.t.ppf(0.001, df), stats.t.ppf(0.999, df), 1000)
y_theoretical = stats.t.pdf(x, df)
plt.plot(x, y_theoretical, 'r-', linewidth=2,
         label=f't-распределение ({df} степ. своб.)')

plt.xlabel('Z')
plt.ylabel('Плотность вероятности')
plt.title('Гистограмма и теоретическое распределение Z')
plt.legend()
plt.grid(True, alpha=0.3)

plt.axvline(x=0, color='green', linestyle='--',
            label='Теоретическое среднее E[Z] = 0')
plt.legend()

plt.subplot(1, 2, 2)
box_plot = plt.boxplot(Z_values, patch_artist=True)

box_plot['boxes'][0].set_facecolor('lightyellow')
box_plot['whiskers'][0].set_color('black')
box_plot['whiskers'][1].set_color('black')
box_plot['caps'][0].set_color('black')
box_plot['caps'][1].set_color('black')
box_plot['medians'][0].set_color('red')

plt.ylabel('Z')
plt.title('Ящичковая диаграмма Z')
plt.grid(True, alpha=0.3)

# Добавляем точку для теоретического среднего
plt.plot(1, 0, 'go', markersize=8, label='E[Z] = 0')
plt.legend()

plt.tight_layout()
plt.show()


print("....................ОТВЕТЫ НА ВОПРОСЫ ЧАСТИ II....................\n")


# 1) Каков закон распределения точечной оценки γ*?
print("1) Каков закон распределения точечной оценки γ*?")
print("ОТВЕТ: Точечная оценка γ* имеет биномиальное распределение:")
print("       γ* ~ Binomial(K, γ)/K")
print("       При больших K (K > 30) аппроксимируется нормальным распределением:")
print(f"       γ* ≈ N(γ, γ(1-γ)/K) = N({gamma}, {gamma*(1-gamma)/K:.6f})")


# 2) Чему равны математическое ожидание и дисперсия точечной оценки γ*?
print("\n2) Чему равны математическое ожидание и дисперсия точечной оценки γ*?")
E_gamma_star = gamma


# Var[γ*] = Var[количество успехов / K] = (1/K²) × Var[количество успехов]
# Дисперсия количества успехов в биномиальном распределении рассчитывается
# по формуле \(D(X)=npq\)

Var_gamma_star = gamma * (1 - gamma) / K
print(f"ОТВЕТ: Математическое ожидание: E[γ*] = γ = {E_gamma_star}")
print(f"       Дисперсия: Var[γ*] = γ(1-γ)/K = {gamma}×{1-gamma}/{K} = {Var_gamma_star:.6f}")


# 3) Каков закон распределения случайной величины Z?
print("\n3) Каков закон распределения случайной величины Z?")
print(f"ОТВЕТ: Случайная величина Z = (X̄ - a)√n/S имеет")
print(f"       t-распределение Стьюдента с {n-1} степенями свободы:")
print(f"       Z ~ t({n-1})")


# 4) Чему равны математическое ожидание и дисперсия случайной величины Z?
print("\n4) Чему равны математическое ожидание и дисперсия случайной величины Z?")
E_Z_theoretical = 0  # Для t-распределения с ν > 1
if n > 3:
    Var_Z_theoretical = (n - 1) / (n - 3)
else:
    Var_Z_theoretical = float('inf')  # Дисперсия не определена

print(f"ОТВЕТ: Математическое ожидание: E[Z] = 0 (при n > 2)")
if n > 3:
    print(f"       Дисперсия: Var[Z] = (n-1)/(n-3) = {n-1}/{n-3} = {Var_Z_theoretical:.4f} (при n > 3)")
else:
    print(f"       Дисперсия: Var[Z] = ∞ (при n ≤ 3)")


# 5) Согласуются ли выборочные числовые характеристики распределения случайной
# величины Z с ее теоретически ожидаемыми числовыми характеристиками?
print("\n5) Согласуются ли выборочные числовые характеристики распределения")
print("   случайной величины Z с ее теоретически ожидаемыми числовыми характеристиками?")

# Вычисляем выборочные характеристики для сравнения
E_Z_empirical = np.mean(Z_values)
Var_Z_empirical = np.var(Z_values, ddof=1)

print(f"ОТВЕТ: Сравнение теоретических и выборочных характеристик:")
print(f"       Математическое ожидание:")
print(f"       - Теоретическое: E[Z] = 0")
print(f"       - Выборочное:    {E_Z_empirical:.4f}")
print(f"       - Отклонение:    {E_Z_empirical:+.4f}")

if n > 3:
    print(f"       Дисперсия:")
    print(f"       - Теоретическая: Var[Z] = {Var_Z_theoretical:.4f}")
    print(f"       - Выборочная:    {Var_Z_empirical:.4f}")
    print(f"       - Отклонение:    {Var_Z_empirical - Var_Z_theoretical:+.4f}")

# Проверка согласия
E_Z_error = abs(E_Z_empirical)
if n > 3:
    Var_Z_error = abs(Var_Z_empirical - Var_Z_theoretical) / Var_Z_theoretical

print(f"\n       ВЫВОД: ")
if E_Z_error < 0.1:
    print(f"       • Математическое ожидание согласуется с теоретическим"
          f" (отклонение {E_Z_empirical:+.4f})")
else:
    print(f"       • Математическое ожидание не согласуется с теоретическим"
          f" (отклонение {E_Z_empirical:+.4f})")

if n > 3:
    if Var_Z_error < 0.1:
        print(f"       • Дисперсия согласуется с теоретической"
              f" (относительная ошибка {Var_Z_error*100:.1f}%)")
    else:
        print(f"       • Дисперсия не согласуется с теоретической"
              f" (относительная ошибка {Var_Z_error*100:.1f}%)")