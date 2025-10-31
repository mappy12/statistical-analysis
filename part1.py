import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

a = -1
sigma2 = 5
sigma = np.sqrt(sigma2)
gamma = 0.90
n = 16
M = 2400
K = 100

np.random.seed(59)
data = np.random.normal(a, sigma, n)

print("............#1.1............\n")

print("СПОСОБ 1 - Аналитический:")

p = (1 + gamma) / 2
t_critical = stats.norm.ppf(p)

margin = t_critical * sigma / np.sqrt(n)
sample_mean = np.mean(data)

ci_analytical = (sample_mean - margin, sample_mean + margin)

print(f"Критическое значение t = {t_critical:.4f}")
print(f"Выборочное среднее = {sample_mean:.4f}")
print(f"Полуширина интервала: {margin:.4f}")
print(f"Доверительный интервал: ({ci_analytical[0]:.4f}, {ci_analytical[1]:.4f})")

ci_scipy = stats.norm.interval(gamma, loc=sample_mean, scale=sigma/np.sqrt(n))

print("\nСПОСОБ 2 - scipy.stats.interval:")
print(f"Доверительный интервал: ({ci_scipy[0]:.4f}, {ci_scipy[1]:.4f})\n")

print("............#1.2............\n")

print("СПОСОБ 1 - Аналитический:")

p = (1 + gamma) / 2
t = stats.t.ppf(p, df=n-1)

sample_mean = np.mean(data)

s2 = np.var(data, ddof=1)  # S² = исправленная дисперсия
s = np.sqrt(s2)

margin = t * s / np.sqrt(n)
ci_analytical = (sample_mean - margin, sample_mean + margin)

print(f"Критическое значение t({n-1}) = {t:.4f}")
print(f"Выборочное среднее = {sample_mean:.4f}")
print(f"Исправленная дисперсия S² = {s2:.4f}")
print(f"Стандартное отклонение S = {s:.4f}")
print(f"Полуширина интервала: {margin:.4f}")
print(f"Доверительный интервал: ({ci_analytical[0]:.4f}, {ci_analytical[1]:.4f})")

print("\nСПОСОБ 2 - scipy.stats.interval:")
ci_scipy = stats.t.interval(gamma, df=n-1, loc=sample_mean, scale=s/np.sqrt(n))
print(f"Доверительный интервал: ({ci_scipy[0]:.4f}, {ci_scipy[1]:.4f})")

print("............#1.3............\n")

alpha = (1 - gamma) / 2
beta = (1 + gamma) / 2

chi2_alpha = stats.chi2.ppf(alpha, df=n-1)
chi2_beta = stats.chi2.ppf(beta, df=n-1)

s2 = np.var(data, ddof=1)

ci_variance = ((n-1) * s2 / chi2_beta, (n-1) * s2 / chi2_alpha)

print(f"Надежность γ = {gamma}")
print(f"α = (1-γ)/2 = {alpha:.4f}")
print(f"β = (1+γ)/2 = {beta:.4f}")
print(f"χ²_α({n-1}) = {chi2_alpha:.4f}")
print(f"χ²_β({n-1}) = {chi2_beta:.4f}")
print(f"Исправленная выборочная дисперсия S² = {s2:.4f}")
print(f"Доверительный интервал для σ²: ({ci_variance[0]:.4f}, {ci_variance[1]:.4f})")

print(f"Длина интервала: {ci_variance[1] - ci_variance[0]:.4f}")

covers_true_variance = ci_variance[0] <= sigma2 <= ci_variance[1]
print(f"ДИ покрывает истинное значение σ² = {sigma2}: {covers_true_variance}")

print("............#2............\n")

gamma_range = np.linspace(0.5, 0.99, 50)

lengths_mean_known = []
lengths_mean_unknown = []
lengths_variance = []

for g in gamma_range:
    # Для мат. ожидания при известной дисперсии
    z_quantile = stats.norm.ppf((1 + g) / 2)
    length_mean_known = 2 * z_quantile * sigma / np.sqrt(n)
    lengths_mean_known.append(length_mean_known)

    # Для мат. ожидания при неизвестной дисперсии
    t_quantile = stats.t.ppf((1 + g) / 2, df=n - 1)
    length_mean_unknown = 2 * t_quantile * s / np.sqrt(n)
    lengths_mean_unknown.append(length_mean_unknown)

    # Для дисперсии
    alpha_var = (1 - g) / 2
    beta_var = (1 + g) / 2
    chi2_alpha = stats.chi2.ppf(alpha_var, df=n - 1)
    chi2_beta = stats.chi2.ppf(beta_var, df=n - 1)
    length_var = (n - 1) * s2 * (1 / chi2_alpha - 1 / chi2_beta)
    lengths_variance.append(length_var)

plt.figure(figsize=(12, 8))

# График для математического ожидания
plt.subplot(2, 1, 1)
plt.plot(gamma_range, lengths_mean_known, 'b-', linewidth=2, label='Мат. ожидание (известная дисперсия)')
plt.plot(gamma_range, lengths_mean_unknown, 'r-', linewidth=2, label='Мат. ожидание (неизвестная дисперсия)')
plt.xlabel('Надежность γ')
plt.ylabel('Длина доверительного интервала')
plt.title('Зависимость длины ДИ от надежности - Математическое ожидание')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=gamma, color='green', linestyle='--', alpha=0.7, label=f'γ = {gamma}')

# График для дисперсии
plt.subplot(2, 1, 2)
plt.plot(gamma_range, lengths_variance, 'g-', linewidth=2)
plt.xlabel('Надежность γ')
plt.ylabel('Длина доверительного интервала')
plt.title('Зависимость длины ДИ от надежности - Дисперсия')
plt.grid(True, alpha=0.3)
plt.axvline(x=gamma, color='green', linestyle='--', alpha=0.7, label=f'γ = {gamma}')
plt.legend()

plt.tight_layout()
plt.show()

print("............#3............\n")

n_range = np.arange(5, 101, 5)  # от 5 до 100 с шагом 5

lengths_mean_known_n = []
lengths_mean_unknown_n = []
lengths_variance_n = []  #

for n_val in n_range:
    z_quantile = stats.norm.ppf((1 + gamma) / 2)
    length_mean_known = 2 * z_quantile * sigma / np.sqrt(n_val)
    lengths_mean_known_n.append(length_mean_known)

    t_quantile = stats.t.ppf((1 + gamma) / 2, df=n_val - 1)

    length_mean_unknown = 2 * t_quantile * sigma / np.sqrt(n_val)
    lengths_mean_unknown_n.append(length_mean_unknown)


    alpha_var = (1 - gamma) / 2
    beta_var = (1 + gamma) / 2
    chi2_alpha = stats.chi2.ppf(alpha_var, df=n_val - 1)
    chi2_beta = stats.chi2.ppf(beta_var, df=n_val - 1)

    length_var = (n_val - 1) * sigma2 * (1 / chi2_alpha - 1 / chi2_beta)
    lengths_variance_n.append(length_var)

plt.figure(figsize=(12, 8))

# График для математического ожидания
plt.subplot(2, 1, 1)
plt.plot(n_range, lengths_mean_known_n, 'b-', linewidth=2, label='Мат. ожидание (известная дисперсия)')
plt.plot(n_range, lengths_mean_unknown_n, 'r-', linewidth=2, label='Мат. ожидание (неизвестная дисперсия)')
plt.xlabel('Объем выборки n')
plt.ylabel('Длина доверительного интервала')
plt.title('Зависимость длины ДИ от объема выборки - Математическое ожидание')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axvline(x=n, color='green', linestyle='--', alpha=0.7, label=f'n = {n}')

# График для дисперсии
plt.subplot(2, 1, 2)
plt.plot(n_range, lengths_variance_n, 'g-', linewidth=2)
plt.xlabel('Объем выборки n')
plt.ylabel('Длина доверительного интервала')
plt.title('Зависимость длины ДИ от объема выборки - Дисперсия')
plt.grid(True, alpha=0.3)
plt.axvline(x=n, color='green', linestyle='--', alpha=0.7, label=f'n = {n}')
plt.legend()

plt.tight_layout()
plt.show()

print("............#4............\n")

M = 2400
cover_count = 0
confidence_intervals = []

print(f"Моделируем {M} выборок объема n = {n}...")

for i in range(M):
    sample_m = np.random.normal(a, sigma, n)
    sample_mean = np.mean(sample_m)
    s_m = np.std(sample_m, ddof=1)

    # Строим доверительный интервал
    t_critical = stats.t.ppf((1 + gamma) / 2, df=n - 1)
    margin = t_critical * s_m / np.sqrt(n)
    ci = (sample_mean - margin, sample_mean + margin)
    confidence_intervals.append(ci)

    if ci[0] <= a <= ci[1]:
        cover_count += 1

# Точечная оценка надежности
gamma_star = cover_count / M

print(f"Построено {M} доверительных интервалов")
print("\nПримеры доверительных интервалов (первые 5):")
for i in range(5):
    covers = "ПОКРЫВАЕТ" if confidence_intervals[i][0] <= a <= \
                            confidence_intervals[i][1] else "НЕ ПОКРЫВАЕТ"
    print(
        f"Интервал {i + 1}: ({confidence_intervals[i][0]:.3f}, {confidence_intervals[i][1]:.3f}) - {covers} a = {a}")

print(f"\nТочечная оценка надежности γ* = {gamma_star:.4f}")
print(f"Количество интервалов, покрывающих a = {a}: {cover_count} из {M}")