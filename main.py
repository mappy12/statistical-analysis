import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, bartlett, levene, ttest_1samp, ttest_ind, ttest_rel, f_oneway, pearsonr, tukey_hsd
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10

print("ЛАБОРАТОРНАЯ РАБОТА №5")
print("Вариант 6 - Дальневосточный федеральный округ")
print("Уровень значимости α = 0.035\n")

# =============================================================================
# ЧАСТЬ 1
# =============================================================================

print("="*60)
print("ЧАСТЬ 1")
print("="*60)

# 1. Загрузка данных
print("\n1. ЗАГРУЗКА ДАННЫХ")
df = pd.read_excel('CHISLO_DOCTORS.xlsx', sheet_name='MyList')

# Извлекаем данные по регионам ДФО (строки 2–12, 11 регионов)
years = ['2005', '2010', '2015', '2019', '2020', '2021']
region_names = df.iloc[1:13, 0].tolist()
regions_data = df.iloc[1:13, 1:7].values
df_regions = pd.DataFrame(regions_data, columns=years, index=region_names)

print(f"Количество регионов ДФО: {len(region_names)}")
print(f"Годы анализа: {', '.join(years)}")


print("\nТаблица для сравнения с Excel файлом:")
print("="*80)
print(f"{'Регион':<35}", end="")
for year in years:
    print(f"{year:>8}", end="")
print()
print("-"*80)

for idx, region in enumerate(df_regions.index):
    print(f"{region:<35}", end="")
    for year in years:
        print(f"{df_regions.loc[region, year]:>8.1f}", end="")
    print()

print("-"*80)
print(f"{'Среднее по ДФО':<35}", end="")
for year in years:
    print(f"{df_regions[year].mean():>8.1f}", end="")
print()
print("="*80)

# 2. Визуализация
print("\n2. ВИЗУАЛИЗАЦИЯ ДАННЫХ")
fig, axes = plt.subplots(2, 2, figsize=(15,12))
fig.suptitle('Анализ данных по Дальневосточному федеральному округу', fontsize=16, fontweight='bold')

# Боксплот по годам
data_box = [df_regions[year] for year in years]
axes[0,0].boxplot(data_box, labels=years)
axes[0,0].set_title('Распределение показателя X по годам')
axes[0,0].set_ylabel('Число врачей на 10 тыс. населения')
axes[0,0].grid(True, alpha=0.3)

# Динамика по регионам (первые 8 для наглядности)
for region in df_regions.index[:8]:
    axes[0,1].plot(years, df_regions.loc[region], marker='o', label=region)
axes[0,1].set_title('Динамика показателя X по регионам ДФО')
axes[0,1].set_ylabel('Число врачей на 10 тыс. населения')
axes[0,1].legend(bbox_to_anchor=(1.05,1), loc='upper left')
axes[0,1].grid(True, alpha=0.3)

# Среднее по ДФО
dfo_mean = df_regions.mean()
axes[1,0].plot(years, dfo_mean, marker='o', color='blue', label='Среднее по ДФО', linewidth=2)
axes[1,0].set_title('Среднее значение показателя X по ДФО')
axes[1,0].set_ylabel('Число врачей на 10 тыс. населения')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Гистограмма распределения за 2021 год
axes[1,1].hist(df_regions['2021'], bins=8, alpha=0.7, edgecolor='black')
axes[1,1].axvline(df_regions['2021'].mean(), color='red', linestyle='--', label=f'Среднее: {df_regions["2021"].mean():.1f}')
axes[1,1].set_title('Распределение показателя X в 2021 году')
axes[1,1].set_xlabel('Число врачей на 10 тыс. населения')
axes[1,1].set_ylabel('Количество регионов')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Описательная статистика
print("\n3. ОПИСАТЕЛЬНАЯ СТАТИСТИКА")
desc_stats = pd.DataFrame()
for year in years:
    desc_stats[year] = [
        df_regions[year].mean(),
        df_regions[year].std(),
        df_regions[year].quantile(0.25),
        df_regions[year].median(),
        df_regions[year].quantile(0.75),
        df_regions[year].min(),
        df_regions[year].max()
    ]
desc_stats.index = ['Среднее', 'Стд. отклонение', 'Q1', 'Медиана', 'Q3', 'Min', 'Max']
print(desc_stats.round(2))

# 4. Проверка на нормальность
print("\n4. ПРОВЕРКА НА НОРМАЛЬНОСТЬ (Тест Шапиро-Уилка)")
alpha = 0.035
normality_results = []
for year in years:
    stat, p_value = shapiro(df_regions[year])
    normality_results.append({
        'Год': year,
        'Статистика': stat,
        'p-value': p_value,
        'Нормальное': p_value > alpha
    })
normal_df = pd.DataFrame(normality_results)
print(normal_df.round(4))

normal_years = normal_df[normal_df['Нормальное']]['Год'].tolist()
print(f"\nГоды с нормальным распределением: {normal_years}")

# 5. Проверка равенства дисперсий
print("\n5. ПРОВЕРКА РАВЕНСТВА ДИСПЕРСИЙ")
bart_stat, bart_p = bartlett(*[df_regions[year] for year in normal_years])
lev_stat, lev_p = levene(*[df_regions[year] for year in normal_years])
print(f"Тест Бартлетта: p-value = {bart_p:.4f}")
print(f"Тест Левена: p-value = {lev_p:.4f}")
equal_var = bart_p > alpha and lev_p > alpha
print(f"Дисперсии равны: {equal_var}")

# 6. Сравнение средних между годами
print("\n6. СРАВНЕНИЕ СРЕДНИХ МЕЖДУ ГОДАМИ (t-тест для независимых выборок)")
year_pairs = [('2005','2021'), ('2010','2015'), ('2015','2021')]
pair_results = []
for y1, y2 in year_pairs:
    if y1 in normal_years and y2 in normal_years:
        t_stat, p_val = ttest_ind(df_regions[y1], df_regions[y2], equal_var=equal_var)
        pair_results.append({
            'Сравнение': f"{y1} vs {y2}",
            'p-value': p_val,
            'Значимо': p_val<alpha,
            'Различие': f"{df_regions[y1].mean():.1f} vs {df_regions[y2].mean():.1f}"
        })
pair_df = pd.DataFrame(pair_results)
print(pair_df.round(4))

# 7. Множественное сравнение средних (ANOVA)
print("\n7. МНОЖЕСТВЕННОЕ СРАВНЕНИЕ СРЕДНИХ")
f_stat, p_anova = f_oneway(*[df_regions[year] for year in normal_years])
print(f"ANOVA тест: F = {f_stat:.4f}, p-value = {p_anova:.4f}")
print(f"Есть значимые различия: {p_anova < alpha}")

# =============================================================================
# ЧАСТЬ 2
# =============================================================================
print("\n" + "="*60)
print("ЧАСТЬ 2")
print("="*60)
print(f"Используем годы с нормальным распределением: {normal_years}")

# 1. Проверка гипотез о равенстве средних (t-test)
print("\n1. t-тест для всех пар")
for i in range(len(normal_years)):
    for j in range(i+1, len(normal_years)):
        y1, y2 = normal_years[i], normal_years[j]
        t_stat, p_val = ttest_ind(df_regions[y1], df_regions[y2], equal_var=equal_var)
        print(f"  {y1} vs {y2}: p-value = {p_val:.4f} {'✓' if p_val<alpha else ''}")

# 2. Проверка корреляции
print("\n2. КОРРЕЛЯЦИЯ МЕЖДУ ВЫБОРКАМИ")
corr_results = []
for i in range(len(normal_years)):
    for j in range(i+1, len(normal_years)):
        y1, y2 = normal_years[i], normal_years[j]
        r, p_val = pearsonr(df_regions[y1], df_regions[y2])
        corr_results.append({'Пары': f"{y1}-{y2}", 'Коэффициент': r, 'p-value': p_val, 'Значимо': p_val<alpha})
corr_df = pd.DataFrame(corr_results)
print(corr_df.round(4))

# 3. Зависимые выборки
print("\n3. t-тест для зависимых выборок")
dep_results = []
for i in range(len(normal_years)):
    for j in range(i+1, len(normal_years)):
        y1, y2 = normal_years[i], normal_years[j]
        t_stat, p_val = ttest_rel(df_regions[y1], df_regions[y2])
        dep_results.append({'Пары': f"{y1}-{y2}", 'p-value': p_val, 'Значимо': p_val<alpha})
dep_df = pd.DataFrame(dep_results)
print(dep_df.round(4))

# 4. Проверка значимости отличий средних в группе
print("\n4. ANOVA повторно")
f_stat, p_anova = f_oneway(*[df_regions[year] for year in normal_years])
print(f"ANOVA тест: F = {f_stat:.4f}, p-value = {p_anova:.4f}")
print(f"Статистически значимые различия между годами: {p_anova < alpha}")

# Визуализация корреляции
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df_regions[normal_years].corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
ax.set_title('Матрица корреляции между годами (ДФО)')
plt.tight_layout()
plt.show()
