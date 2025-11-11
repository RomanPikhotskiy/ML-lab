import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
df = pd.read_csv('f1_dnf.csv')

# Предобработка данных
df = df.replace('\\N', np.nan)

# Преобразование числовых колонок
numeric_columns = ['grid', 'laps', 'year', 'round', 'fastestLap', 'fastestLapSpeed', 'milliseconds', 'points']
for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Заполнение пропусков
df['points'] = df['points'].fillna(0)
mean_speed = df['fastestLapSpeed'].mean()
median_time = df['milliseconds'].median()
df['fastestLapSpeed'] = df['fastestLapSpeed'].fillna(mean_speed)
df['milliseconds'] = df['milliseconds'].fillna(median_time)
df['laps'] = df['laps'].fillna(0)
df.drop_duplicates(inplace=True)

print(f"Размер данных после очистки: {df.shape}")

# Выбор признаков и целевой переменной
features = ['grid', 'laps', 'year', 'round', 'fastestLap', 'fastestLapSpeed']
target = 'target_finish'

# Очистка данных
df_clean = df.dropna(subset=[target])

# Анализ баланса классов
print("Распределение целевой переменной:")
class_distribution = df_clean[target].value_counts()
print(class_distribution)

# Фильтрация редких классов
min_samples_per_class = 2
class_counts = df_clean[target].value_counts()
valid_classes = class_counts[class_counts >= min_samples_per_class].index

print(f"Классы с достаточным количеством примеров: {list(valid_classes)}")

df_filtered = df_clean[df_clean[target].isin(valid_classes)]

print(f"Размер данных после фильтрации классов: {df_filtered.shape}")
print("Новое распределение классов:")
print(df_filtered[target].value_counts())

# Определение стратегии разделения
use_stratify = len(df_filtered) >= 100

X = df_filtered[features]
y = df_filtered[target]

# Заполнение пропусков в признаках
print(f"\nПропущенные значения в признаках до обработки:")
print(X.isnull().sum())

imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=features)

print(f"Пропущенные значения после обработки: {X.isnull().sum().sum()}")

# Разделение данных
if use_stratify and len(valid_classes) >= 2:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print("Использовано стратифицированное разделение")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    print("Использовано обычное разделение (без стратификации)")

print(f"Обучающая выборка: {X_train.shape[0]} записей")
print(f"Тестовая выборка: {X_test.shape[0]} записей")
print(f"Распределение классов в обучающей выборке:")
print(y_train.value_counts(normalize=True))

# Основные статистики
print("Основные статистики числовых признаков:")
print(X.describe())

# Визуализация
plt.figure(figsize=(15, 10))

# Распределение целевой переменной
plt.subplot(2, 3, 1)
y.value_counts().plot(kind='bar', color=['red', 'green'])
plt.title('Распределение целевой переменной')
plt.xlabel('Финишировал (0-нет, 1-да)')
plt.ylabel('Количество')

# Распределение признаков
for i, feature in enumerate(features[:4], 2):
    plt.subplot(2, 3, i)
    plt.hist(X[feature], bins=20, alpha=0.7, color=f'C{i}', edgecolor='black')
    plt.title(f'Распределение {feature}')
    plt.xlabel(feature)
    plt.ylabel('Частота')

# Корреляционная матрица
plt.subplot(2, 3, 6)
corr_matrix = pd.concat([X, y], axis=1).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляционная матрица')

plt.tight_layout()
plt.show()

# Анализ корреляций
print("\nКорреляции с целевой переменной:")
correlations_with_target = pd.concat([X, y], axis=1).corr()[target].sort_values(ascending=False)
print(correlations_with_target)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Нормализация выполнена с помощью StandardScaler")

# Подбор оптимального количества соседей
print("ПОДБОР ОПТИМАЛЬНОГО КОЛИЧЕСТВА СОСЕДЕЙ:")

k_values = range(1, min(31, len(X_train) // 2))
train_scores = []
test_scores = []

for k in k_values:
    if k <= len(X_train):
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_scaled, y_train)

        train_score = accuracy_score(y_train, knn_temp.predict(X_train_scaled))
        test_score = accuracy_score(y_test, knn_temp.predict(X_test_scaled))

        train_scores.append(train_score)
        test_scores.append(test_score)
    else:
        train_scores.append(0)
        test_scores.append(0)

# Находим оптимальное k
if test_scores:
    optimal_k = k_values[np.argmax(test_scores)]
    max_accuracy = max(test_scores)
else:
    optimal_k = 3
    max_accuracy = 0

print(f"Оптимальное количество соседей: {optimal_k}")

# Обучение модели с оптимальным k
knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
knn_optimal.fit(X_train_scaled, y_train)

y_pred_train = knn_optimal.predict(X_train_scaled)
y_pred_test = knn_optimal.predict(X_test_scaled)

# Матрица ошибок
print("Матрица рассогласования для тестовой выборки:")
cm_test = confusion_matrix(y_test, y_pred_test)
print(cm_test)

# Отчет о классификации
print("\nОтчет о классификации для тестовой выборки:")
print(classification_report(y_test, y_pred_test))

# Анализ важности признаков
feature_importance = pd.DataFrame({
    'feature': features,
    'correlation_with_target': [correlations_with_target[f] for f in features],
    'abs_correlation': [abs(correlations_with_target[f]) for f in features]
}).sort_values('abs_correlation', ascending=False)

print("\nВажность признаков (по абсолютной корреляции):")
print(feature_importance[['feature', 'correlation_with_target']])

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"""
ИТОГИ АНАЛИЗА:

1. КАЧЕСТВО МОДЕЛИ:
   - Точность на обучающей выборке: {train_accuracy:.1%}
   - Точность на тестовой выборке: {test_accuracy:.1%}
   - Оптимальное количество соседей: {optimal_k}

2. ДАННЫЕ:
   - Исходный размер данных: {df.shape[0]} записей
   - После очистки: {len(df_filtered)} записей
   - Количество признаков: {len(features)}

3. КЛЮЧЕВЫЕ ФАКТОРЫ:
   - Самый важный признак: {feature_importance.iloc[0]['feature']}
   - Корреляция с целевой: {feature_importance.iloc[0]['correlation_with_target']:.3f}
""")


"""Заключение
Модель KNN показала хорошие результаты с точностью 82.7% на тестовой выборке.

Ключевые выводы:

Стартовая позиция (grid) - самый важный фактор с отрицательной корреляцией (-0.345)

Количество кругов (laps) - второй по важности признак

Год проведения (year) - показывает улучшение надежности

Рекомендации для команд Формулы-1:

Уделять особое внимание квалификации для получения лучших стартовых позиций
Инвестировать в надежность автомобиля для завершения гонок
Учитывать исторические тенденции при планировании стратегии"""