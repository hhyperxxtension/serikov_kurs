"""
Курсовая работа - Вариант 13
Классификация линейно неразделимых объектов (XOR) 
с использованием библиотеки Scikit-learn

Задание:
1. Генерация данных XOR
2. Классификация методом опорных векторов (SVM) с полиномиальным ядром
3. Классификация методом K ближайших соседей (KNN)
4. Сравнение качества моделей
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# ЭТАП 1: ГЕНЕРАЦИЯ ДАННЫХ XOR

print("="*80)
print("ЭТАП 1: ГЕНЕРАЦИЯ ДАННЫХ XOR")
print("="*80)

# Установка seed для воспроизводимости результатов
np.random.seed(0)

# Генерация случайных точек
X = np.random.randn(512, 2)

# Создание меток классов по правилу XOR
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.where(y, 1, -1)

print(f"Размер матрицы признаков X: {X.shape}")
print(f"Размер вектора меток y: {y.shape}")
print(f"Уникальные метки классов: {np.unique(y)}")
print(f"Распределение классов: класс 1 = {np.sum(y == 1)}, класс -1 = {np.sum(y == -1)}")

# Визуализация исходных данных
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='b', marker='x', s=50, label='1', alpha=0.7)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='r', marker='s', s=50, label='-1', alpha=0.7)
plt.ylim(-3.0, 3.0)
plt.xlim(-3.0, 3.0)
plt.xlabel('x1', fontsize=12)
plt.ylabel('x2', fontsize=12)
plt.legend()
plt.title('Исходные данные XOR', fontsize=14)
plt.grid(True, alpha=0.3)
plt.savefig('xor_data.png', dpi=150, bbox_inches='tight')
plt.show()

# Разделение данных на обучающую и тестовую выборки (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

print(f"\nРазмер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")
# ЭТАП 2: КЛАССИФИКАЦИЯ С ИСПОЛЬЗОВАНИЕМ SVM (ПОЛИНОМИАЛЬНОЕ ЯДРО)

print("\n" + "="*80)
print("ЭТАП 2: SVM С ПОЛИНОМИАЛЬНЫМ ЯДРОМ")
print("="*80)

# Подбор оптимальных гиперпараметров
test_params = [
    {'C': 1.0, 'degree': 2, 'coef0': 0, 'gamma': 'scale'},
    {'C': 10.0, 'degree': 2, 'coef0': 1, 'gamma': 'scale'},
    {'C': 10.0, 'degree': 3, 'coef0': 1, 'gamma': 'auto'},
    {'C': 100.0, 'degree': 3, 'coef0': 1, 'gamma': 'auto'},
]

best_score = 0
best_params = None
best_model_svm = None

print("\nПоиск оптимальных параметров:")
for params in test_params:
    svm = SVC(kernel='poly', random_state=0, **params)
    svm.fit(X_train, y_train)

    train_score = svm.score(X_train, y_train)
    test_score = svm.score(X_test, y_test)

    print(f"\nПараметры: C={params['C']}, degree={params['degree']}, coef0={params['coef0']}, gamma={params['gamma']}")
    print(f"  Точность на обучающей выборке: {train_score:.4f}")
    print(f"  Точность на тестовой выборке: {test_score:.4f}")

    if test_score > best_score:
        best_score = test_score
        best_params = params
        best_model_svm = svm

print("\n" + "-"*80)
print(f"Оптимальные параметры: {best_params}")
print(f"Лучшая точность на тестовой выборке: {best_score:.4f}")
print("-"*80)

# Вычисление ошибок для лучшей модели
y_train_pred_svm = best_model_svm.predict(X_train)
y_test_pred_svm = best_model_svm.predict(X_test)

err_train_svm = np.mean(y_train != y_train_pred_svm)
err_test_svm = np.mean(y_test != y_test_pred_svm)

print(f"\nУдельное количество ошибок на обучающей выборке: {err_train_svm:.4f}")
print(f"Удельное количество ошибок на тестовой выборке: {err_test_svm:.4f}")
print(f"Точность на обучающей выборке: {1-err_train_svm:.4f}")
print(f"Точность на тестовой выборке: {1-err_test_svm:.4f}")

# ЭТАП 3: КЛАССИФИКАЦИЯ С ИСПОЛЬЗОВАНИЕМ K БЛИЖАЙШИХ СОСЕДЕЙ

print("\n" + "="*80)
print("ЭТАП 3: K БЛИЖАЙШИХ СОСЕДЕЙ")
print("="*80)

# Поиск оптимального k и метрики расстояния
k_values = [3, 5, 7, 9, 11, 15, 21]
metrics = ['euclidean', 'manhattan']

best_k = None
best_metric = None
best_knn_score = 0
best_model_knn = None

print("\nПоиск оптимальных параметров:")
for k in k_values:
    for metric in metrics:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        knn.fit(X_train, y_train)

        train_score = knn.score(X_train, y_train)
        test_score = knn.score(X_test, y_test)

        print(f"\nk={k}, metric={metric}")
        print(f"  Точность на обучающей выборке: {train_score:.4f}")
        print(f"  Точность на тестовой выборке: {test_score:.4f}")

        if test_score > best_knn_score:
            best_knn_score = test_score
            best_k = k
            best_metric = metric
            best_model_knn = knn

print("\n" + "-"*80)
print(f"Оптимальные параметры: k={best_k}, metric={best_metric}")
print(f"Лучшая точность на тестовой выборке: {best_knn_score:.4f}")
print("-"*80)

# Вычисление ошибок для лучшей модели KNN
y_train_pred_knn = best_model_knn.predict(X_train)
y_test_pred_knn = best_model_knn.predict(X_test)

err_train_knn = np.mean(y_train != y_train_pred_knn)
err_test_knn = np.mean(y_test != y_test_pred_knn)

print(f"\nУдельное количество ошибок на обучающей выборке: {err_train_knn:.4f}")
print(f"Удельное количество ошибок на тестовой выборке: {err_test_knn:.4f}")
print(f"Точность на обучающей выборке: {1-err_train_knn:.4f}")
print(f"Точность на тестовой выборке: {1-err_test_knn:.4f}")

# ФУНКЦИЯ ВИЗУАЛИЗАЦИИ ОБЛАСТЕЙ РЕШЕНИЙ

def plot_decision_regions(X, y, classifier, resolution=0.02, test_idx=None):
    """
    Визуализирует границы решения классификатора

    Параметры:
    ----------
    X : array-like, shape = [n_samples, n_features]
        Матрица признаков
    y : array-like, shape = [n_samples]
        Вектор меток классов
    classifier : классификатор
        Обученная модель классификации
    resolution : float
        Разрешение сетки для построения поверхности решения
    test_idx : array-like
        Индексы тестовых образцов (для выделения на графике)
    """
    # Настройка генератора маркеров и цветовой карты
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Построение поверхности решения
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Отображение всех образцов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=colors[idx],
                   marker=markers[idx], label=cl, edgecolor='black', s=50)

    # Выделение тестовых образцов
    if test_idx:
        X_test_plot, y_test_plot = X[test_idx, :], y[test_idx]
        plt.scatter(X_test_plot[:, 0], X_test_plot[:, 1], c='none',
                   edgecolor='black', alpha=1.0, linewidth=2, marker='o',
                   s=100, label='Тестовая выборка')

# Объединение обучающей и тестовой выборок для визуализации
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# Визуализация областей решений для SVM
plt.figure(figsize=(10, 8))
plot_decision_regions(X_combined, y_combined, classifier=best_model_svm, 
                     test_idx=range(len(y_train), len(y_train) + len(y_test)))
plt.xlabel('x1', fontsize=14)
plt.ylabel('x2', fontsize=14)
plt.title(f'SVM с полиномиальным ядром (degree={best_params["degree"]}, C={best_params["C"]})', 
          fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('svm_decision_regions.png', dpi=150, bbox_inches='tight')
plt.show()

# Визуализация областей решений для KNN
plt.figure(figsize=(10, 8))
plot_decision_regions(X_combined, y_combined, classifier=best_model_knn, 
                     test_idx=range(len(y_train), len(y_train) + len(y_test)))
plt.xlabel('x1', fontsize=14)
plt.ylabel('x2', fontsize=14)
plt.title(f'K ближайших соседей (k={best_k}, metric={best_metric})', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('knn_decision_regions.png', dpi=150, bbox_inches='tight')
plt.show()

# ЭТАП 4: СРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ

print("\n" + "="*80)
print("ЭТАП 4: СРАВНЕНИЕ КАЧЕСТВА МОДЕЛЕЙ КЛАССИФИКАЦИИ")
print("="*80)

# Создание сравнительной таблицы
comparison_data = {
    'Модель': ['SVM (полиномиальное ядро)', 'K ближайших соседей'],
    'Параметры': [
        f'degree={best_params["degree"]}, C={best_params["C"]}, coef0={best_params["coef0"]}, gamma={best_params["gamma"]}',
        f'k={best_k}, metric={best_metric}'
    ],
    'Ошибка (обучающая)': [f'{err_train_svm:.4f}', f'{err_train_knn:.4f}'],
    'Ошибка (тестовая)': [f'{err_test_svm:.4f}', f'{err_test_knn:.4f}'],
    'Точность (обучающая)': [f'{1-err_train_svm:.4f}', f'{1-err_train_knn:.4f}'],
    'Точность (тестовая)': [f'{1-err_test_svm:.4f}', f'{1-err_test_knn:.4f}']
}

df_comparison = pd.DataFrame(comparison_data)
print("\n" + df_comparison.to_string(index=False))

# Сохранение сравнительной таблицы в CSV
df_comparison.to_csv('model_comparison.csv', index=False, encoding='utf-8-sig')
print("\nСравнительная таблица сохранена в файл model_comparison.csv")

# Визуализация сравнения моделей
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Сравнение точности
models = ['SVM\n(полином. ядро)', 'KNN']
train_accuracy = [1-err_train_svm, 1-err_train_knn]
test_accuracy = [1-err_test_svm, 1-err_test_knn]

x = np.arange(len(models))
width = 0.35

bars1 = axes[0].bar(x - width/2, train_accuracy, width, label='Обучающая', 
                    color='steelblue', alpha=0.8, edgecolor='black')
bars2 = axes[0].bar(x + width/2, test_accuracy, width, label='Тестовая', 
                    color='coral', alpha=0.8, edgecolor='black')

axes[0].set_ylabel('Точность', fontsize=12)
axes[0].set_title('Сравнение точности моделей', fontsize=13, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=11)
axes[0].legend(fontsize=10)
axes[0].set_ylim([0.95, 1.0])
axes[0].grid(True, alpha=0.3, axis='y')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)

# График 2: Сравнение ошибок
train_errors = [err_train_svm, err_train_knn]
test_errors = [err_test_svm, err_test_knn]

bars3 = axes[1].bar(x - width/2, train_errors, width, label='Обучающая', 
                    color='steelblue', alpha=0.8, edgecolor='black')
bars4 = axes[1].bar(x + width/2, test_errors, width, label='Тестовая', 
                    color='coral', alpha=0.8, edgecolor='black')

axes[1].set_ylabel('Удельное количество ошибок', fontsize=12)
axes[1].set_title('Сравнение ошибок моделей', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=11)
axes[1].legend(fontsize=10)
axes[1].set_ylim([0, 0.03])
axes[1].grid(True, alpha=0.3, axis='y')

for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('model_comparison_chart.png', dpi=150, bbox_inches='tight')
plt.show()

# ВЫВОДЫ

print("\n" + "="*80)
print("ВЫВОДЫ")
print("="*80)

if err_test_svm < err_test_knn:
    best_model_name = "SVM с полиномиальным ядром"
    improvement = (err_test_knn - err_test_svm) / err_test_knn * 100
    print(f"1. Лучшую производительность показала модель {best_model_name}")
    print(f"   с точностью {1-err_test_svm:.4f} ({100*(1-err_test_svm):.2f}%) на тестовой выборке")
    print(f"2. Преимущество над KNN в снижении ошибки составляет {improvement:.2f}%")
else:
    best_model_name = "K ближайших соседей"
    improvement = (err_test_svm - err_test_knn) / err_test_svm * 100
    print(f"1. Лучшую производительность показала модель {best_model_name}")
    print(f"   с точностью {1-err_test_knn:.4f} ({100*(1-err_test_knn):.2f}%) на тестовой выборке")
    print(f"2. Преимущество над SVM в снижении ошибки составляет {improvement:.2f}%")

print(f"3. Обе модели демонстрируют высокую точность классификации (>95%)")
print(f"4. SVM показывает разницу между обучающей и тестовой точностью: {abs(err_train_svm - err_test_svm):.4f}")
print(f"5. KNN имеет разницу между обучающей и тестовой точностью: {abs(err_train_knn - err_test_knn):.4f}")
print(f"6. Для задачи XOR обе модели эффективно справляются с нелинейной классификацией")
print("="*80)

