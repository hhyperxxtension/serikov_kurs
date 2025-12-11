import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

# ================================================================================
# ЭТАП 1: ГЕНЕРАЦИЯ ДАННЫХ XOR
# ================================================================================

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
pri
