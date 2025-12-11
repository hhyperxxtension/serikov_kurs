# Импорт
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# reproducibility
np.random.seed(0)

# 1) Генерация данных XOR
X = np.random.randn(200, 2)
y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
y = np.where(y, 1, 0)

# Разбиение
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Вспомогательная функция для визуализации областей решений
def plot_decision_regions(X, y, pipeline, ax=None, title=None, test_idx=None, h=0.02):
    # pipeline: Pipeline со StandardScaler и классификатором, поэтому используем pipeline.predict
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = pipeline.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X[y==0,0], X[y==0,1], marker='s', edgecolor='k', label='class 0')
    ax.scatter(X[y==1,0], X[y==1,1], marker='x', edgecolor='k', label='class 1')
    if test_idx is not None:
        ax.scatter(X[test_idx,0], X[test_idx,1], facecolors='none', edgecolors='k',
                   s=100, linewidths=1.2, label='test samples')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    if title:
        ax.set_title(title)
    ax.legend(loc='upper left')

# -------------------------
# SVM с полиномиальным ядром (упрощённый GridSearch)
param_grid_svm = {
    'svc__C': [1.0, 5.0, 10.0],
    'svc__degree': [2, 3],
    'svc__coef0': [0.0, 1.0]
}
pipe_svm = Pipeline([('sc', StandardScaler()), ('svc', SVC(kernel='poly', gamma='auto', random_state=0))])
grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=3, n_jobs=1)
grid_svm.fit(X_train, y_train)

best_svm = grid_svm.best_estimator_
y_train_pred_svm = best_svm.predict(X_train)
y_test_pred_svm = best_svm.predict(X_test)
err_train_svm = 1 - accuracy_score(y_train, y_train_pred_svm)
err_test_svm = 1 - accuracy_score(y_test, y_test_pred_svm)

# -------------------------
# KNN (упрощённый GridSearch)
param_grid_knn = {
    'knn__n_neighbors': [1, 3, 5],
    'knn__p': [1, 2],
}
pipe_knn = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier())])
grid_knn = GridSearchCV(pipe_knn, param_grid_knn, cv=3, n_jobs=1)
grid_knn.fit(X_train, y_train)

best_knn = grid_knn.best_estimator_
y_train_pred_knn = best_knn.predict(X_train)
y_test_pred_knn = best_knn.predict(X_test)
err_train_knn = 1 - accuracy_score(y_train, y_train_pred_knn)
err_test_knn = 1 - accuracy_score(y_test, y_test_pred_knn)

# -------------------------
# Вывод результатов
print("=== SVM (poly) ===")
print("Best params (SVM):", grid_svm.best_params_)
print(f"Train error (SVM) = {err_train_svm:.4f}")
print(f"Test  error (SVM) = {err_test_svm:.4f}\n")

print("=== KNN ===")
print("Best params (KNN):", grid_knn.best_params_)
print(f"Train error (KNN) = {err_train_knn:.4f}")
print(f"Test  error (KNN) = {err_test_knn:.4f}\n")

# -------------------------
# Графики областей решений (объединяем train+test для плотности в сетке)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
test_idx = np.arange(len(X_train), len(X_train) + len(X_test))

fig, axes = plt.subplots(1, 2, figsize=(12,5))
plot_decision_regions(X_combined, y_combined, pipeline=best_svm, ax=axes[0],
                      title=f"SVM poly (best)  test_err={err_test_svm:.3f}", test_idx=test_idx)
plot_decision_regions(X_combined, y_combined, pipeline=best_knn, ax=axes[1],
                      title=f"KNN (best) test_err={err_test_knn:.3f}", test_idx=test_idx)
plt.tight_layout()
plt.show()

# -------------------------
# Краткое сравнение
if err_test_svm < err_test_knn:
    winner = "SVM (poly)"
elif err_test_knn < err_test_svm:
    winner = "KNN"
else:
    winner = "Одинаково (tie)"

print("Краткое сравнение:")
print(f"Test error SVM = {err_test_svm:.4f}; Test error KNN = {err_test_knn:.4f}")
print(f"Лучший по тестовой ошибке: {winner}")