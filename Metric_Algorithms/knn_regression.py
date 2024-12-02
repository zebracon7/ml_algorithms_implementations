import pandas as pd
import numpy as np


class MyKNNReg():
    """
    Регрессор на основе алгоритма k ближайших соседей (kNN).

    Параметры
    ----------
    k : int, optional
        Количество ближайших соседей для классификации.
        Дефолтное значение: 3
    metric: str, optional
        Способ нахождения расстояния между соседями
        Дефолтное значение: euclidean (Евклидово расстояние)
    weight: str, optional
        Определяет вариант подбора весов: без весов, rank (по индексу), distance (по расстоянию)
        Дефолтное значение: uniform (без весов)
    """
    def __init__(self, k=3, metric='euclidean', weight='uniform'):
        self.k = k
        self.train_size = None
        self.X_train = None
        self.y_train = None
        self.metric = metric
        self.weight = weight

    def __str__(self):
        return f'MyKNNReg class: k={self.k}'
    
    def __repr__(self):
        return f'MyKNNReg class: k={self.k}'
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        "Обучение" модели на тренировочных данных.
        
        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков обучающей выборки.
        y : pd.Series
            Вектор целевых переменных обучающей выборки.
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.train_size = X.shape

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание целевых переменных для тестовых объектов.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.

        Returns
        -------
        np.ndarray
            Вектор предсказанных значений.
        """
        # Вычисляем расстояния до каждого объекта
        distance = self.compute_distances(X)
        # Массив для хранения предсказаний
        prediction = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            # Сортируем индексы в порядке возрастания расстояний и выбираем K ближайших
            nearest_idx = np.argsort(distance[i])[:self.k]
            # Значения целевой переменной K ближайших соседей
            nearest_targets = self.y_train.iloc[nearest_idx].values
            # Расстояния до K ближайших соседей
            nearest_distances = distance[i, nearest_idx]

            if self.weight == 'uniform':
                # Среднее значение целевых переменных ближайших соседей
                prediction[i] = nearest_targets.mean()
            elif self.weight == 'rank':
                # Вес для каждого соседа (обратный его рангу)
                weights = 1 / (np.arange(1, self.k + 1))
                # Взвешенное среднее целевых переменных
                prediction[i] = np.sum(nearest_targets * weights) / np.sum(weights)
            elif self.weight == 'distance':
                # Вес для каждого соседа (обратный его дистанции)
                weights = 1 / nearest_distances
                prediction[i] = np.sum(nearest_targets * weights) / np.sum(weights)

        return prediction

    def compute_distances(self, X: pd.DataFrame) -> np.ndarray:
        """
        Вычисление расстояний между тестовыми и обучающими объектами.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.

        Returns
        -------
        np.ndarray
            Матрица расстояний размером (количество тестовых объектов, количество обучающих объектов).
        """

        distance = np.zeros((X.shape[0], self.X_train.shape[0]))

        if self.metric == 'euclidean':
            for i in range(X.shape[0]):
                distance[i, :] = np.sqrt(np.sum((self.X_train.values - X.iloc[i].values) ** 2, axis=1))
        elif self.metric == 'chebyshev':
            for i in range(X.shape[0]):
                distance[i, :] = np.max(np.abs(self.X_train.values - X.iloc[i].values), axis=1)
        elif self.metric == 'manhattan':
            for i in range(X.shape[0]):
                distance[i, :] = np.sum(np.abs(self.X_train.values - X.iloc[i].values), axis=1)
        elif self.metric == 'cosine':
            # Нормы обучающих объектов
            norm_X_train = np.linalg.norm(self.X_train.values, axis=1)
            # Нормы тестовых объектов
            norm_X_test = np.linalg.norm(X.values, axis=1)
            # Проверка на нулевые вектора
            if np.any(norm_X_train == 0) or np.any(norm_X_test == 0):
                raise ValueError("Нулевые вектора обнаружены в обучающих или тестовых данных.")
            # Вычисление скалярного произведения
            dot_product = np.dot(X.values, self.X_train.values.T)
            # Косинусное расстояние
            distance = 1 - dot_product / (norm_X_test[:, None] * norm_X_train[None, :])
        else:
            raise ValueError('Некорректно введено название метрики')
            
        return distance