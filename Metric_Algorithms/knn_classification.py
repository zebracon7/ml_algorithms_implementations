import pandas as pd
import numpy as np


class MyKNNClf:
    """
    Классификатор на основе алгоритма k ближайших соседей (kNN).

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
    def __init__(self, k=3, metric='euclidean', weight='uniform') -> None:
        self.k = k
        self.train_size = None
        self.X = None
        self.y = None
        self.metric = metric
        self.weight = weight

        self.dict_metric = {
            'euclidean': self.euclidean_distance,
            'chebyshev': self.chebyshev_distance,
            'manhattan': self.manhattan_distance,
            'cosine': self.cosine_distance
        }

        self.dict_weight = {
            'uniform': None,
            'rank': self.rank_metric,
            'distance': self.distance_metric
        }

    def __str__(self):
        return f'MyKNNClf class: k={self.k}'
   
    def __repr__(self):
        return f'MyKNNClf class: k={self.k}'

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
        self.X = X.copy()
        self.y = y.copy()
        self.train_size = X.shape

    def compute_distances(self, X: pd.DataFrame) -> np.ndarray:
        """
        Вычисление евклидовых расстояний между тестовыми и обучающими объектами.
        
        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.
        
        Returns
        -------
        np.ndarray
            Матрица расстояний размером (количество тестовых объектов, количество обучающих объектов).
        """
        distance = self.dict_metric[self.metric](X)
        return distance

    def get_k_nearest_targets(self, distance: np.ndarray) -> np.ndarray:
        """
        Получение меток k ближайших соседей для каждого тестового объекта.
        
        Параметры
        ----------
        distance : np.ndarray
            Матрица расстояний между тестовыми и обучающими объектами.
        
        Returns
        -------
        np.ndarray
            Массив меток k ближайших соседей для каждого тестового объекта.
        """
        nearest_targets = np.zeros((distance.shape[0], self.k))
        
        for i in range(distance.shape[0]):
            # Сортируем индексы в порядке возрастания значений расстояний
            nearest_idx = np.argsort(distance[i])

            # Отбираем k ближайших соседей
            nearest_idx = nearest_idx[:self.k]
            nearest_targets[i] = self.y.iloc[nearest_idx].values
        
        return nearest_targets

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание классов для тестовых объектов.
        
        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.
        
        Returns
        -------
        np.ndarray
            Вектор предсказанных классов.
        """
        # Вычисляем расстояния до каждого объекта
        distance = self.compute_distances(X)

        # Получаем массив меток k ближайших соседей для каждого тестового объекта
        nearest_targets = self.get_k_nearest_targets(distance)
        
        if self.weight == 'uniform':
            # Предсказываем класс как 0, если среднее значение < 0.5, иначе 1
            predictions = (nearest_targets.mean(axis=1) >= 0.5).astype(int)
        else:
            # Получаем веса для k ближайших соседей
            weights = self.dict_weight[self.weight](distance)
            # Вычисляем взвешенное среднее для каждой тестовой точки
            weighted_sum = np.sum(weights * nearest_targets, axis=1)
            sum_weights = np.sum(weights, axis=1)
            proba = weighted_sum / sum_weights
            # Предсказываем класс 1, если вероятность >= 0.5, иначе 0
            predictions = (proba >= 0.5).astype(int)

        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказание вероятностей классов для тестовых объектов.
        
        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.
        
        Returns
        -------
        np.ndarray
            Вектор предсказанных вероятностей принадлежности к классу 1.
        """
        # Вычисляем расстояния до каждого объекта
        distance = self.compute_distances(X)

        # Получаем массив меток k ближайших соседей для каждого тестового объекта
        nearest_targets = self.get_k_nearest_targets(distance)
        
        if self.weight == 'uniform':
            # Вероятность класса 1 как среднее значение меток соседей
            proba = nearest_targets.mean(axis=1)
        elif self.weight == 'rank':
            # Генерируем веса на основе ранга (обратные рангу)
            weights = self.rank_metric(distance)
            
            # Суммируем веса для класса 1
            weighted_sum_cls1 = np.sum(weights * (nearest_targets == 1), axis=1)
            # Сумма всех весов
            sum_weights = np.sum(weights, axis=1)
            # Вычисляем нормированные веса для класса 1
            proba = weighted_sum_cls1 / sum_weights
        elif self.weight == 'distance':
            # Реализация весовой схемы 'distance'
            weights = self.distance_metric(distance)
            
            # Суммируем веса для класса 1
            weighted_sum_cls1 = np.sum(weights * (nearest_targets == 1), axis=1)
            # Сумма всех весов
            sum_weights = np.sum(weights, axis=1)
            # Вычисляем нормированные веса для класса 1
            proba = weighted_sum_cls1 / sum_weights

        else:
            raise ValueError("Некорректное значение weight")
        
        return proba

    def euclidean_distance(self, X: pd.DataFrame):
        """
        Вычисление Евклидовых расстояний между тестовыми и обучающими объектами.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.

        Returns
        -------
        np.ndarray
            Матрица Евклидовых расстояний размером.
        """
        distance = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            distance[i, :] = np.sqrt(np.sum((self.X.values - X.iloc[i].values) ** 2, axis=1))
        
        return distance
    
    def chebyshev_distance(self, X: pd.DataFrame):
        """
        Вычисление Расстояний Чебышева между тестовыми и обучающими объектами.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.

        Returns
        -------
        np.ndarray
            Матрица Расстояний Чебышева размером.
        """
        distance = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            distance[i, :] = np.max(np.abs(self.X.values - X.iloc[i].values), axis=1)
        
        return distance

    def manhattan_distance(self, X: pd.DataFrame):
        """
        Вычисление Манхэттенских расстояний между тестовыми и обучающими объектами.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.

        Returns
        -------
        np.ndarray
            Матрица Манхэттенских расстояний размером.
        """
        distance = np.zeros((X.shape[0], self.X.shape[0]))
        for i in range(X.shape[0]):
            distance[i, :] = np.sum(np.abs(self.X.values - X.iloc[i].values), axis=1)
        
        return distance
    
    def cosine_distance(self, X: pd.DataFrame):
        """
        Вычисление Косинусных расстояний между тестовыми и обучающими объектами.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков тестовой выборки.

        Returns
        -------
        np.ndarray
            Матрица Косинусных расстояний размером.

        Raises
        ------
        ValueError
            Если норма тестового вектора равна нулю, что приводит к делению на ноль.
        """
        distance = np.zeros((X.shape[0], self.X.shape[0]))
    
        # Предварительно вычислим нормы обучающих объектов
        norm_X = np.linalg.norm(self.X.values, axis=1)
        
        for i in range(X.shape[0]):
            x = X.iloc[i].values
            norm_x = np.linalg.norm(x)
            
            if norm_x == 0:
                raise ValueError(f"Норма тестового вектора с индексом {i} равна нулю, заполните корректно данные.")
            
            # Вычисляем скалярное произведение между тестовым вектором и всеми обучающими
            dot_product = np.dot(self.X.values, x)
            
            # Косинусное расстояние
            distance[i, :] = 1 - dot_product / (norm_X * norm_x)
        
        return distance
    
    def rank_metric(self, distance: np.ndarray) -> np.ndarray:
        """
        Генерация весов на основе обратных рангов соседей.

        Параметры
        ----------
        distance : np.ndarray
            Матрица расстояний между тестовыми и обучающими объектами.

        Returns
        -------
        np.ndarray
            Матрица весов размером (количество тестовых объектов, k), 
            где веса основаны на обратных рангах соседей.
        """
        # Генерируем ранги: ближайший сосед имеет ранг 1, следующий 2, ..., последний k
        ranks = np.arange(1, self.k + 1)  
        
        # Вычисляем обратные ранги
        weights = 1.0 / ranks  # [1.0, 0.5, 0.333..., ..., 1/k]
        
        # Создаём матрицу весов, повторяя массив весов для всех тестовых объектов
        num_test = distance.shape[0]
        weights_matrix = np.tile(weights, (num_test, 1))
        
        return weights_matrix

    def distance_metric(self, distance: np.ndarray) -> np.ndarray:
        """
        Реализация весовой схемы 'distance'.

        Параметры
        ----------
        distance : np.ndarray
            Матрица расстояний между тестовыми и обучающими объектами.

        Returns
        -------
        np.ndarray
            Матрица весов, основанных на расстоянии.
            Веса рассчитываются как обратные расстояния к k ближайшим соседям.
        """
        # Получаем индексы k ближайших соседей для каждого тестового объекта
        sorted_indices = np.argsort(distance, axis=1)[:, :self.k]
        
        # Создаем массив индексов строк
        row_indices = np.arange(distance.shape[0])[:, np.newaxis]
        
        # Извлекаем k ближайших расстояний для каждого тестового объекта
        distance_k = distance[row_indices, sorted_indices]
        
        # Вычисляем веса как обратные расстояния
        weights = 1.0 / distance_k
        
        return weights
