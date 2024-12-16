import pandas as pd
import numpy as np


class MyTreeReg:
    """
    Класс для обучения и предсказания с использованием дерева решений для задачи регрессии.

    Параметры
    ----------
    max_depth : int, optional
        Максимальная глубина дерева. Дефолтное значение: 5.
    min_samples_split : int, optional
        Минимальное количество объектов в узле для его разделения. Дефолтное значение: 2.
    max_leafs : int, optional
        Максимальное количество листьев в дереве. Дефолтное значение: 20.
    bins : int, optional
        Количество бинов для дискретизации признаков. Если None, используются уникальные значения. Дефолтное значение: None.
    """

    def __init__(self, max_depth: int = 5, min_samples_split: int = 2, max_leafs: int = 20, bins: int = None):
        self.max_depth = max_depth  # Максимальная глубина дерева
        self.min_samples_split = min_samples_split  # Минимальное количество объектов для разбиения
        self.max_leafs = max_leafs if max_leafs > 1 else 2  # Максимальное количество листьев
        self.leafs_cnt = 1  # Счётчик листьев
        self.tree = None  # Корень дерева
        self.bins = bins  # Количество бинов для разбиения признаков
        self.split_candidates = {}  # Словарь возможных точек разбиения
        self.fi = {}  # Важность признаков
        self.total_samples = None  # Общее число обучающих образцов

    def __str__(self):
        return f"MyTreeReg class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def __repr__(self):
        return self.__str__()

    def calculate_mse(self, y: pd.Series) -> float:
        """
        Вычисление среднеквадратичной ошибки (MSE) для заданного целевого вектора.

        Параметры
        ----------
        y : pd.Series
            Целевой вектор.

        Возвращает
        ----------
        float
            Значение MSE.
        """
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)

    def get_best_split(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Поиск наилучшего разбиения для текущего узла, минимизирующего MSE.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор целевой переменной.

        Возвращает
        ----------
        tuple
            Наилучший признак, значение разбиения и прирост качества.
        """
        current_mse = self.calculate_mse(y)
        best_split = {
            'col_name': None,
            'split_value': None,
            'gain': -np.inf
        }

        for col in X.columns:
            candidates = self.split_candidates.get(col, [])
            for split_value in candidates:
                left_mask = X[col] <= split_value
                right_mask = ~left_mask

                y_left, y_right = y[left_mask], y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                mse_left = self.calculate_mse(y_left)
                mse_right = self.calculate_mse(y_right)

                n_left, n_right = len(y_left), len(y_right)
                weighted_mse = (n_left * mse_left + n_right * mse_right) / len(y)

                gain = current_mse - weighted_mse

                if gain > best_split['gain']:
                    best_split = {
                        'col_name': col,
                        'split_value': split_value,
                        'gain': gain
                    }

        return best_split['col_name'], best_split['split_value'], best_split['gain']

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Обучение дерева регрессии на заданных данных.

        Параметры
        ----------
        X_train : pd.DataFrame
            Матрица признаков для обучения.
        y_train : pd.Series
            Целевая переменная для обучения.
        """
        self.total_samples = len(y_train)  # Сохраняем общее число обучающих образцов
        self.fi = {col: 0 for col in X_train.columns}  # Инициализация важности признаков
        self.prepare_split_candidates(X_train)
        self.leafs_cnt = 1  # Сброс счётчика листьев
        self.tree = self.build_tree(X_train, y_train, depth=1)

    def prepare_split_candidates(self, X_train: pd.DataFrame):
        """
        Подготовка возможных точек разбиения для каждого признака.

        Параметры
        ----------
        X_train : pd.DataFrame
            Матрица признаков для обучения.
        """
        for col in X_train.columns:
            values = X_train[col].sort_values().unique()
            if self.bins is None:
                candidates = (values[:-1] + values[1:]) / 2 if len(values) > 1 else []
            else:
                if len(values) <= self.bins - 1:
                    candidates = (values[:-1] + values[1:]) / 2 if len(values) > 1 else []
                else:
                    _, bin_edges = np.histogram(X_train[col], bins=self.bins)
                    candidates = bin_edges[1:-1] if len(bin_edges) > 2 else []

            self.split_candidates[col] = candidates

    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> dict:
        """
        Рекурсивное построение дерева регрессии.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков текущего узла.
        y : pd.Series
            Целевая переменная текущего узла.
        depth : int
            Текущая глубина дерева.

        Возвращает
        ----------
        dict
            Узел дерева в виде словаря.
        """
        if (
            depth > self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
            or self.leafs_cnt >= self.max_leafs
        ):
            return {"leaf": np.mean(y)}

        col, split_value, gain = self.get_best_split(X, y)

        if col is None or gain <= 1e-12:
            return {"leaf": np.mean(y)}

        if self.leafs_cnt >= self.max_leafs:
            return {"leaf": np.mean(y)}

        fi_increment = (len(y) / self.total_samples) * gain  # Приводим прирост качества к сопоставимой метрике
        self.fi[col] += fi_increment

        self.leafs_cnt += 1

        left_mask = X[col] <= split_value
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        left_node = self.build_tree(X_left, y_left, depth + 1)
        right_node = self.build_tree(X_right, y_right, depth + 1)

        return {
            "split_feature": col,
            "split_value": split_value,
            "left": left_node,
            "right": right_node
        }

    def predict(self, X: pd.DataFrame) -> list:
        """
        Предсказание значений целевой переменной для новых данных.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков для предсказания.

        Возвращает
        ----------
        list
            Список предсказанных значений.
        """
        def traverse_tree(row, node):
            if "leaf" in node:
                return node["leaf"]
            if row[node["split_feature"]] <= node["split_value"]:
                return traverse_tree(row, node["left"])
            else:
                return traverse_tree(row, node["right"])

        return [traverse_tree(row, self.tree) for _, row in X.iterrows()]

    def print_tree(self, node: dict = None, depth: int = 0):
        """
        Рекурсивный вывод структуры дерева.

        Параметры
        ----------
        node : dict, optional
            Узел дерева для вывода. Если None, используется корень дерева.
        depth : int, optional
            Текущая глубина дерева для форматирования. Дефолтное значение: 0.
        """
        if node is None:
            node = self.tree

        if "leaf" in node:
            print("  " * depth + f"leaf = {node['leaf']:.4f}")
        else:
            print("  " * depth + f"{node['split_feature']} <= {node['split_value']:.4f}")
            self.print_tree(node["left"], depth + 1)
            self.print_tree(node["right"], depth + 1)
