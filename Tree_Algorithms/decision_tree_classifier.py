import pandas as pd
import numpy as np

class MyTreeClf:
    """
    Класс для обучения и предсказания с использованием дерева решений.

    Параметры
    ----------
    max_depth : int, optional
        Максимальная глубина дерева. Дефолтное значение: 5.
    min_samples_split : int, optional
        Минимальное количество объектов в узле для его разделения. Дефолтное значение: 2.
    max_leafs : int, optional
        Максимальное количество листьев в дереве. Дефолтное значение: 20.
    bins : int, optional
        Количество бинов для дискретизации признаков. Если None, используется уникальные значения. Дефолтное значение: None.
    criterion : str, optional
        Критерий для измерения качества разбиения ("entropy" или "gini"). Дефолтное значение: "entropy".
    """

    def __init__(self, max_depth=5, min_samples_split=2, max_leafs=20, bins=None, criterion='entropy'):
        self.max_depth = max_depth  # Максимальная глубина дерева
        self.min_samples_split = min_samples_split  # Минимальное количество объектов для разбиения
        self.max_leafs = max_leafs if max_leafs > 1 else 2  # Максимальное количество листьев (минимум 2)
        self.leafs_cnt = 1
        self.tree = None  # Корень дерева
        self.bins = bins  # Количество бинов для разбиения признаков
        self.split_candidates = {}  # Словарь возможных точек разбиения для каждого признака
        self.criterion = criterion  # Критерий для оценки качества разбиения
        self.fi = {}  # Словарь для хранения важности фичей

    def __str__(self):
        return f"MyTreeClf class: max_depth={self.max_depth}, min_samples_split={self.min_samples_split}, max_leafs={self.max_leafs}"

    def fit(self, X, y):
        """
        Обучение дерева решений на заданных данных.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков.
        y : pd.Series
            Вектор целевой переменной.
        """
        # Инициализируем важности признаков
        self.fi = {col: 0 for col in X.columns}  
        self.prepare_split_candidates(X)
        self.leafs_cnt = 1
        self.tree = self.build_tree(X, y, depth=0, total_samples=len(y))

    def prepare_split_candidates(self, X):
        """
        Подготовка возможных точек разделения для каждого признака.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков.
        """
        for col in X.columns:
            # Уникальные значения признака в отсортированном порядке
            values = np.sort(X[col].unique())
            # Если уникальное значение только одно, разделений нет
            if len(values) < 2:
                self.split_candidates[col] = []
            elif self.bins is None:
                # Если бины не заданы, берем середины между соседними значениями
                self.split_candidates[col] = (values[:-1] + values[1:]) / 2
            else:
                # Если бины заданы, используем их для определения точек разделения
                _, bin_edges = np.histogram(values, bins=self.bins)
                self.split_candidates[col] = bin_edges[1:-1] if len(bin_edges) > 2 else []

    def build_tree(self, X, y, depth, total_samples):
        """
        Рекурсивное построение дерева решений.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков текущего узла.
        y : pd.Series
            Целевая переменная текущего узла.
        depth : int
            Текущая глубина дерева.
        total_samples : int
            Общее количество объектов в датасете.

        Возвращает
        ----------
        dict
            Структура узла дерева.
        """
        # Условия остановки построения дерева
        if len(y) == 0 or len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split or self.leafs_cnt >= self.max_leafs:
            # Возвращаем листовой узел
            return {"leaf": y.mean() if len(y) > 0 else 0}

        # Поиск лучшего разбиения для текущего узла
        best_col, best_split, best_ig = self.get_best_split(X, y)

        if best_col is None or best_ig <= 1e-6:
            # Если разбиение не найдено, возвращаем лист
            return {"leaf": y.mean()}

        # Расчет важности признака для текущего узла
        N_p = len(y)
        left_mask = X[best_col] <= best_split
        right_mask = ~left_mask

        # Разделение целевой переменной на подвыборки
        y_left, y_right = y[left_mask], y[right_mask]

        N_l = len(y_left)
        N_r = len(y_right)

        I_p = self.calculate_criterion(y)
        I_l = self.calculate_criterion(y_left)
        I_r = self.calculate_criterion(y_right)

        # Увеличение важности признака
        fi_increment = (N_p / total_samples) * (I_p - (N_l / N_p) * I_l - (N_r / N_p) * I_r)
        self.fi[best_col] += fi_increment

        # Разделение данных на подвыборки
        X_left, X_right = X[left_mask], X[right_mask]
        self.leafs_cnt += 1

        return {
            "split_feature": best_col,  # Признак для разбиения
            "split_value": best_split,   # Значение разбиения
            "left": self.build_tree(X_left, y_left, depth + 1, total_samples),
            "right": self.build_tree(X_right, y_right, depth + 1, total_samples),
        }

    def get_best_split(self, X, y):
        """
        Поиск лучшего разделения для текущего узла.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков текущего узла.
        y : pd.Series
            Целевая переменная текущего узла.

        Возвращает
        ----------
        tuple
            Наилучший признак, значение разделения и информационный прирост.
        """
        count_labels = y.value_counts()  # Распределение меток классов
        p_zero = count_labels / count_labels.sum()  # Вероятности классов
        s_zero = -np.sum([p * np.log2(p) for p in p_zero if p > 0]) if self.criterion == 'entropy' else 1 - np.sum(p_zero ** 2)

        best_col = None
        best_split = None
        best_score = float('inf')

        for col in X.columns:
            candidates = self.split_candidates[col]  # Возможные значения для разбиения
            if len(candidates) == 0:
                continue

            for split_value in candidates:
                left_mask = X[col] <= split_value
                right_mask = ~left_mask

                y_left, y_right = y[left_mask], y[right_mask]

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                left_count = y_left.value_counts()
                p_left = left_count / left_count.sum()
                s_left = -np.sum([p * np.log2(p) for p in p_left if p > 0]) if self.criterion == 'entropy' else 1 - np.sum(p_left ** 2)

                right_count = y_right.value_counts()
                p_right = right_count / right_count.sum()
                s_right = -np.sum([p * np.log2(p) for p in p_right if p > 0]) if self.criterion == 'entropy' else 1 - np.sum(p_right ** 2)

                weighted_score = (len(y_left) / len(y)) * s_left + (len(y_right) / len(y)) * s_right

                if weighted_score < best_score:
                    best_score = weighted_score
                    best_col = col
                    best_split = split_value

        return best_col, best_split, s_zero - best_score

    def calculate_criterion(self, y):
        """
        Вычисление критерия разбиения (Gini или Entropy).

        Параметры
        ----------
        y : pd.Series
            Целевая переменная текущего узла.

        Возвращает
        ----------
        float
            Значение критерия.
        """
        if len(y) == 0:
            return 0
        count_labels = y.value_counts()
        p = count_labels / count_labels.sum()
        if self.criterion == 'entropy':
            return -np.sum([p_i * np.log2(p_i) for p_i in p if p_i > 0])
        elif self.criterion == 'gini':
            return 1 - np.sum(p ** 2)

    def predict_proba(self, X):
        """
        Предсказание вероятностей классов для каждого объекта.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков.

        Возвращает
        ----------
        pd.Series
            Предсказанные вероятности.
        """
        def traverse(row, node):
            if "leaf" in node:
                return node["leaf"]
            if row[node["split_feature"]] <= node["split_value"]:
                return traverse(row, node["left"])
            else:
                return traverse(row, node["right"])

        return X.apply(lambda row: traverse(row, self.tree), axis=1)

    def predict(self, X):
        """
        Предсказание классов для каждого объекта.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков.

        Возвращает
        ----------
        pd.Series
            Предсказанные классы.
        """
        return (self.predict_proba(X) >= 0.5).astype(int)

    def print_tree(self, node=None, depth=0):
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
            print(f"{'  ' * depth}leaf = {node['leaf']:.4f}")
        else:
            print(f"{'  ' * depth}{node['split_feature']} <= {node['split_value']:.4f}")
            self.print_tree(node['left'], depth + 1)
            self.print_tree(node['right'], depth + 1)