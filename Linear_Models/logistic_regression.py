import pandas as pd
import numpy as np


class MyLogReg():
    """
    Логистическая регрессия
    """

    def __init__(self, n_iter, learning_rate, metric=None) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric

        # Создаем словарь метрик
        self.dict_metric = {
            'accuracy': self.accuracy, 
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1
            # 'roc_auc': self.roc_auc
        }

        # Атрибут для хранения последнего значения метрики
        self.best_score = None

    def __str__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def __repr__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        # Вычисляем количество наблюдений
        n_samples = X.shape[0]

        # Добавляем единичный столбец слева в матрицу фичей:
        X = X.copy()
        X.insert(0, 'bias', np.ones(n_samples))

        # Определяем количество фичей
        features_num = X.shape[1]

        # Создаем вектор весов
        weights = np.ones(features_num)

        # В формуле logloss'a мы вычисляем логарифм, и если аргумент будет равен 0, то даст -inf
        # поэтому, будем добавлять к аргументу логарифма eps = 1e-15
        eps = 1e-15

        for i in range(self.n_iter):
            # Вызываем predict_proba для получения вероятностей
            self.weights = weights  # Обновляем веса внутри класса, чтобы использовать predict_proba
            y_pred_proba = self.predict_proba(X)
            # Делаем предсказание по формуле y_pred = 1 / (1 + e ^ (-z))

            # Считаем ошибку
            logloss = -1 * (y @ np.log10(y_pred_proba + eps) + (1 - y) * np.log10(1 - y_pred_proba + eps)).mean()

            # Считаем градиент
            gradient_logloss = ((y_pred_proba - y) @ X) / n_samples

            # Делаем шаг регрессии
            weights -= self.learning_rate * gradient_logloss

            # Вызываем predict для получения бинарных предсказаний
            y_pred = self.predict(X)

            # Считаем доп. метрику для вывода, если задана
            metric_value = None
            if self.metric in self.dict_metric: 
                metric_value = self.dict_metric[self.metric](y, y_pred)
                self.best_score = metric_value

            # Выводим лог
            if verbose and (i == 0 or i % verbose == 0):
                if i == 0:
                    log = f'start | loss: {logloss:.2f}'
                else:
                    log = f'{i} | loss: {logloss:.2f}'
                if metric_value is not None:
                    log = f'{log} | {self.metric}: {metric_value:.2f}'
                print(log)

        # Сохраняем веса внутри класса, после обучения
        self.weights = weights
    
    def predict_proba(self, X: pd.DataFrame):
        """Doc"""
        predict_vector = 1 / (1 + np.exp(-X @ self.weights))
        return predict_vector
    
    def predict(self, X: pd.DataFrame):
        """Doc"""
        predict_vector = self.predict_proba(X)
        # Преобразуем вероятности в классы 0 или 1
        # Если значение меньше или равно 0.5, результатом будет, иначе True
        return (predict_vector > 0.5).astype(int)

    def get_coef(self) -> np.array:
        """Doc"""
        return self.weights[1:]
    
    def accuracy(self, y_true: pd.Series, y_pred: pd.Series):
        """Вычисляет точность модели"""
        correct_predictions = (y_true == y_pred).sum()  # Количество правильных предсказаний
        total_predictions = len(y_true)  # Общее количество предсказаний
        return correct_predictions / total_predictions
    
    def precision(self, y_true: pd.Series, y_pred: pd.Series):
        true_positive = ((y_true == 1) & (y_pred == 1)).sum()  # Истинно положительные
        tp_fp = (y_pred == 1).sum()  # Общее количество предсказанных положительных классов
        return true_positive / tp_fp if tp_fp != 0 else 0
    
    def recall(self, y_true: pd.Series, y_pred: pd.Series):
        tp = ((y_true == 1) & (y_pred == 1)).sum()  # Истинно положительные
        fn = ((y_true == 1) & (y_pred == 0)).sum()  # Ложно отрицательные
        tp_fn = tp + fn  # Истинно положительные + Ложно отрицательные
        return tp / tp_fn if tp_fn != 0 else 0
    
    def f1(self, y_true: pd.Series, y_pred: pd.Series):
        b = 1  # Для F1-score (b = 1)
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        f1 = (1 + b**2) * (precision * recall) / (b**2 * precision + recall)
        return f1 if (precision + recall) != 0 else 0
    
    def get_best_score(self):
        return self.best_score
