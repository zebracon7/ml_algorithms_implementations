import pandas as pd
import numpy as np


class MyLogReg():
    """
    Логистическая регрессия
    """

    def __init__(self, n_iter: int, learning_rate: float) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def __repr__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
        # Вычисляем количество наблюдений
        n_samples = X.shape[0]

        # Добавляем единичный столбец слева в матрицу фичей:
        X.insert(0, 'bias', np.ones(n_samples))

        # Определяем количество фичей
        features_num = X.shape[1]

        # Создаем вектор весов
        weights = np.ones(features_num)

        # В формуле logloss'a мы вычисляем логарифм, и если аргумент будет равен 0, то даст -inf
        # поэтому, будем добавлять к аргументу логарифма eps = 1e-15
        eps = 1e-15

        for i in range(self.n_iter):
            # Делаем предсказание по формуле y_pred = 1 / (1 + e ^ (-z))
            z = np.clip(X @ weights, -500, 500) # clip, чтобы избежать переполнения
            y_pred = 1 / (1 + np.exp(-z))

            # Считаем ошибку
            logloss = -1 * (y @ np.log10(y_pred + eps) + (1 - y) * np.log10(1 - y_pred + eps)).mean()

            # Считаем градиент
            gradient_logloss = ((y_pred - y) @ X) / n_samples

            # Делаем шаг регрессии
            weights -= self.learning_rate * gradient_logloss

            # Выводим лог
            if verbose and (i == 0 or i % verbose == 0):
                if i == 0:
                    log_num = ('start |')
                else:
                    log_num = f'{i} |'
                print(f'{log_num} loss: {logloss:.2f}')
                
        # Сохраняем веса внутри класса
        self.weights = weights

    def get_coef(self) -> np.array:
        return self.weights[1:]