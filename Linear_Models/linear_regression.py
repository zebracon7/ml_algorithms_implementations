import pandas as pd
import numpy as np
import random


class MyLineReg():
    """
    Линейная регрессия
    
    Параметры
    ----------
    n_iter: int, optional
        Количество шагов градиентного спуска
        Дефолтное значение: 100
    learning_rate: Union[float, Callable], optional
        Коэффициент скорости обучения градиентного спуска
        Если на вход пришла lambda ф-я, то вычисляется на каждом шаге обучения
        Дефолтное значение: 0.1
    metric: str, optional
        Вывод дополнительной метрики в лог обучения
        Возможные значения: mae, mse, rmse, mape, r2
        Дефолтное значение: None
    reg: str, optional
        Подключение регуляризации
        Возможные значения: l1, l2, elasticnet
        Дефолтное значение: None
    l1_coef: float, optional
        Коэффициент регуляризации l1
        Дефолтное значение: 0
    l2_coef: float, optional
        Коэффициент регуляризации l2
        Дефолтное значение: 0
    sgd_sample: Union[int, float], optional
        Количество образцов для обучения на каждом шагу
        Может принимать целые числа, либо дробные от 0.0 до 1.0
        В случае дробного числя - является долей от всего датасета
        Дефолтное значение: None
    random_state: int
        Сид для воспроизводимости результата
        Дефолтное значение: 42
    """
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        self.random_state = random_state

        # Создаем словарь метрик
        self.dict_metric = {
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "mape": self.mape,
            "r2": self.r2
        }

        # Создаем словарь типа регуляризации
        self.dict_reg = {
            "l1": self.l1_reg,
            "l2": self.l2_reg,
            "elasticnet": self.elasticnet_reg
        }

        # Атрибут для хранения последнего значения метрики
        self.best_score = None

    def __str__(self) -> str:
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def __repr__(self) -> str:
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        """
        Обучение модели с использованием градиентного спуска

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков
        y : pd.Series
            Вектор целевой переменной
        verbose : int, optional
            Периодичность логирования потерь и метрик в процессе обучения
        """
        # Фиксируем сид (для SDG)
        random.seed(self.random_state)

        # Добавляем единичный столбец слева в матрицу фичей:
        X.insert(0, 'bias', np.ones(X.shape[0]))

        # Количество фичей и число наблюдений
        features_num = X.shape[1]
        n_samples = X.shape[0]

        # Создаем вектор весов
        weights = np.ones(features_num)

        # Градиентный спуск
        for i in range(1, self.n_iter + 1):
            # Если "включен" режим SDG, то отбираем мини-батч
            if self.sgd_sample:
                # Формируем порядковые номера строк для отбора SDG
                if self.sgd_sample > 1: 
                    batch_size = self.sgd_sample
                else:
                    batch_size = round(X.shape[0] * self.sgd_sample)
                    batch_size = 1 if batch_size < 1 else batch_size
                
                sample_rows_idx = random.sample(range(X.shape[0]), batch_size)

                # Отбираем mini-batch
                # X_batch = X.iloc[sample_rows_idx]
                # y_batch = pd.Series(y).loc[sample_rows_idx]
                X_batch = X.iloc[sample_rows_idx]
                y_batch = y.iloc[sample_rows_idx]

            else:
                X_batch = X
                y_batch = y
                batch_size = n_samples

            # Делаем предсказание (умножаем матрицу фичей на вектор весов)
            y_pred = X_batch @ weights

            # Если включена регуляризация
            mse_add, gradient_mse_add = 0, 0
            if self.reg in self.dict_reg:
                mse_add, gradient_mse_add = self.dict_reg[self.reg](weights)

            # Считаем MSE
            mse = self.mse(y, y_pred) + mse_add
 
            # Вычисляем градиент используем mini-batch для градиента
            gradient_mse = 2 / batch_size * (y_pred - y_batch) @ X_batch + gradient_mse_add

            # Если learning_rate задан как число
            if isinstance(self.learning_rate, float):
                # Обновляем веса
                weights -= self.learning_rate * gradient_mse
            else:
                weights -= self.learning_rate(i) * gradient_mse

            # Считаем доп. метрику для вывода, если задана
            metric_value = None
            if self.metric in self.dict_metric:
                # Можно добавить, что если self.metric = 'rmse', то тут же ее 
                # посчитать, чтобы не высчитывать повторно каждую итерацию mse
                metric_value = self.dict_metric[self.metric](y, y_pred)

            # Печать лога на первой итерации и на каждой verbose-й итерации
            if verbose and (i == 0 or i % verbose == 0):
                if i == 0:
                    log = f'start | loss: {mse:.2f}'
                else:
                    log = f'{i} | loss: {mse:.2f}'
                if metric_value is not None:
                    log += f' | {self.metric}: {metric_value:.2f}'
                print(log)

        # Сохраняем веса внутри класса
        self.weights = weights

        # Финальное предсказание и вычисление метрики с новыми весами
        final_y_pred = X @ self.weights
        if self.metric in self.dict_metric:
            self.best_score = self.dict_metric[self.metric](y, final_y_pred)
        else:
            self.best_score = mse

    def get_coef(self) -> np.array:
        """
        Возвращает веса модели без значения для свободного члена.

        Returns
        -------
        np.array
            Массив коэффициентов.
        """

        # Возвращаем без 1 значения, тк оно соответствует фиктивной единичке
        return self.weights[1:]
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Выдает предсказания для новых данных на основе обученной модели

        Returns
        -------
        pd.DataFrame
            Вектор предсказаний
        """
        
        # Добавляем единичный столбец для свободного коэфициента
        X.insert(0, 'bias', np.ones(X.shape[0]))

        # Делаем предсказани: y = X * W
        predict_vector = X @ self.weights

        return predict_vector
    
    @staticmethod
    def mse(y: pd.Series, y_pred: pd.Series) -> float:
        """
        Вычисляет среднеквадратичную ошибку
        """
        return ((y_pred - y) ** 2).mean()

    @staticmethod
    def mae(y: pd.Series, y_pred: pd.Series) -> float:
        """
        Вычисляет среднюю абсолютную ошибку
        """
        return np.abs(y-y_pred).mean()
    
    @staticmethod
    def rmse(y: pd.Series, y_pred: pd.Series) -> float:
        """
        Вычисляет корень из среднеквадратичной ошибки (RMSE)
        """
        return np.sqrt(((y_pred - y) ** 2).mean())
    
    @staticmethod
    def mape(y: pd.Series, y_pred: pd.Series) -> float:
        """
        Вычисляет среднюю абсолютную процентную ошибку (MAPE)
        """
        return 100 * np.abs((y-y_pred)/y).mean()
    
    @staticmethod
    def r2(y: pd.Series, y_pred: pd.Series) -> float:
        """
        Вычисляет коэффициент детерминации (R^2)
        """
        ss_res = np.sum((y - y_pred) ** 2)        # Сумма квадратов остатков (Sum of Squares of Residuals)
        ss_tot = np.sum((y - np.mean(y)) ** 2)    # Сумма квадратов отклонений от среднего (Total Sum of Squares)
        return 1 - (ss_res / ss_tot)

    def l1_reg(self, weights: np.array) -> tuple:
        """
        Вычисляет регуляризацию L1
        """
        loss_add, grad_add = 0, 0
        if self.l1_coef > 0:
            loss_add = self.l1_coef * np.sum(abs(weights))
            grad_add = self.l1_coef * np.sign(weights)
        return loss_add, grad_add
    
    def l2_reg(self, weights: np.array) -> tuple:
        """
        Вычисляет регуляризацию L2
        """
        loss_add, grad_add = 0, 0
        if self.l2_coef > 0:
            loss_add = self.l2_coef * weights ** 2
            grad_add = self.l2_coef * 2 * weights
        return loss_add, grad_add
    
    def elasticnet_reg(self, weights: np.array) -> tuple:
        """
        Вычисляет регуляризацию ElasticNet (смешанный L1 и L2)
        """
        loss_add, grad_add = 0, 0
        if self.l1_coef > 0:
            l1_loss, grad1_loss = self.l1_reg(weights)
            loss_add += l1_loss
            grad_add += grad1_loss
        if self.l2_coef > 0:
            l2_loss, grad2_loss = self.l2_reg(weights)
            loss_add += l2_loss
            grad_add += grad2_loss
        return loss_add, grad_add

    def get_best_score(self):
        """
        Возвращает последнее значение метрики уже полностью обученной модели
        """
        return self.best_score


if __name__ == '__main__':
    """
    Пример использования класса MyLineReg.

    Этот блок выполняется при запуске скрипта напрямую.
    1. Создается синтетический набор данных с помощью make_regression из sklearn.
    2. Инициализируется и обучается модель линейной регрессии MyLineReg.
    3. Выводятся коэффициенты модели и предсказания для нового набора данных.
    """    
    # Создаем датасет
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]

    X1, y1 = make_regression(n_samples=400, n_features=14, n_informative=5, noise=5, random_state=42)
    X1 = pd.DataFrame(X1)
    y1 = pd.Series(y1)
    X1.columns = [f'col_{col}' for col in X1.columns]

    # Проверяем работоспособность
    linreg = MyLineReg(100, 0.05, 'rmse')
    print(linreg)
    linreg.fit(X, y, 20)
    print(np.sum(linreg.get_coef()))
    print(np.sum(linreg.predict(X1)))
