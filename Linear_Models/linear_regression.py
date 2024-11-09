import pandas as pd
import numpy as np


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

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
        """Функция обучения"""

        # Создаем копию и добавляем единичный столбец слева в матрицу фичей:
        X = pd.concat([pd.Series(1, index=X.index, name='bias'), X], axis=1)

        # Количество фичей и число наблюдений
        features_num = X.shape[1]
        n_samples = X.shape[0]

        # Создаем вектор весов
        weights = np.ones(features_num)

        # Градиентный спуск
        for i in range(self.n_iter):
            # Делаем предсказание (умножаем матрицу фичей на вектор весов)
            y_pred = X @ weights

            # Если включена регуляризация
            mse_add, gradient_mse_add = 0, 0
            if self.reg in self.dict_reg:
                mse_add, gradient_mse_add = self.dict_reg[self.reg](weights)

            # Считаем MSE
            mse = self.mse(y, y_pred) + mse_add

            # Вычисляем градиент
            gradient_mse = 2 / n_samples * (y_pred - y) @ X + gradient_mse_add

            # Обновляем веса
            weights -= self.learning_rate * gradient_mse

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
        """Функция возвращающая веса"""

        # Возвращаем без 1 значения, тк оно соответствует фиктивной единичке
        return self.weights[1:]
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Функция выдающая предсказания"""
        
        # Добавляем единичный столбец для свободного коэфициента
        X.insert(0, 'bias', np.ones(X.shape[0]))

        # Делаем предсказани: y = X * W
        predict_vector = X @ self.weights

        return predict_vector
    
    @staticmethod
    def mse(y: pd.Series, y_pred: pd.Series) -> float:
        return ((y_pred - y) ** 2).mean()

    @staticmethod
    def mae(y: pd.Series, y_pred: pd.Series) -> float:
        return np.abs(y-y_pred).mean()
    
    @staticmethod
    def rmse(y: pd.Series, y_pred: pd.Series) -> float:
        return np.sqrt(((y_pred - y) ** 2).mean())
    
    @staticmethod
    def mape(y: pd.Series, y_pred: pd.Series) -> float:
        return 100 * np.abs((y-y_pred)/y).mean()
    
    @staticmethod
    def r2(y: pd.Series, y_pred: pd.Series) -> float:
        ss_res = np.sum((y - y_pred) ** 2)        # Сумма квадратов остатков (Sum of Squares of Residuals)
        ss_tot = np.sum((y - np.mean(y)) ** 2)    # Сумма квадратов отклонений от среднего (Total Sum of Squares)
        return 1 - (ss_res / ss_tot)

    def l1_reg(self, weights: np.array) -> tuple:
        loss_add, grad_add = 0, 0
        if self.l1_coef > 0:
            loss_add = self.l1_coef * np.sum(abs(weights))
            grad_add = self.l1_coef * np.sign(weights)
        return loss_add, grad_add
    
    def l2_reg(self, weights: np.array) -> tuple:
        loss_add, grad_add = 0, 0
        if self.l2_coef > 0:
            loss_add = self.l2_coef * weights ** 2
            grad_add = self.l2_coef * 2 * weights
        return loss_add, grad_add
    
    def elasticnet_reg(self, weights: np.array) -> tuple:
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

    def get_best_score(self,):
        return self.best_score


if __name__ == '__main__':
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
