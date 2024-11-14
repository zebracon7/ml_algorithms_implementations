import pandas as pd
import numpy as np
import random


class MyLogReg():
    """
    Логистическая регрессия
    
    Параметры
    ----------
    n_iter: int, optional
        Количество шагов градиентного спуска
        Дефолтное значение: 10
    learning_rate: Union[float, Callable], optional
        Коэффициент скорости обучения градиентного спуска
        Если на вход пришла lambda ф-я, то вычисляется на каждом шаге обучения
        Дефолтное значение: 0.1
    metric: str, optional
        Вывод дополнительной метрики в лог обучения
        Возможные значения: accuracy, precision, recall, f1, roc_auc
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
        В случае дробного числа - является долей от всего датасета
        Дефолтное значение: None
    random_state: int
        Сид для воспроизводимости результата
        Дефолтное значение: 42
    """

    def __init__(self, n_iter=10, learning_rate=0.1, metric=None, reg=None, l1_coef=0, l2_coef=0, sgd_sample=None, random_state=42) -> None:
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
            'accuracy': self.accuracy, 
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc
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
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def __repr__(self) -> str:
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False):
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

        # Вычисляем количество наблюдений
        n_samples = X.shape[0]

        # Добавляем единичный столбец слева в матрицу фичей:
        X = X.copy()
        X.insert(0, 'bias', np.ones(n_samples))

        # Определяем количество фичей
        features_num = X.shape[1]

        # Создаем вектор весов
        weights = np.ones(features_num)

        # Инициализируем self.weights до начала цикла
        self.weights = weights.copy()

        # В формуле logloss'a мы вычисляем логарифм, и если аргумент будет равен 0, то даст -inf
        # поэтому, будем добавлять к аргументу логарифма eps = 1e-15
        eps = 1e-15

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
                X_batch = X.iloc[sample_rows_idx]
                y_batch = y.iloc[sample_rows_idx]

            else:
                X_batch = X
                y_batch = y 
                batch_size = n_samples
            # Вызываем predict_proba для получения вероятностей с текущими весами
            y_pred_proba = self.predict_proba(X_batch)

            # Если включена регуляризия
            logloss_add, gradient_logloss_add = 0, 0
            if self.reg in self.dict_reg:
                logloss_add, gradient_logloss_add = self.dict_reg[self.reg](weights)

            # Считаем ошибку (используем натуральный логарифм)
            logloss = -1 * (y_batch * np.log(y_pred_proba + eps) + (1 - y_batch) * np.log(1 - y_pred_proba + eps)).mean() + logloss_add

            # Считаем градиент
            gradient_logloss = ((y_pred_proba - y_batch) @ X_batch) / batch_size + gradient_logloss_add

            # Если learning_rate задан как число
            if isinstance(self.learning_rate, float):
                # Делаем шаг регрессии
                weights -= self.learning_rate * gradient_logloss
            else: 
                weights -= self.learning_rate(i) * gradient_logloss

            # Обновляем веса внутри класса
            self.weights = weights.copy()  

            # После обновления весов, вычисляем новые вероятности предсказаний
            y_pred_proba_new = self.predict_proba(X_batch)

            # Вызываем predict для получения бинарных предсказаний на основе обновленных вероятностей
            y_pred = (y_pred_proba_new > 0.5).astype(int)

            # Считаем доп. метрику для вывода, если задана
            metric_value = None
            if self.metric in self.dict_metric:
                if self.metric == 'roc_auc':
                    metric_value = self.dict_metric[self.metric](y, y_pred_proba_new)
                else:
                    metric_value = self.dict_metric[self.metric](y, y_pred)
                # Сохраняем наилучший результат
                if self.best_score is None or metric_value > self.best_score:
                    self.best_score = metric_value

            # Выводим лог
            if verbose and (i == 0 or i % verbose == 0):
                if i == 0:
                    log = f'start | loss: {logloss:.4f}'
                else:
                    log = f'{i} | loss: {logloss:.4f}'
                if metric_value is not None:
                    log = f'{log} | {self.metric}: {metric_value:.4f}'
                print(log)
        
        # Сохраняем веса внутри класса
        self.weights = weights.copy()
    
    def predict_proba(self, X: pd.DataFrame):
        """
        Возвращает вероятности принадлежности к классу 1 для каждого наблюдения.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков.
        """
        predict_vector = 1 / (1 + np.exp(-X @ self.weights))
        return predict_vector
    
    def predict(self, X: pd.DataFrame):
        """
        Возвращает предсказания классов (0 или 1) для каждого наблюдения.

        Параметры
        ----------
        X : pd.DataFrame
            Матрица признаков.
        """
        predict_vector = self.predict_proba(X)
        # Преобразуем вероятности в классы 0 или 1
        # Если значение меньше или равно 0.5, результатом будет, иначе True
        return (predict_vector > 0.5).astype(int)

    def get_coef(self) -> np.array:
        """
        Возвращает веса модели без значения для свободного члена.

        Возвращает
        ----------
        np.array
            Массив коэффициентов модели.
        """
        return self.weights[1:]
    
    def accuracy(self, y_true: pd.Series, y_pred: pd.Series):
        """
        Вычисляет accuracy модели.

        Параметры
        ----------
        y_true : pd.Series
            Истинные значения классов.
        y_pred : pd.Series
            Предсказанные значения классов.
        """
        correct_predictions = (y_true == y_pred).sum()  # Количество правильных предсказаний
        total_predictions = len(y_true)  # Общее количество предсказаний
        return correct_predictions / total_predictions
    
    def precision(self, y_true: pd.Series, y_pred: pd.Series):
        """ 
        Вычисляет точность (precision) модели.

        Параметры
        ----------
        y_true : pd.Series
            Истинные значения классов.
        y_pred : pd.Series
            Предсказанные значения классов.
        """
        true_positive = ((y_true == 1) & (y_pred == 1)).sum()  # Истинно положительные
        tp_fp = (y_pred == 1).sum()  # Общее количество предсказанных положительных классов
        return true_positive / tp_fp if tp_fp != 0 else 0
    
    def recall(self, y_true: pd.Series, y_pred: pd.Series):
        """ 
        Вычисляет полноту (recall) модели.

        Параметры
        ----------
        y_true : pd.Series
            Истинные значения классов.
        y_pred : pd.Series
            Предсказанные значения классов.
        """
        tp = ((y_true == 1) & (y_pred == 1)).sum()  # Истинно положительные
        fn = ((y_true == 1) & (y_pred == 0)).sum()  # Ложно отрицательные
        tp_fn = tp + fn  # Истинно положительные + Ложно отрицательные
        return tp / tp_fn if tp_fn != 0 else 0
    
    def f1(self, y_true: pd.Series, y_pred: pd.Series):
        """ 
        Вычисляет F1-метрику модели.

        Параметры
        ----------
        y_true : pd.Series
            Истинные значения классов.
        y_pred : pd.Series
            Предсказанные значения классов.
        """
        b = 1  # Для F1-score (b = 1)
        precision = self.precision(y_true, y_pred)
        recall = self.recall(y_true, y_pred)
        f1 = (1 + b**2) * (precision * recall) / (b**2 * precision + recall)
        return f1 if (precision + recall) != 0 else 0
    
    def roc_auc(self, y_true: pd.Series, y_pred_proba: pd.Series):
        """ 
        Вычисляет метрику ROC AUC.

        Параметры
        ----------
        y_true : pd.Series
            Истинные значения классов.
        y_pred_proba : pd.Series
            Предсказанные вероятности классов.
        """
        # Округляем скоры до 10 знака, чтобы избежать проблем с точностью
        y_pred_proba = y_pred_proba.round(10)
        
        # Создаем DataFrame для удобства обработки и сортировки
        df = pd.DataFrame({'y_true': y_true, 'y_pred_proba': y_pred_proba})
        # Сортируем по убыванию предсказанных вероятностей
        df = df.sort_values(by='y_pred_proba', ascending=False).reset_index(drop=True)
        
        # Подсчитываем количество положительных и отрицательных классов
        P = (df['y_true'] == 1).sum()  # Количество положительных классов
        N = (df['y_true'] == 0).sum()  # Количество отрицательных классов

        if P == 0 or N == 0:
            return 0  # Если нет положительных или отрицательных классов, AUC не определен

        # Переменная для хранения итоговой суммы
        auc_sum = 0

        # Количество положительных классов выше текущего отрицательного и с одинаковыми значениями скорингов
        pos_count_above = 0
        pos_count_same_score = 0

        # Проходим по отсортированным данным сверху вниз
        for i in range(len(df)):
            if df.loc[i, 'y_true'] == 1:  # Если текущий класс положительный
                pos_count_above += 1
            else:  # Если текущий класс отрицательный
                # Считаем положительные классы с таким же значением скоров
                pos_count_same_score = ((df['y_pred_proba'] == df.loc[i, 'y_pred_proba']) & (df['y_true'] == 1)).sum()
                # Увеличиваем сумму AUC
                auc_sum += pos_count_above + pos_count_same_score / 2

        # Финальный расчет AUC
        return auc_sum / (P * N)

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