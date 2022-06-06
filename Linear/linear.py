from this import s
import numpy as np
from Util.features import prepare_fro_training

class LinearRegression:
    """
    1. 预处理函数
    2. 初始化参数矩阵
    """
    def __init__(self, data, lables, polynomial_degree=0, sinusoid_degree=0, normalize_data=True) -> None:
        (data_processed, features_mean, features_deviation) = prepare_fro_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        self.lables = lables

        # 初始化参数矩阵 
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))
    
    def train(self, alpha, num = 5000):
        cost_history = self.gradient_descent(alpha, num)
        return cost_history, self.theta

    def gradient_descent(self, alpha, num = 5000):
        cost_history = []
        for _ in range(num):
            self.gradient_setp(alpha)
            cost_history.append(self.cost_function())
        return cost_history



    def gradient_setp(self, alpha):
        """
        梯度下降函数实现
        1. 获取误差
        2. 更新参数
        """
        calculation = self.calculation() - self.lables
        theta = self.theta
        self.theta = theta - (alpha / self.data.shape[1]) * np.dot(calculation.T, self.data)
        

    def calculation(self):
        return np.dot(self.data, self.theta)

    def cost_function(self):
        calulation = LinearRegression.calculation(self.data, self.theta)
        return np.dot(calulation.T, calulation) / 2

    def reckon(self, data):
        """
        fdas
        1. as
        2. as
        """
         data_processed = prepare_fro_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
         ca =  self.calculation()
         cost = self.cost_function()
         return (ca, cost)