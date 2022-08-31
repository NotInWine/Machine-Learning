from this import s
import numpy as np
import os
import sys
# 获取当前代码文件绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将需要导入模块代码文件相对于当前文件目录的绝对路径加入到sys.path中
sys.path.append(os.path.join(current_dir, ".."))

from Util.features import prepare_fro_training

class LinearRegression:
    """
    1. 预处理函数
    2. 初始化参数矩阵
    """
    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True) -> None:
        (data_processed, features_mean, features_deviation) = prepare_fro_training.prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data)
        self.data = data_processed
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data
        self.labels = labels

        # 初始化参数矩阵 
        num_features = self.data.shape[1]
        self.theta = np.zeros((num_features, 1))
    
    def train(self, alpha, num = 5000):
        """
        训练方法
        """
        cost_history = self.gradient_descent(alpha, num)
        return cost_history, self.theta

    def gradient_descent(self, alpha, num = 5000):
        """
        执行梯度下降
        """
        cost_history = []
        for _ in range(num):
            self.gradient_setp(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history



    def gradient_setp(self, alpha):
        """
        梯度下降函数实现
        1. 获取误差
        2. 更新参数
        """
        calculation = LinearRegression.calculation(self.data, self.theta) - self.labels
        theta = self.theta
        self.theta = theta - (alpha / self.data.shape[1]) * np.dot(calculation.T, self.data).T
        
    @staticmethod
    def calculation(data, theta):
        """
        预测方法
        """
        return np.dot(data, theta)

    def cost_function(self, data, labels):
        """
        损失函数-最小二乘法
        """
        calulation = LinearRegression.calculation(data, self.theta) - labels
        #return (np.dot(calulation.T, calulation) / 2) [0][0]
        cos = np.dot(calulation.T, calulation)
        return cos [0][0]

    def reckon(self, data):
        """
        fdas
        1. as
        2. as
        """
        (data_processed, features_mean, features_deviation) = prepare_fro_training.prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        ca =  self.calculation(data_processed, self.theta)
        return ca

    def get_cost(self, data, labels):
        (data_processed, features_mean, features_deviation) = prepare_fro_training.prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)
        return self.cost_function(data_processed, labels)
