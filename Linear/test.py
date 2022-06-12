import imp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linear import LinearRegression

# 导入数据
data = pd.read_csv('/Users/yc/Movies/pythonCode/Machine-Learning/Linear/Appendix_2_Data_for_Figure_2.1.csv')
print(data)

# 得到测试数据, 和训练数据
train_data= data.sample(frac=0.8)
test_data = data.drop(train_data.index)

y_title = ''
x_title = ''