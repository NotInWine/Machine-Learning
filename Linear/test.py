import imp
import numpy as np
import pandas as pd
# 系统库
import os
import matplotlib.pyplot as plt

from linear import LinearRegression

# 导入数据
file_patch = os.path.join(os.path.dirname(__file__), 'Appendix_2_Data_for_Figure_2.1.csv')
data = pd.read_csv(file_patch)
# 得到测试数据, 和训练数据
train_data= data.sample(frac=0.75)
test_data = data.drop(train_data.index)

y_title = 'Happiness score'
x_title = 'Explained by: GDP per capita'

label_data = train_data[[y_title]].to_numpy()
x_data = train_data[[x_title]].to_numpy()

linear = LinearRegression(x_data, label_data)

cost_history, theta = linear.train(0.0001, 2500)

print('损失结果', cost_history[len(cost_history) - 1], ' 权重', theta)

# 损失计算过程图
index = list(range(len(cost_history)))
plt.plot(index, cost_history, label = 'loss:' + str(cost_history[len(cost_history) - 1]))
#plt.show()


xl = test_data[x_title]

# 进行预测 获取损失
train = linear.reckon(test_data[[x_title]].to_numpy())
cost = linear.get_cost(test_data[[x_title]].to_numpy(), test_data[[y_title]].to_numpy())
print('最终损失值 ', cost)


#### 多维度
x_title = 'Explained by: GDP per capita'
x1_title = 'Explained by: Freedom to make life choices'
x2_title = 'Explained by: Healthy life expectancy'

x_data = train_data[[x_title]].to_numpy()
x1_data = train_data[[x1_title]].to_numpy()
x2_data = train_data[[x2_title]].to_numpy()

x_data = np.hstack((x_data, x1_data, x2_data))
linear = LinearRegression(x_data, label_data)

cost_history, theta = linear.train(0.0001, 2500)
print('多维度损失结果', cost_history[len(cost_history) - 1], ' 权重', theta)

# 损失计算过程图
index = list(range(len(cost_history)))

cost = linear.get_cost(
     np.hstack((test_data[[x_title]].to_numpy(), test_data[[x1_title]].to_numpy(), test_data[[x2_title]].to_numpy())),
     test_data[[y_title]].to_numpy()
)
print('最终损失值 ', cost)

plt.plot(index, cost_history, label="m-loss:" + str(cost_history[len(cost_history) - 1]))
plt.legend()
plt.show()


# 画预测线
## 画散点
plt.scatter(xl, test_data[y_title], label='test')
plt.plot(xl, train, color = 'red', label='line')
plt.legend()
plt.show()