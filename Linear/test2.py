import imp
import numpy as np
# 表格解析
import pandas as pd
# 系统库
import os
import matplotlib.pyplot as plt
# 3d绘图
import plotly as plotly
import plotly.graph_objs as go

# 线性回归类
from linear import LinearRegression

# 读取数据
file_patch = os.path.join(os.path.dirname(__file__), 'Appendix_2_Data_for_Figure_2.1.csv')
data = pd.read_csv(file_patch)

# 得到测试数据, 和训练数据
train_data= data.sample(frac=0.8)
test_data = data.drop(train_data.index)

y_title = 'Happiness score'
x_title = 'Explained by: Social support'

label_data = train_data[[y_title]].to_numpy()
x_data = train_data[[x_title]].to_numpy()

linear = LinearRegression(x_data, label_data)

cost_history, theta = linear.train(0.0001, 2500)

print('损失结果', cost_history[len(cost_history) - 1], ' 权重\b', theta)

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
x1_title = 'Explained by: Social support'

x_data = train_data[[x_title]].to_numpy()
x1_data = train_data[[x1_title]].to_numpy()

x_data = np.hstack((x_data, x1_data))
linear = LinearRegression(x_data, label_data)

cost_history, theta = linear.train(0.0001, 2500)
print('多维度损失结果', cost_history[len(cost_history) - 1], ' 权重\n', theta)

# 损失计算过程图
index = list(range(len(cost_history)))

cost = linear.get_cost(
     np.hstack((test_data[[x_title]].to_numpy(), test_data[[x1_title]].to_numpy())),
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

# 训练集
plot_training_trace = go.Scatter3d(
    x=train_data[[x_title]].to_numpy().flatten(),
    y=train_data[[x1_title]].to_numpy().flatten(),
    z=train_data[[y_title]].to_numpy().flatten(),
    name="Training Set",
    mode="markers",
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        }
    }
)

# 测试集
plot_test_trace = go.Scatter3d(
    x=test_data[[x_title]].to_numpy().flatten(),
    y=test_data[[x1_title]].to_numpy().flatten(),
    z=test_data[[y_title]].to_numpy().flatten(),
    name="Test Set",
    mode="markers",
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        }
    }
)



# 布局
plot_layout = go.Layout(
    title = 'Date Sets',
    scene = {
        'xaxis': {'title': x_title},
        'yaxis': {'title': x1_title},
        'zaxis': {'title': y_title}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

predictions_num = 10

x_min = test_data[[x_title]].min()
x_max = test_data[[x_title]].max()

y_min = test_data[[x1_title]].min()
y_max = test_data[[x1_title]].max()

x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))


x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear.reckon(np.hstack((x_predictions, y_predictions)))

plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)
