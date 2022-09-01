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
all_data= data.sample(frac=1)
train_data= data.sample(frac=0.6)
test_data = data.drop(train_data.index)


# 结果
y_title = 'Happiness score'

# 特征
x_title = 'Explained by: GDP per capita'
x1_title = 'Explained by: Healthy life expectancy'
x_data = train_data[[x_title]].to_numpy()
x1_data = train_data[[x1_title]].to_numpy()

# 建立模型 进行学习
linear = LinearRegression(np.hstack((x_data, x1_data)), train_data[[y_title]].to_numpy())
cost_history, theta = linear.train(0.001, 300)

print('多维度损失结果', cost_history[len(cost_history) - 1], ' 权重\n', theta)

# 检验
cost = linear.get_cost(
     np.hstack((test_data[[x_title]].to_numpy(), test_data[[x1_title]].to_numpy())),
     test_data[[y_title]].to_numpy()
)
print('最终损失值 ', cost)

# 损失过程图
index = list(range(len(cost_history)))
plt.plot(index, cost_history, label="0.001m-loss:" + str(cost))


# 画预测线
# 训练集
plot_training_trace = go.Scatter3d(
    x=train_data[[x_title]].to_numpy().flatten(),
    y=train_data[[x1_title]].to_numpy().flatten(),
    z=train_data[[y_title]].to_numpy().flatten(),
    name="Train",
    mode="markers",
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'width': 1
        }
    }
)

# 测试集
plot_test_trace = go.Scatter3d(
    x=test_data[[x_title]].to_numpy().flatten(),
    y=test_data[[x1_title]].to_numpy().flatten(),
    z=test_data[[y_title]].to_numpy().flatten(),
    name="Test",
    mode="markers",
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'width': 1
        }
    }
)


# 布局
plot_layout = go.Layout(
    title = 'Data',
    scene = {
        'xaxis': {'title': x_title},
        'yaxis': {'title': x1_title},
        'zaxis': {'title': y_title}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

# 生成 10*10 底面的所有点
predictions_num = 10

x_min = all_data[[x_title]].min()
x_max = all_data[[x_title]].max()

x1_min = all_data[[x1_title]].min()
x1_max = all_data[[x1_title]].max()

x_axis = np.linspace(x_min, x_max, predictions_num)
x1_axis = np.linspace(x1_min, x1_max, predictions_num)

x_predictions = np.zeros((predictions_num * predictions_num, 1))
x1_predictions = np.zeros((predictions_num * predictions_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(x1_axis):
        x_predictions[x_y_index] = x_value
        x1_predictions[x_y_index] = y_value
        x_y_index += 1

# 线性回归得到Z轴的点
z_predictions = linear.reckon(np.hstack((x_predictions, x1_predictions)))

# 绘图
plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=x1_predictions.flatten(),
    z=z_predictions.flatten(),
    name='section',
    marker={
        'size': 1,
    },
    opacity=0.618,
    surfaceaxis=2
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)


plt.legend()
plt.show()