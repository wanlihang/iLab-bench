# 导入 paddle
import sys
import warnings

import numpy as np
import paddle
import paddle.nn.functional as F
import pandas as pd
import tqdm
from matplotlib import pyplot as plt

# 训练轮数
EPOCH = 20

warnings.filterwarnings("ignore")

# 正常数据预览
df_small_noise_path = './data/art_daily_small_noise.csv'
df_small_noise = pd.read_csv(
    df_small_noise_path, parse_dates=True, index_col="timestamp"
)

# 异常数据预览
df_daily_jumpsup_path = './data/art_daily_jumpsup.csv'
df_daily_jumpsup = pd.read_csv(
    df_daily_jumpsup_path, parse_dates=True, index_col="timestamp"
)
print(df_small_noise.head())

print(df_daily_jumpsup.head())

# 正常的时序数据可视化
fig, ax = plt.subplots()
df_small_noise.plot(legend=False, ax=ax)
plt.show()

# 异常的时序数据可视化
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
plt.show()

# 初始化并保存得到的均值和方差，用于初始化数据。
training_mean = df_small_noise.mean()
training_std = df_small_noise.std()
df_training_value = (df_small_noise - training_mean) / training_std
print("训练数据总量:", len(df_training_value))

# 时序步长
TIME_STEPS = 288


class MyDataset(paddle.io.Dataset):
    """
    步骤一：继承paddle.io.Dataset类
    """

    def __init__(self, data, time_steps):
        """
        步骤二：实现构造函数，定义数据读取方式，划分训练和测试数据集
        注意：这个是不需要label
        """
        super(MyDataset, self).__init__()
        self.time_steps = time_steps
        self.data = paddle.to_tensor(self.transform(data), dtype='float32')

    def transform(self, data):
        '''
        构造时序数据
        '''
        output = []
        for i in range(len(data) - self.time_steps):
            output.append(np.reshape(data[i: (i + self.time_steps)], (1, self.time_steps)))
        return np.stack(output)

    def __getitem__(self, index):
        """
        步骤三：实现__getitem__方法，定义指定index时如何获取数据，并返回单条数据（训练数据）
        """
        data = self.data[index]
        label = self.data[index]
        return data, label

    def __len__(self):
        """
        步骤四：实现__len__方法，返回数据集总数目
        """
        return len(self.data)


# 实例化数据集
train_dataset = MyDataset(df_training_value.values, TIME_STEPS)


class AutoEncoder(paddle.nn.Layer):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv0 = paddle.nn.Conv1D(in_channels=1, out_channels=32, kernel_size=7, stride=2)
        self.conv1 = paddle.nn.Conv1D(in_channels=32, out_channels=16, kernel_size=7, stride=2)
        self.convT0 = paddle.nn.Conv1DTranspose(in_channels=16, out_channels=32, kernel_size=7, stride=2)
        self.convT1 = paddle.nn.Conv1DTranspose(in_channels=32, out_channels=1, kernel_size=7, stride=2)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.convT0(x)
        x = F.relu(x)
        x = F.dropout(x, 0.2)
        x = self.convT1(x)
        return x


# 参数设置
batch_size = 128
learning_rate = 0.001


def train():
    print('训练开始')
    # 实例化模型
    model = AutoEncoder()
    # 将模型转换为训练模式
    model.train()
    # 设置优化器，学习率，并且把模型参数给优化器
    opt = paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters())
    # 设置损失函数
    mse_loss = paddle.nn.MSELoss()
    # 设置数据读取器
    data_reader = paddle.io.DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last=True)
    history_loss = []
    iter_epoch = []
    for epoch in tqdm.tqdm(range(EPOCH)):
        for batch_id, data in enumerate(data_reader()):
            x = data[0]
            y = data[1]
            out = model(x)
            avg_loss = mse_loss(out, (y[:, :, :-1]))  # 输入的数据经过卷积会丢掉最后一个数据
            avg_loss.backward()
            opt.step()
            opt.clear_grad()
        iter_epoch.append(epoch)
        history_loss.append(avg_loss.numpy()[0])
    # 绘制loss
    plt.plot(iter_epoch, history_loss, label='loss')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel('Loss')
    plt.show()
    # 保存模型参数
    paddle.save(model.state_dict(), 'model')


train()

# 计算阀值

param_dict = paddle.load('model')  # 读取保存的参数
model = AutoEncoder()
model.load_dict(param_dict)  # 加载参数
model.eval()  # 预测
total_loss = []
datas = []
# 预测所有正常时序
mse_loss = paddle.nn.loss.MSELoss()
# 这里设置batch_size为1，单独求得每个数据的loss
data_reader = paddle.io.DataLoader(train_dataset,
                                   places=[paddle.CPUPlace()],
                                   batch_size=1,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=0)
for batch_id, data in enumerate(data_reader()):
    x = data[0]
    y = data[1]
    out = model(x)
    avg_loss = mse_loss(out, (y[:, :, :-1]))
    total_loss.append(avg_loss.numpy()[0])
    datas.append(batch_id)

plt.bar(datas, total_loss)
plt.ylabel("reconstruction loss")
plt.xlabel("data samples")
plt.show()

# 获取重建loss的阀值
threshold = np.max(total_loss)
print("阀值:", threshold)

param_dict = paddle.load('model')  # 读取保存的参数
model = AutoEncoder()
model.load_dict(param_dict)  # 加载参数
model.eval()  # 预测
data_reader = paddle.io.DataLoader(train_dataset,
                                   places=[paddle.CPUPlace()],
                                   batch_size=128,
                                   shuffle=False,
                                   drop_last=False,
                                   num_workers=0)
for batch_id, data in enumerate(data_reader()):
    x = data[0]
    out = model(x)
    step = np.arange(287)
    plt.plot(step, x[0, 0, :-1].numpy())
    plt.plot(step, out[0, 0].numpy())
    plt.show()
    sys.exit

df_test_value = (df_daily_jumpsup - training_mean) / training_std
fig, ax = plt.subplots()
df_test_value.plot(legend=False, ax=ax)
plt.show()
# 这是测试集里面的异常数据，可以看到第11~~12天发生了异常


# 探测异常数据
threshold = 0.033  # 阀值设定，即刚才求得的值
param_dict = paddle.load('model')  # 读取保存的参数
model = AutoEncoder()
model.load_dict(param_dict)  # 加载参数
model.eval()  # 预测
mse_loss = paddle.nn.loss.MSELoss()


def create_sequences(values, time_steps=288):
    '''
    探测数据预处理
    '''
    output = []
    for i in range(len(values) - time_steps):
        output.append(values[i: (i + time_steps)])
    return np.stack(output)


x_test = create_sequences(df_test_value.values)
x = paddle.to_tensor(x_test).astype('float32')

abnormal_index = []  # 记录检测到异常时数据的索引

for i in range(len(x_test)):
    input_x = paddle.reshape(x[i], (1, 1, 288))
    out = model(input_x)
    loss = mse_loss(input_x[:, :, :-1], out)
    if loss.numpy()[0] > threshold:
        # 开始检测到异常时序列末端靠近异常点，所以要加上序列长度，得到真实索引位置
        abnormal_index.append(i + 288)

# 不再检测异常时序列的前端靠近异常点，所以要减去索引长度得到异常点真实索引，为了结果明显，给异常位置加宽40单位
abnormal_index = abnormal_index[:(-288 + 40)]
print(len(abnormal_index))
print(abnormal_index)

# 异常检测结果可视化
df_subset = df_daily_jumpsup.iloc[abnormal_index]
fig, ax = plt.subplots()
df_daily_jumpsup.plot(legend=False, ax=ax)
df_subset.plot(legend=False, ax=ax, color="r")
plt.show()
