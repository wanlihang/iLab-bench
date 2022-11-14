# 导入项目运行所需的包
import random

import numpy as np
import paddle
import paddle.nn as nn
from visualdl import LogWriter

# 训练轮数
EPOCH = 2


# 编码函数
def encoder(text, LEN, label_dict):
    # 文本转ID
    ids = [label_dict[word] for word in text]
    # 对长度进行补齐
    ids += [label_dict[' ']] * (LEN - len(ids))
    return ids


# 单个数据生成函数
def make_data(inputs, labels, DIGITS, label_dict):
    MAXLEN = DIGITS + 1 + DIGITS
    # 对输入输出文本进行ID编码
    inputs = encoder(inputs, MAXLEN, label_dict)
    labels = encoder(labels, DIGITS + 1, label_dict)
    return inputs, labels


# 批量数据生成函数
def gen_datas(DATA_NUM, MAX_NUM, DIGITS, label_dict):
    datas = []
    while len(datas) < DATA_NUM:
        # 随机取两个数
        a = random.randint(0, MAX_NUM)
        b = random.randint(0, MAX_NUM)
        # 生成输入文本
        inputs = '%d+%d' % (a, b)
        # 生成输出文本
        labels = str(eval(inputs))
        # 生成单个数据
        inputs, labels = [np.array(_).astype('int64') for _ in make_data(inputs, labels, DIGITS, label_dict)]
        datas.append([inputs, labels])
    return datas


# 继承paddle.io.Dataset来构造数据集
class Addition_Dataset(paddle.io.Dataset):
    # 重写数据集初始化函数
    def __init__(self, datas):
        super(Addition_Dataset, self).__init__()
        self.datas = datas

    # 重写生成样本的函数
    def __getitem__(self, index):
        data, label = [paddle.to_tensor(_) for _ in self.datas[index]]
        return data, label

    # 重写返回数据集大小的函数
    def __len__(self):
        return len(self.datas)


print('generating datas..')

# 定义字符表
label_dict = {
    '0': 0, '1': 1, '2': 2, '3': 3,
    '4': 4, '5': 5, '6': 6, '7': 7,
    '8': 8, '9': 9, '+': 10, ' ': 11
}

# 输入数字最大位数
DIGITS = 2

# 数据数量
train_num = 5000
dev_num = 500

# 数据批大小
batch_size = 32

# 读取线程数
num_workers = 8

# 定义一些所需变量
MAXLEN = DIGITS + 1 + DIGITS
MAX_NUM = 10 ** (DIGITS) - 1

# 生成数据
train_datas = gen_datas(
    train_num,
    MAX_NUM,
    DIGITS,
    label_dict
)
dev_datas = gen_datas(
    dev_num,
    MAX_NUM,
    DIGITS,
    label_dict
)

# 实例化数据集
train_dataset = Addition_Dataset(train_datas)
dev_dataset = Addition_Dataset(dev_datas)

print('making the dataset...')

# 实例化数据读取器
train_reader = paddle.io.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
dev_reader = paddle.io.DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)

print('finish')


# 继承paddle.nn.Layer类
class Addition_Model(nn.Layer):
    # 重写初始化函数
    # 参数：字符表长度、嵌入层大小、隐藏层大小、解码器层数、处理数字的最大位数
    def __init__(self, char_len=12, embedding_size=128, hidden_size=128, num_layers=1, DIGITS=2):
        super(Addition_Model, self).__init__()
        # 初始化变量
        self.DIGITS = DIGITS
        self.MAXLEN = DIGITS + 1 + DIGITS
        self.hidden_size = hidden_size
        self.char_len = char_len

        # 嵌入层
        self.emb = nn.Embedding(
            char_len,
            embedding_size
        )

        # 编码器
        self.encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1
        )

        # 解码器
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        # 全连接层
        self.fc = nn.Linear(
            hidden_size,
            char_len
        )

    # 重写模型前向计算函数
    # 参数：输入[None, MAXLEN]、标签[None, DIGITS + 1]
    def forward(self, inputs, labels=None):
        # 嵌入层
        out = self.emb(inputs)

        # 编码器
        out, (_, _) = self.encoder(out)

        # 按时间步切分编码器输出
        out = paddle.split(out, self.MAXLEN, axis=1)

        # 取最后一个时间步的输出并复制 DIGITS + 1 次
        out = paddle.expand(out[-1], [out[-1].shape[0], self.DIGITS + 1, self.hidden_size])

        # 解码器
        out, (_, _) = self.decoder(out)

        # 全连接
        out = self.fc(out)

        # 如果标签存在，则计算其损失和准确率
        if labels is not None:
            # 计算交叉熵损失
            loss = nn.functional.cross_entropy(out, labels)

            # 计算准确率
            acc = paddle.metric.accuracy(paddle.reshape(out, [-1, self.char_len]), paddle.reshape(labels, [-1, 1]))

            # 返回损失和准确率
            return loss, acc

        # 返回输出
        return out


# 继承paddle.nn.Layer类
class Addition_Model(nn.Layer):
    # 重写初始化函数
    # 参数：字符表长度、嵌入层大小、隐藏层大小、解码器层数、处理数字的最大位数
    def __init__(self, char_len=12, embedding_size=128, hidden_size=128, num_layers=1, DIGITS=2):
        super(Addition_Model, self).__init__()
        # 初始化变量
        self.DIGITS = DIGITS
        self.MAXLEN = DIGITS + 1 + DIGITS
        self.hidden_size = hidden_size
        self.char_len = char_len

        # 嵌入层
        self.emb = nn.Embedding(
            char_len,
            embedding_size
        )

        # 编码器
        self.encoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1
        )

        # 解码器
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )

        # 全连接层
        self.fc = nn.Linear(
            hidden_size,
            char_len
        )

    # 重写模型前向计算函数
    # 参数：输入[None, MAXLEN]、标签[None, DIGITS + 1]
    def forward(self, inputs, labels=None):
        # 嵌入层
        out = self.emb(inputs)

        # 编码器
        out, (_, _) = self.encoder(out)

        # 按时间步切分编码器输出
        out = paddle.split(out, self.MAXLEN, axis=1)

        # 取最后一个时间步的输出并复制 DIGITS + 1 次
        out = paddle.expand(out[-1], [out[-1].shape[0], self.DIGITS + 1, self.hidden_size])

        # 解码器
        out, (_, _) = self.decoder(out)

        # 全连接
        out = self.fc(out)

        # 如果标签存在，则计算其损失和准确率
        if labels is not None:
            # 计算交叉熵损失
            loss = nn.functional.cross_entropy(out, labels)

            # 计算准确率
            acc = paddle.metric.accuracy(paddle.reshape(out, [-1, self.char_len]), paddle.reshape(labels, [-1, 1]))

            # 返回损失和准确率
            return loss, acc

        # 返回输出
        return out


# 初始化log写入器
log_writer = LogWriter(logdir="./log")

# 模型参数设置
embedding_size = 128
hidden_size = 128
num_layers = 1

# 训练参数设置
learning_rate = 0.001
log_iter = 2000
eval_iter = 500

# 定义一些所需变量
global_step = 0
log_step = 0
max_acc = 0

# 实例化模型
model = Addition_Model(
    char_len=len(label_dict),
    embedding_size=embedding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    DIGITS=DIGITS)

# 将模型设置为训练模式
model.train()

# 设置优化器，学习率，并且把模型参数给优化器
opt = paddle.optimizer.Adam(
    learning_rate=learning_rate,
    parameters=model.parameters()
)

# 启动训练，循环epoch_num个轮次
for epoch in range(EPOCH):
    # 遍历数据集读取数据
    for batch_id, data in enumerate(train_reader()):
        # 读取数据
        inputs, labels = data

        # 模型前向计算
        loss, acc = model(inputs, labels=labels)

        # 打印训练数据
        if global_step % log_iter == 0:
            print('train epoch:%d step: %d loss:%f acc:%f' % (epoch, global_step, loss.numpy(), acc.numpy()))
            log_writer.add_scalar(tag="train/loss", step=log_step, value=loss.numpy())
            log_writer.add_scalar(tag="train/acc", step=log_step, value=acc.numpy())
            log_step += 1

        # 模型验证
        if global_step % eval_iter == 0:
            model.eval()
            losses = []
            accs = []
            for data in dev_reader():
                loss_eval, acc_eval = model(inputs, labels=labels)
                losses.append(loss_eval.numpy())
                accs.append(acc_eval.numpy())
            avg_loss = np.concatenate(losses).mean()
            avg_acc = np.concatenate(accs).mean()
            print('eval epoch:%d step: %d loss:%f acc:%f' % (epoch, global_step, avg_loss, avg_acc))
            log_writer.add_scalar(tag="dev/loss", step=log_step, value=avg_loss)
            log_writer.add_scalar(tag="dev/acc", step=log_step, value=avg_acc)

            # 保存最佳模型
            if avg_acc > max_acc:
                max_acc = avg_acc
                print('saving the best_model...')
                paddle.save(model.state_dict(), 'best_model')
            model.train()

        # 反向传播
        loss.backward()

        # 使用优化器进行参数优化
        opt.step()

        # 清除梯度
        opt.clear_grad()

        # 全局步数加一
        global_step += 1

# 保存最终模型
paddle.save(model.state_dict(), 'final_model')

# 反转字符表
label_dict_adv = {v: k for k, v in label_dict.items()}

# 输入计算题目
input_text = '12+40'

# 编码输入为ID
inputs = encoder(input_text, MAXLEN, label_dict)

# 转换输入为向量形式
inputs = np.array(inputs).reshape(-1, MAXLEN)
inputs = paddle.to_tensor(inputs)

# 加载模型
params_dict = paddle.load('best_model')
model.set_dict(params_dict)

# 设置为评估模式
model.eval()

# 模型推理
out = model(inputs)

# 结果转换
result = ''.join([label_dict_adv[_] for _ in np.argmax(out.numpy(), -1).reshape(-1)])

# 打印结果
print('the model answer: %s=%s' % (input_text, result))
print('the true answer: %s=%s' % (input_text, eval(input_text)))
