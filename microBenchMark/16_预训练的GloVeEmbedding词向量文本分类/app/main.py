import paddle
from paddle.io import Dataset
import numpy as np
import paddle.text as text
import random

print(paddle.__version__)

print('自然语言相关数据集：', paddle.text.__all__)

imdb_train = text.Imdb(mode='train', cutoff=150)
imdb_test = text.Imdb(mode='test', cutoff=150)

print("训练集样本数量: %d; 测试集样本数量: %d" % (len(imdb_train), len(imdb_test)))
print(f"样本标签: {set(imdb_train.labels)}")
print(f"样本字典: {list(imdb_train.word_idx.items())[:10]}")
print(f"单个样本: {imdb_train.docs[0]}")
print(f"最小样本长度: {min([len(x) for x in imdb_train.docs])};最大样本长度: {max([len(x) for x in imdb_train.docs])}")

shuffle_index = list(range(len(imdb_train)))
random.shuffle(shuffle_index)
train_x = [imdb_train.docs[i] for i in shuffle_index]
train_y = [imdb_train.labels[i] for i in shuffle_index]

test_x = imdb_test.docs
test_y = imdb_test.labels

def vectorizer(input, label=None, length=2000):
    if label is not None:
        for x, y in zip(input, label):
            yield np.array((x + [0]*length)[:length]).astype('int64'), np.array([y]).astype('int64')
    else:
        for x in input:
            yield np.array((x + [0]*length)[:length]).astype('int64')

glove_path = "C:\dataset\glove.6B\glove.6B/glove.6B.100d.txt"
embeddings = {}

# 使用utf8编码解码
with open(glove_path, encoding='utf-8') as gf:
    line = gf.readline()
    print("GloVe单行数据：'%s'" % line)

with open(glove_path, encoding='utf-8') as gf:
    for glove in gf:
        word, embedding = glove.split(maxsplit=1)
        embedding = [float(s) for s in embedding.split(' ')]
        embeddings[word] = embedding
print("预训练词向量总数：%d" % len(embeddings))
print(f"单词'the'的向量是：{embeddings['the']}")

word_idx = imdb_train.word_idx
vocab = [w for w in word_idx.keys()]
print(f"词表的前5个单词：{vocab[:5]}")
print(f"词表的后5个单词：{vocab[-5:]}")

# 定义词向量的维度，注意与预训练词向量保持一致
dim = 100

vocab_embeddings = np.zeros((len(vocab), dim))
for ind, word in enumerate(vocab):
    if word != '<unk>':
        word = word.decode()
    embedding = embeddings.get(word, np.zeros((dim,)))
    vocab_embeddings[ind, :] = embedding

pretrained_attr = paddle.ParamAttr(name='embedding',
                                   initializer=paddle.nn.initializer.Assign(vocab_embeddings),
                                   trainable=False)
embedding_layer = paddle.nn.Embedding(num_embeddings=len(vocab),
                                      embedding_dim=dim,
                                      padding_idx=word_idx['<unk>'],
                                      weight_attr=pretrained_attr)

def cal_output_shape(input_shape, out_channels, kernel_size, stride, padding=0, dilation=1):
    return out_channels, int((input_shape + 2*padding - (dilation*(kernel_size - 1) + 1)) / stride) + 1


# 定义每个样本的长度
length = 2000

# 定义卷积层参数
kernel_size = 5
out_channels = 10
stride = 2
padding = 0

output_shape = cal_output_shape(length, out_channels, kernel_size, stride, padding)
output_shape = cal_output_shape(output_shape[1], output_shape[0], 2, 2, 0)
sim_model = paddle.nn.Sequential(embedding_layer,
                             paddle.nn.Conv1D(in_channels=dim, out_channels=out_channels, kernel_size=kernel_size,
                                              stride=stride, padding=padding, data_format='NLC', bias_attr=True),
                             paddle.nn.ReLU(),
                             paddle.nn.MaxPool1D(kernel_size=2, stride=2),
                             paddle.nn.Flatten(),
                             paddle.nn.Linear(in_features=np.prod(output_shape), out_features=2, bias_attr=True),
                             paddle.nn.Softmax())

paddle.summary(sim_model, input_size=(-1, length), dtypes='int64')


class DataReader(Dataset):
    def __init__(self, input, label, length):
        self.data = list(vectorizer(input, label, length=length))

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


# 定义输入格式
input_form = paddle.static.InputSpec(shape=[None, length], dtype='int64', name='input')
label_form = paddle.static.InputSpec(shape=[None, 1], dtype='int64', name='label')

model = paddle.Model(sim_model, input_form, label_form)
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              loss=paddle.nn.loss.CrossEntropyLoss(),
              metrics=paddle.metric.Accuracy())

# 分割训练集和验证集
eval_length = int(len(train_x) * 1 / 4)
model.fit(train_data=DataReader(train_x[:-eval_length], train_y[:-eval_length], length),
          eval_data=DataReader(train_x[-eval_length:], train_y[-eval_length:], length),
          batch_size=32, epochs=10, verbose=1)

# 评估
model.evaluate(eval_data=DataReader(test_x, test_y, length), batch_size=32, verbose=1)

# 预测
true_y = test_y[100:105] + test_y[-110:-105]
pred_y = model.predict(DataReader(test_x[100:105] + test_x[-110:-105], None, length), batch_size=1)
test_x_doc = test_x[100:105] + test_x[-110:-105]

# 标签编码转文字
label_id2text = {0: 'positive', 1: 'negative'}

for index, y in enumerate(pred_y[0]):
    print("原文本：%s" % ' '.join([vocab[i].decode() for i in test_x_doc[index] if i < len(vocab) - 1]))
    print("预测的标签是：%s, 实际标签是：%s" % (label_id2text[np.argmax(y)], label_id2text[true_y[index]]))
