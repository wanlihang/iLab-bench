import paddle
import numpy as np
print(paddle.__version__)

print('loading dataset...')
train_dataset = paddle.text.datasets.Imdb(mode='train')
test_dataset = paddle.text.datasets.Imdb(mode='test')
print('loading finished')

word_dict = train_dataset.word_idx

# add a pad token to the dict for later padding the sequence
word_dict['<pad>'] = len(word_dict)

for k in list(word_dict)[:5]:
    print("{}:{}".format(k.decode('ASCII'), word_dict[k]))

print("...")

for k in list(word_dict)[-5:]:
    print("{}:{}".format(k if isinstance(k, str) else k.decode('ASCII'), word_dict[k]))

print("totally {} words".format(len(word_dict)))

vocab_size = len(word_dict) + 1
emb_size = 256
seq_len = 200
batch_size = 32
epochs = 2
pad_id = word_dict['<pad>']

classes = ['negative', 'positive']

def ids_to_str(ids):
    #print(ids)
    words = []
    for k in ids:
        w = list(word_dict)[k]
        words.append(w if isinstance(w, str) else w.decode('ASCII'))
    return " ".join(words)

vocab_size = len(word_dict) + 1
emb_size = 256
seq_len = 200
batch_size = 32
epochs = 2
pad_id = word_dict['<pad>']

classes = ['negative', 'positive']

def ids_to_str(ids):
    #print(ids)
    words = []
    for k in ids:
        w = list(word_dict)[k]
        words.append(w if isinstance(w, str) else w.decode('ASCII'))
    return " ".join(words)

# 取出来第一条数据看看样子。
sent = train_dataset.docs[0]
label = train_dataset.labels[1]
print('sentence list id is:', sent)
print('sentence label id is:', label)
print('--------------------------')
print('sentence list is: ', ids_to_str(sent))
print('sentence label is: ', classes[label])

def create_padded_dataset(dataset):
    padded_sents = []
    labels = []
    for batch_id, data in enumerate(dataset):
        sent, label = data[0], data[1]
        padded_sent = np.concatenate([sent[:seq_len], [pad_id] * (seq_len - len(sent))]).astype('int32')
        padded_sents.append(padded_sent)
        labels.append(label)
    return np.array(padded_sents), np.array(labels, dtype=np.int64)

train_sents, train_labels = create_padded_dataset(train_dataset)
test_sents, test_labels = create_padded_dataset(test_dataset)

print(train_sents.shape)
print(train_labels.shape)
print(test_sents.shape)
print(test_labels.shape)

for sent in train_sents[:3]:
    print(ids_to_str(sent))

class IMDBDataset(paddle.io.Dataset):
    def __init__(self, sents, labels):
        self.sents = sents
        self.labels = labels

    def __getitem__(self, index):
        data = self.sents[index]
        label = self.labels[index]

        return data, label

    def __len__(self):
        return len(self.sents)


train_dataset = IMDBDataset(train_sents, train_labels)
test_dataset = IMDBDataset(test_sents, test_labels)

train_loader = paddle.io.DataLoader(train_dataset, return_list=True, shuffle=True,
                                    batch_size=batch_size, drop_last=True)
test_loader = paddle.io.DataLoader(test_dataset, return_list=True, shuffle=True,
                                   batch_size=batch_size, drop_last=True)

class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.emb = paddle.nn.Embedding(vocab_size, emb_size)
        self.fc = paddle.nn.Linear(in_features=emb_size, out_features=2)
        self.dropout = paddle.nn.Dropout(0.5)

    def forward(self, x):
        x = self.emb(x)
        x = paddle.mean(x, axis=1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 方式1：用高层API训练与验证 ##
model = paddle.Model(MyNet()) # 用 Model封装 MyNet

# 模型配置
model.prepare(optimizer=paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters()),
              loss=paddle.nn.CrossEntropyLoss())

# 模型训练
model.fit(train_loader,
          test_loader,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1)


# # 方式2：用底层API训练与验证 ##
# def train(model):
#     model.train()
#     opt = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
#
#     for epoch in range(epochs):
#         for batch_id, data in enumerate(train_loader):
#
#             sent = data[0]
#             label = data[1]
#
#             logits = model(sent)
#             loss = paddle.nn.functional.cross_entropy(logits, label)
#
#             if batch_id % 500 == 0:
#                 print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, loss.numpy()))
#
#             loss.backward()
#             opt.step()
#             opt.clear_grad()
#
#         # evaluate model after one epoch
#         model.eval()
#         accuracies = []
#         losses = []
#
#         for batch_id, data in enumerate(test_loader):
#             sent = data[0]
#             label = data[1]
#
#             logits = model(sent)
#             loss = paddle.nn.functional.cross_entropy(logits, label)
#             acc = paddle.metric.accuracy(logits, label)
#
#             accuracies.append(acc.numpy())
#             losses.append(loss.numpy())
#
#         avg_acc, avg_loss = np.mean(accuracies), np.mean(losses)
#         print("[validation] accuracy/loss: {}/{}".format(avg_acc, avg_loss))
#
#         model.train()
#
#
# model = MyNet()
# train(model)
