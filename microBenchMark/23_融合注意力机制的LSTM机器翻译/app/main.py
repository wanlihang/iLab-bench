import paddle
import paddle.nn.functional as F
import re
import numpy as np

# 训练轮数
EPOCH = 1

MAX_LEN = 10

lines = open('./data/cmn-eng/cmn.txt', encoding='utf-8').read().strip().split('\n')
words_re = re.compile(r'\w+')

pairs = []
for l in lines:
    en_sent, cn_sent, _ = l.split('\t')
    pairs.append((words_re.findall(en_sent.lower()), list(cn_sent)))

# create a smaller dataset to make the demo process faster
filtered_pairs = []

for x in pairs:
    if len(x[0]) < MAX_LEN and len(x[1]) < MAX_LEN and \
            x[0][0] in ('i', 'you', 'he', 'she', 'we', 'they'):
        filtered_pairs.append(x)

print(len(filtered_pairs))
for x in filtered_pairs[:10]: print(x)

en_vocab = {}
cn_vocab = {}

# create special token for pad, begin of sentence, end of sentence
en_vocab['<pad>'], en_vocab['<bos>'], en_vocab['<eos>'] = 0, 1, 2
cn_vocab['<pad>'], cn_vocab['<bos>'], cn_vocab['<eos>'] = 0, 1, 2

en_idx, cn_idx = 3, 3
for en, cn in filtered_pairs:
    for w in en:
        if w not in en_vocab:
            en_vocab[w] = en_idx
            en_idx += 1
    for w in cn:
        if w not in cn_vocab:
            cn_vocab[w] = cn_idx
            cn_idx += 1

print(len(list(en_vocab)))
print(len(list(cn_vocab)))

padded_en_sents = []
padded_cn_sents = []
padded_cn_label_sents = []
for en, cn in filtered_pairs:
    # reverse source sentence
    padded_en_sent = en + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(en))
    padded_en_sent.reverse()
    padded_cn_sent = ['<bos>'] + cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn))
    padded_cn_label_sent = cn + ['<eos>'] + ['<pad>'] * (MAX_LEN - len(cn) + 1)

    padded_en_sents.append([en_vocab[w] for w in padded_en_sent])
    padded_cn_sents.append([cn_vocab[w] for w in padded_cn_sent])
    padded_cn_label_sents.append([cn_vocab[w] for w in padded_cn_label_sent])

train_en_sents = np.array(padded_en_sents)
train_cn_sents = np.array(padded_cn_sents)
train_cn_label_sents = np.array(padded_cn_label_sents)

print(train_en_sents.shape)
print(train_cn_sents.shape)
print(train_cn_label_sents.shape)

embedding_size = 128
hidden_size = 256
num_encoder_lstm_layers = 1
en_vocab_size = len(list(en_vocab))
cn_vocab_size = len(list(cn_vocab))
batch_size = 16


# encoder: simply learn representation of source sentence
class Encoder(paddle.nn.Layer):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = paddle.nn.Embedding(en_vocab_size, embedding_size, )
        self.lstm = paddle.nn.LSTM(input_size=embedding_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_encoder_lstm_layers)

    def forward(self, x):
        x = self.emb(x)
        x, (_, _) = self.lstm(x)
        return x


# only move one step of LSTM,
# the recurrent loop is implemented inside training loop
class AttentionDecoder(paddle.nn.Layer):
    def __init__(self):
        super(AttentionDecoder, self).__init__()
        self.emb = paddle.nn.Embedding(cn_vocab_size, embedding_size)
        self.lstm = paddle.nn.LSTM(input_size=embedding_size + hidden_size,
                                   hidden_size=hidden_size)

        # for computing attention weights
        self.attention_linear1 = paddle.nn.Linear(hidden_size * 2, hidden_size)
        self.attention_linear2 = paddle.nn.Linear(hidden_size, 1)

        # for computing output logits
        self.outlinear = paddle.nn.Linear(hidden_size, cn_vocab_size)

    def forward(self, x, previous_hidden, previous_cell, encoder_outputs):
        x = self.emb(x)

        attention_inputs = paddle.concat((encoder_outputs,
                                          paddle.tile(previous_hidden, repeat_times=[1, MAX_LEN + 1, 1])),
                                         axis=-1
                                         )

        attention_hidden = self.attention_linear1(attention_inputs)
        attention_hidden = F.tanh(attention_hidden)
        attention_logits = self.attention_linear2(attention_hidden)
        attention_logits = paddle.squeeze(attention_logits)

        attention_weights = F.softmax(attention_logits)
        attention_weights = paddle.expand_as(paddle.unsqueeze(attention_weights, -1),
                                             encoder_outputs)

        context_vector = paddle.multiply(encoder_outputs, attention_weights)
        context_vector = paddle.sum(context_vector, 1)
        context_vector = paddle.unsqueeze(context_vector, 1)

        lstm_input = paddle.concat((x, context_vector), axis=-1)

        # LSTM requirement to previous hidden/state:
        # (number_of_layers * direction, batch, hidden)
        previous_hidden = paddle.transpose(previous_hidden, [1, 0, 2])
        previous_cell = paddle.transpose(previous_cell, [1, 0, 2])

        x, (hidden, cell) = self.lstm(lstm_input, (previous_hidden, previous_cell))

        # change the return to (batch, number_of_layers * direction, hidden)
        hidden = paddle.transpose(hidden, [1, 0, 2])
        cell = paddle.transpose(cell, [1, 0, 2])

        output = self.outlinear(hidden)
        output = paddle.squeeze(output)
        return output, (hidden, cell)


encoder = Encoder()
atten_decoder = AttentionDecoder()

opt = paddle.optimizer.Adam(learning_rate=0.001,
                            parameters=encoder.parameters() + atten_decoder.parameters())

for epoch in range(EPOCH):
    print("epoch:{}".format(epoch))

    # shuffle training data
    perm = np.random.permutation(len(train_en_sents))
    train_en_sents_shuffled = train_en_sents[perm]
    train_cn_sents_shuffled = train_cn_sents[perm]
    train_cn_label_sents_shuffled = train_cn_label_sents[perm]

    for iteration in range(train_en_sents_shuffled.shape[0] // batch_size):
        x_data = train_en_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))]
        sent = paddle.to_tensor(x_data)
        en_repr = encoder(sent)

        x_cn_data = train_cn_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))]
        x_cn_label_data = train_cn_label_sents_shuffled[(batch_size * iteration):(batch_size * (iteration + 1))].astype(
            'int64')

        # shape: (batch,  num_layer(=1 here) * num_of_direction(=1 here), hidden_size)
        hidden = paddle.zeros([batch_size, 1, hidden_size])
        cell = paddle.zeros([batch_size, 1, hidden_size])

        loss = paddle.zeros([1])
        # the decoder recurrent loop mentioned above
        for i in range(MAX_LEN + 2):
            cn_word = paddle.to_tensor(x_cn_data[:, i:i + 1])
            cn_word_label = paddle.to_tensor(x_cn_label_data[:, i])

            logits, (hidden, cell) = atten_decoder(cn_word, hidden, cell, en_repr)
            step_loss = F.cross_entropy(logits, cn_word_label)
            loss += step_loss

        loss = loss / (MAX_LEN + 2)
        if (iteration % 200 == 0):
            print("iter {}, loss:{}".format(iteration, loss.numpy()))

        loss.backward()
        opt.step()
        opt.clear_grad()

encoder.eval()
atten_decoder.eval()

num_of_exampels_to_evaluate = 10

indices = np.random.choice(len(train_en_sents), num_of_exampels_to_evaluate, replace=False)
x_data = train_en_sents[indices]
sent = paddle.to_tensor(x_data)
en_repr = encoder(sent)

word = np.array(
    [[cn_vocab['<bos>']]] * num_of_exampels_to_evaluate
)
word = paddle.to_tensor(word)

hidden = paddle.zeros([num_of_exampels_to_evaluate, 1, hidden_size])
cell = paddle.zeros([num_of_exampels_to_evaluate, 1, hidden_size])

decoded_sent = []
for i in range(MAX_LEN + 2):
    logits, (hidden, cell) = atten_decoder(word, hidden, cell, en_repr)
    word = paddle.argmax(logits, axis=1)
    decoded_sent.append(word.numpy())
    word = paddle.unsqueeze(word, axis=-1)

results = np.stack(decoded_sent, axis=1)
for i in range(num_of_exampels_to_evaluate):
    en_input = " ".join(filtered_pairs[indices[i]][0])
    ground_truth_translate = "".join(filtered_pairs[indices[i]][1])
    model_translate = ""
    for k in results[i]:
        w = list(cn_vocab)[k]
        if w != '<pad>' and w != '<eos>':
            model_translate += w
    print(en_input)
    print("true: {}".format(ground_truth_translate))
    print("pred: {}".format(model_translate))
