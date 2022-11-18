import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset

# 训练轮数
EPOCH = 3

df = pd.read_csv('./data/ml-latest-small/ratings.csv')
user_ids = df["userId"].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = df["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
df["user"] = df["userId"].map(user2user_encoded)
df["movie"] = df["movieId"].map(movie2movie_encoded)

num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
df["rating"] = df["rating"].values.astype(np.float32)
# 最小和最大额定值将在以后用于标准化额定值
min_rating = min(df["rating"])
max_rating = max(df["rating"])

print(
    "Number of users: {}, Number of Movies: {}, Min rating: {}, Max rating: {}".format(
        num_users, num_movies, min_rating, max_rating
    )
)

df = df.sample(frac=1, random_state=42)
x = df[["user", "movie"]].values
# 规范化0和1之间的目标。使训练更容易。
y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
# 假设对90%的数据进行训练，对10%的数据进行验证。
train_indices = int(0.9 * df.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)
y_train = y_train[:, np.newaxis]
y_val = y_val[:, np.newaxis]
y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)


# 自定义数据集
# 映射式(map-style)数据集需要继承paddle.io.Dataset
class SelfDefinedDataset(Dataset):
    def __init__(self, data_x, data_y, mode='train'):
        super(SelfDefinedDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

    def __getitem__(self, idx):
        if self.mode == 'predict':
            return self.data_x[idx]
        else:
            return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)


traindataset = SelfDefinedDataset(x_train, y_train)
for data, label in traindataset:
    print(data.shape, label.shape)
    print(data, label)
    break
train_loader = paddle.io.DataLoader(traindataset, batch_size=128, shuffle=True)
for batch_id, data in enumerate(train_loader()):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

testdataset = SelfDefinedDataset(x_val, y_val)
test_loader = paddle.io.DataLoader(testdataset, batch_size=128, shuffle=True)
for batch_id, data in enumerate(test_loader()):
    x_data = data[0]
    y_data = data[1]

    print(x_data.shape)
    print(y_data.shape)
    break

EMBEDDING_SIZE = 50


class RecommenderNet(nn.Layer):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        weight_attr_user = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            weight_attr=weight_attr_user
        )
        self.user_bias = nn.Embedding(num_users, 1)
        weight_attr_movie = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_size,
            weight_attr=weight_attr_movie
        )
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = paddle.dot(user_vector, movie_vector)
        x = dot_user_movie + user_bias + movie_bias
        x = nn.functional.sigmoid(x)

        return x


class RecommenderNet(nn.Layer):
    def __init__(self, num_users, num_movies, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        weight_attr_user = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            weight_attr=weight_attr_user
        )
        self.user_bias = nn.Embedding(num_users, 1)
        weight_attr_movie = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.movie_embedding = nn.Embedding(
            num_movies,
            embedding_size,
            weight_attr=weight_attr_movie
        )
        self.movie_bias = nn.Embedding(num_movies, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = paddle.dot(user_vector, movie_vector)
        x = dot_user_movie + user_bias + movie_bias
        x = nn.functional.sigmoid(x)

        return x


model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model = paddle.Model(model)

optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.0003)
loss = nn.BCELoss()
metric = paddle.metric.Accuracy()

# 设置visualdl路径
log_dir = './visualdl'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)

model.prepare(optimizer, loss, metric)
model.fit(train_loader, epochs=EPOCH, save_dir='./checkpoints', verbose=1, callbacks=callback)

model.evaluate(test_loader, batch_size=64, verbose=1)

movie_df = pd.read_csv('./data/ml-latest-small/movies.csv')

# 获取一个用户，查看他的推荐电影
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[
    ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
]["movieId"]
movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
)
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
)
testdataset = SelfDefinedDataset(user_movie_array, user_movie_array, mode='predict')
test_loader = paddle.io.DataLoader(testdataset, batch_size=9703, shuffle=False, return_list=True, )

ratings = model.predict(test_loader)
ratings = np.array(ratings)
ratings = np.squeeze(ratings, 0)
ratings = np.squeeze(ratings, 2)
ratings = np.squeeze(ratings, 0)
top_ratings_indices = ratings.argsort()[::-1][0:10]

print(top_ratings_indices)
recommended_movie_ids = [
    movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
]

print("用户的ID为: {}".format(user_id))
print("====" * 8)
print("用户评分较高的电影：")
print("----" * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)

print("----" * 8)
print("为用户推荐的10部电影：")
print("----" * 8)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)
