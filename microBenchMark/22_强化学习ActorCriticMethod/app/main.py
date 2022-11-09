import gym, os
from itertools import count
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
import paddle.nn.functional as F
from paddle.distribution import Categorical

print(paddle.__version__)

device = paddle.get_device()
env = gym.make("CartPole-v0")  ### 或者 env = gym.make("CartPole-v0").unwrapped 开启无锁定环境训练

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
lr = 0.001

class Actor(nn.Layer):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        distribution = Categorical(F.softmax(output, axis=-1))
        return distribution


class Critic(nn.Layer):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear1 = nn.Linear(self.state_size, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        value = self.linear3(output)
        return value

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns


def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(lr, parameters=actor.parameters())
    optimizerC = optim.Adam(lr, parameters=critic.parameters())
    for iter in range(n_iters):
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        masks = []
        entropy = 0
        env.reset()

        for i in count():
            # env.render()
            state = paddle.to_tensor(state,dtype="float32",place=device)
            dist, value = actor(state), critic(state)

            action = dist.sample([1])
            next_state, reward, done, _ = env.step(action.cpu().squeeze(0).numpy())

            log_prob = dist.log_prob(action);
            # entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(paddle.to_tensor([reward], dtype="float32", place=device))
            masks.append(paddle.to_tensor([1-done], dtype="float32", place=device))

            state = next_state

            if done:
                if iter % 10 == 0:
                    print('Iteration: {}, Score: {}'.format(iter, i))
                break


        next_state = paddle.to_tensor(next_state, dtype="float32", place=device)
        next_value = critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = paddle.concat(log_probs)
        returns = paddle.concat(returns).detach()
        values = paddle.concat(values)

        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        optimizerA.clear_grad()
        optimizerC.clear_grad()
        actor_loss.backward()
        critic_loss.backward()
        optimizerA.step()
        optimizerC.step()
    paddle.save(actor.state_dict(), 'model/actor.pdparams')
    paddle.save(critic.state_dict(), 'model/critic.pdparams')
    env.close()


if __name__ == '__main__':
    if os.path.exists('model/actor.pdparams'):
        actor = Actor(state_size, action_size)
        model_state_dict  = paddle.load('model/actor.pdparams')
        actor.set_state_dict(model_state_dict )
        print('Actor Model loaded')
    else:
        actor = Actor(state_size, action_size)
    if os.path.exists('model/critic.pdparams'):
        critic = Critic(state_size, action_size)
        model_state_dict  = paddle.load('model/critic.pdparams')
        critic.set_state_dict(model_state_dict )
        print('Critic Model loaded')
    else:
        critic = Critic(state_size, action_size)
    trainIters(actor, critic, n_iters=201)
