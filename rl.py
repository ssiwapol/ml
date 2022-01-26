import sys

import numpy as np

from env import MountainCar


class LinearModel:
    def __init__(self, state_size, action_size, lr, indices):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.indices = indices
        # initialize weight and intercept to zero
        self.w = np.zeros((state_size, action_size))
        self.b = 0
    
    # linear prediction
    def predict(self, state):
        state = np.array([state.get(i, 0) for i in range(self.state_size)]).reshape(-1, 1)
        return np.dot(self.w.T, state) + self.b

    # update weight and intercept
    def update(self, state, action, target):
        q = self.predict(state)
        q = q[action][0]
        g = np.zeros((self.state_size, self.action_size))
        g[:, action] = np.array([state.get(i, 0) for i in range(self.state_size)])
        self.w = self.w - (self.lr * (q-target) * g)
        self.b = self.b - (self.lr * (q-target) * 1)


class QlearningAgent:
    def __init__(self, env, mode, actions, gamma=0.9, lr=0.01, epsilon=0.05):
        self.env = env
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.actions = actions
        indices = True if mode == 'tile' else False
        # initilize linear model by state space and actions space
        self.lm = LinearModel(env.state_space, len(actions), lr, indices)

    # epsilon-greedy strategy
    def get_action(self, state):
        if np.random.uniform(0.0, 1.0) > self.epsilon:
            action = np.argmax(self.lm.predict(state), axis=0)[0]
            action = self.actions[action]
        else:
            action = np.random.choice(self.actions)
        return action

    # train for episodes iterations with at most max iterations
    def train(self, episodes, max_iterations):
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = self.env.reset()
            for j in range(max_iterations):
                action = self.get_action(state)
                state_, reward, done = self.env.step(action)
                target = reward + (self.gamma * np.max(self.lm.predict(state_), axis=0)[0])
                self.lm.update(state, action, target)
                state = state_
                rewards[i] += reward
                if done:
                    break
        return rewards, self.lm.b, self.lm.w

# write output
def write_output(rewards, intercept, weight, weight_out, returns_out):
    with open(weight_out, 'w') as f:
        f.write(str(intercept) + '\n')
        for i in range(weight.shape[0]):
            for j in range(weight.shape[1]):
                f.write(str(weight[i, j]) + '\n')
    
    with open(returns_out, 'w') as f:
        for i in range(rewards.shape[0]):
            f.write(str(rewards[i]) + '\n')

def main(args):
    mode = args[1]
    weight_out = args[2]
    returns_out = args[3]
    episodes = int(args[4])
    max_iterations = int(args[5])
    epsilon = float(args[6])
    gamma = float(args[7])
    learning_rate = float(args[8])
    actions = [0, 1, 2]
    env = MountainCar(mode)
    agent = QlearningAgent(env, mode, actions, gamma, learning_rate, epsilon)
    rewards, intercept, weight = agent.train(episodes, max_iterations)
    write_output(rewards, intercept, weight, weight_out, returns_out)


if __name__ == "__main__":
    main(sys.argv)
