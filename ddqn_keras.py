from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_action):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # prev 256
    model.add(tf.keras.layers.Dense(n_action, activation=tf.nn.softmax))
    model.compile(loss="mse", optimizer=Adam(lr=lr))

    return model


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.9996, epsilon_end=0.01,
                 mem_size=25000000, fname='ddqn_model.h5', replace_target=100):
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)

        self.brain_eval = build_dqn(alpha, n_actions)
        self.brain_target = build_dqn(alpha, n_actions)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        state = state[np.newaxis, :]

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            actions = self.brain_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_next = self.brain_target.predict(new_state)
            q_eval = self.brain_eval.predict(new_state)
            q_pred = self.brain_eval.predict(state)

            max_actions = np.argmax(q_eval, axis=1)

            q_target = q_pred

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + self.gamma * q_next[
                batch_index, max_actions.astype(int)] * done

            _ = self.brain_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min
            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()

    def update_network_parameters(self):
        self.brain_target.set_weights(self.brain_eval.get_weights())

    def save_model(self):
        self.brain_eval.save(self.model_file)

    def load_model(self):
        self.brain_eval.model = load_model(self.model_file)
        # self.brain_target.model = load_model(self.model_file)

        if self.epsilon <= self.epsilon_min:
            self.update_network_parameters()

# class Brain:
#     def __init__(self, NbrStates, NbrActions, batch_size = 256):
#         self.NbrStates = NbrStates
#         self.NbrActions = NbrActions
#         self.batch_size = batch_size
#         self.model = self.createModel()
#
#
#     def createModel(self):
#         model = tf.keras.Sequential()
#         model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu)) #prev 256
#         model.add(tf.keras.layers.Dense(self.NbrActions, activation=tf.nn.softmax))
#         model.compile(loss = "mse", optimizer="adam")
#
#         return model
#
#     def train(self, x, y, epoch = 1, verbose = 0):
#         self.model.fit(x, y, batch_size = self.batch_size , verbose = verbose)
#
#     def predict(self, s):
#         return self.model.predict(s)
#
#     def predictOne(self, s):
#         return self.model.predict(tf.reshape(s, [1, self.NbrStates])).flatten()
#
#     def copy_weights(self, TrainNet):
#         variables1 = self.model.trainable_variables
#         variables2 = TrainNet.model.trainable_variables
#         for v1, v2 in zip(variables1, variables2):
#             v1.assign(v2.numpy())
