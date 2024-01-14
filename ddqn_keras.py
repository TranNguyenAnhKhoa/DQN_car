# import random , numpy,math
# import tensorflow as tf
#
# class Brain:
#     def __int__(self,NbrStates,NbrActions):
#         self.NbrStates = NbrStates
#         self.NbrActions = NbrActions
#         self.model = self.createModel()
#
#     def createModel(self):
#         model = tf.keras.Sequential()
#         model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
#         model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
#         model.add(tf.keras.layers.Dense(self.NbrActions, activation=tf.nn.softmax))
#         model.compile(loss="mse",optimizer="adam")
#         return model
#
#     def train(self,x,y,epoch,verbose = 0):
#         self.model.fit(x,y,batch_size = 64,epochs = epoch,verbose= verbose)
#
#     def predict(self,s):
#         return self.model.predict(s)
#
#     def predictOne(self,s):
#         return self.model.predict(tf.reshape(s,[1,self.NbrStates])).flatten()
#
# class ExpReplay:
#     samples =[]
#     def __int__(self,capacity):
#         self.capacity = capacity
#     def add(self,sample):
#         self.samples.append(sample)
#         if len(self.samples) > self.capacity:
#             self.samples.pop(0)
#
#     def sample(self,n):
#         n = min(n,len(self.samples))
#         return random.sample(self.samples,n)
#
# #-----------------------------------------------------
# ExpReplay_CAPACITY = 2000000
# OBSERVERPERIOD = 750
# BATCH_SIZE = 1024
# GAMMA = 0.95
# MAX_EPSILON = 1
# MIN_EPSILON = 0.05
# LAMDA = 0.0005
# #------------------------------------------------------
# class Agent:
#     def __init__(self,NbrStates,NbrActions):
#         self.NbrStates = NbrStates
#         self.NbrActions = NbrActions
#         self.brain = Brain(NbrStates,NbrActions)
#         self.ExReplay = ExpReplay(ExpReplay_CAPACITY)
#         self.steps = 0
#         self.epsilon = MAX_EPSILON
#
#     def Act(self,s):
#         if(random.random() < self.epsilon or self.steps < OBSERVERPERIOD):
#             return random.randint(0,self.NbrActions-1)
#         else:
#             return numpy.argmax(self.brain.predictOne(s))
#
#     def CaptureSample(self,sample):
#         self.ExReplay.add(sample)
#         self.steps +=1
#         if(self.steps > OBSERVERPERIOD):
#             self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMDA * (self.steps - OBSERVERPERIOD))
#
#     def Process(self):
#         batch = self.ExReplay.sample(BATCH_SIZE)
#         batchlen = len(batch)
#         no_state = numpy.zeros(self.NbrStates)
#         states = numpy.array([batchitem[0]] for batchitem in batch)
#         states_ = numpy.array([(no_state if batchitem[3] is None else batchitem[3])for batchitem in batch])
#         predictedQ  = self.brain.predict(states)
#         predictedNextQ = self.brainn.predict(states_)
#
#         x = numpy.zeros(batchlen,self.NbrActions)
#         y = numpy.zeros(batchlen,self.NbrStates)
#         for i in range(batchlen):
#             batchitem = batch[i]
#             state = batchitem[0]
#             a = batchitem[1]
#             reward = batchitem[2]
#             nextstate = batchitem[3]
#             targetQ = predictedQ[i]
#             if nextstate is None:
#                 targetQ[a] = reward
#             else:
#                 targetQ[a] = reward + GAMMA *numpy.amax(predictedNextQ[i])
#
#             x[i] = state
#             y[i] = state
#
#             self.brain.train((x,y))



from keras.models import  load_model

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


class DDQNAgent(object):
    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.999995, epsilon_end=0.10,
                 mem_size=25000, fname='ddqn_model.h5', replace_target=25):
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

        self.brain_eval = Brain(input_dims, n_actions, batch_size)
        self.brain_target = Brain(input_dims, n_actions, batch_size)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):

        state = np.array(state)
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

            q_target[batch_index, action_indices] = reward + self.gamma * q_next[batch_index, max_actions.astype(int)] * done

            self.brain_eval.train(state, q_target)

            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_min else self.epsilon_min

    def update_network_parameters(self):
        self.brain_target.copy_weights(self.brain_eval)

    def save_model(self):
        self.brain_eval.model.save(self.model_file)

    def load_model(self):
        self.brain_eval.model = load_model(self.model_file)
        self.brain_target.model = load_model(self.model_file)

        if self.epsilon == 0.0:
            self.update_network_parameters()


class Brain:
    def __init__(self, NbrStates, NbrActions, batch_size=512):
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.model = self.createModel()

    def createModel(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))  # prev 256
        model.add(tf.keras.layers.Dense(self.NbrActions, activation=tf.nn.softmax))
        model.compile(loss="mse", optimizer="adam")

        return model

    def train(self, x, y, epoch=1, verbose=0):
        self.model.fit(x, y, batch_size=self.batch_size, verbose=verbose)

    def predict(self, s):
        return self.model.predict(s)

    def predictOne(self, s):
        return self.model.predict(tf.reshape(s, [1, self.NbrStates])).flatten()

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())
