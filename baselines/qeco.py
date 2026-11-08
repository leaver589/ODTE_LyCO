"""QECO dueling double-DQN/LSTM network adapted under LICENSE-QECO."""

# Copyright (c) 2023 Iman Rahmati.
# Source: https://github.com/ImanRHT/QECO/blob/main/D3QN.py

import numpy as np
from collections import deque
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class DuelingDoubleDeepQNetwork:

    def __init__(self,
                 n_actions,
                 n_features,
                 n_lstm_features,
                 learning_rate = 0.01,
                 reward_decay = 0.9,
                 e_greedy = 0.99,
                 replace_target_iter = 200,
                 memory_size = 500,
                 batch_size=32,
                 e_greedy_increment= 0.00025,
                 n_lstm_step = 10,
                 N_L1 = 20,
                 N_lstm = 20,
                 seed = None,
):

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size    # select self.batch_size number of time sequence for learning
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.learn_step_counter = 0
        self.N_L1 = N_L1
        self.seed = seed
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step       # step_size in lstm
        self.n_lstm_state = n_lstm_features  # [fog1, fog2, ...., fogn, M_n(t)]

        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                    + self.n_features + self.n_lstm_state + self.n_lstm_state))
        self.history_memory = np.zeros(
            (self.memory_size, self.n_lstm_step, 2 * self.n_lstm_state),
            dtype=np.float32,
        )
        self.memory_counter = 0

        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='eval_net')
        self.replace_target_op = [tf.assign(t, e) for t, e in
                                      zip(t_params, e_params)]  # update the parameters in target_net

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for ii in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))


    def _build_net(self):

        tf.reset_default_graph()
        if self.seed is not None:
            tf.set_random_seed(self.seed)

        def build_layers(s,lstm_s,c_names, n_l1, n_lstm, w_initializer, b_initializer):

            with tf.variable_scope('l0'):
                lstm_dnn = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(n_lstm)
                # Preserve upstream graph operation order for the seeded neural initialization.
                lstm_dnn.zero_state(self.batch_size, tf.float32)
                lstm_output,lstm_state = tf.nn.dynamic_rnn(lstm_dnn, lstm_s, dtype=tf.float32)
                lstm_output_reduced = tf.reshape(lstm_output[:, -1, :], shape=[-1, n_lstm])

            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[n_lstm + self.n_features, n_l1], initializer=w_initializer,
                                     collections=c_names)
                b1 = tf.get_variable('b1',[1,n_l1],initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(tf.concat([lstm_output_reduced, s],1), w1) + b1)

            with tf.variable_scope('l12'):
                w12 = tf.get_variable('w12', [n_l1, n_l1], initializer=w_initializer,
                                         collections=c_names)
                b12 = tf.get_variable('b12', [1, n_l1], initializer=b_initializer, collections=c_names)
                l12 = tf.nn.relu(tf.matmul(l1, w12) + b12)

            with tf.variable_scope('Value'):
                w2 = tf.get_variable('w2',[n_l1,1],initializer=w_initializer,collections=c_names)
                b2 = tf.get_variable('b2',[1,1],initializer=b_initializer,collections=c_names)
                self.V = tf.matmul(l12,w2) + b2
            with tf.variable_scope('Advantage'):
                w2 = tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2 = tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.A = tf.matmul(l12,w2) + b2

            with tf.variable_scope('Q'):
                out = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))  # Q = V(s) +A(s,a)

            return out

        self.s = tf.placeholder(tf.float32,[None,self.n_features], name = 's')  # state (observation)
        self.lstm_s = tf.placeholder(tf.float32,[None,self.n_lstm_step,self.n_lstm_state], name='lstm1_s')

        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions], name = 'Q_target') # q_target

        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        self.lstm_s_ = tf.placeholder(tf.float32,[None,self.n_lstm_step,self.n_lstm_state], name='lstm1_s_')

        with tf.variable_scope('eval_net'):

            c_names, n_l1, n_lstm, w_initializer, b_initializer =  \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], self.N_L1, self.N_lstm,\
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, self.lstm_s, c_names, n_l1, n_lstm, w_initializer, b_initializer)

        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, self.lstm_s_, c_names, n_l1, n_lstm, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, lstm_s,  a, r, s_, lstm_s_,
                         *, history=None, next_history=None):

        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))  # stack in horizontal direction

        index = self.memory_counter % self.memory_size
        if history is None or next_history is None:
            raise ValueError('Explicit replay requires both LSTM histories')
        self.history_memory[index] = np.concatenate(
            (np.asarray(history), np.asarray(next_history)), axis=1
        )
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):

        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if self.rng.uniform() < self.epsilon:

            lstm_observation = np.array(self.lstm_history)

            actions_value = self.sess.run(self.q_eval,
                                          feed_dict={self.s: observation,
                                                     self.lstm_s: lstm_observation.reshape(1, self.n_lstm_step,
                                                                                           self.n_lstm_state),
                                                     })


            action = np.argmax(actions_value)

        else:
            action = self.rng.randint(1, self.n_actions)
        return action

    def learn(self):

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)

        sample_index = self.rng.choice(
            min(self.memory_counter, self.memory_size), size=self.batch_size
        )

        batch_memory = self.memory[sample_index, :self.n_features+1+1+self.n_features]
        lstm_batch_memory = self.history_memory[sample_index]

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],  # output
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:], self.lstm_s_: lstm_batch_memory[:,:,self.n_lstm_state:],
                self.s: batch_memory[:, -self.n_features:], self.lstm_s: lstm_batch_memory[:,:,self.n_lstm_state:],
            }
        )
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features],
                                                 self.lstm_s: lstm_batch_memory[:,:,:self.n_lstm_state]})
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)  # action with a single value (int action)
        reward = batch_memory[:, self.n_features + 1]  # reward with a single value

        max_act4next = np.argmax(q_eval4next, axis=1)
        selected_q_next = q_next[batch_index, max_act4next]

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.lstm_s: lstm_batch_memory[:, :, :self.n_lstm_state],
                                                self.q_target: q_target})

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return self.cost


