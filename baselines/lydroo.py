"""LyDROO memory DNN and OPN candidates adapted under LICENSE-LyDROO."""

# Copyright (c) 2021 Liang HUANG. Original memory DNN by Weijian Pan and Liang Huang.
# Source: https://github.com/revenol/LyDROO/blob/main/memory.py


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.01,
        training_interval=10,
        batch_size=100,
        memory_size=1000
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size


        self.memory_counter = 1


        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.Sigmoid()
        )

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))

        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)

        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])


        optimizer = optim.Adam(self.model.parameters(), lr=self.lr,betas = (0.09,0.999),weight_decay=0.0001)
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)
        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        assert(self.cost > 0)

    def decode(self, h, k=1):
        h = torch.Tensor(h[np.newaxis, :])

        self.model.eval()
        m_pred = self.model(h)
        m_pred = m_pred.detach().numpy()


        return self.opn(m_pred[0], k)

    def knm(self, m, k = 1):
        m_list = []
        m_list.append(1*(m>0.5))

        if k > 1:
            m_abs = abs(m-0.5)
            idx_list = np.argsort(m_abs)[:k-1]
            for i in range(k-1):
                if m[idx_list[i]] >0.5:
                    m_list.append(1*(m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1*(m - m[idx_list[i]] >= 0))

        return m_list

    def opn(self, m, k= 1):
        m_noisy = 1 / (1 + np.exp(-(m + np.random.normal(0, 1, len(m))))) # Sigmoid压缩
        return self.knm(m, k) + self.knm(m_noisy, k)


