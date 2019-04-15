# coding:utf-8

import numpy as np
import random


class linearRegression:
    def __init__(self, lr=0.001):
        self.w = 0
        self.b = 0
        self.dw = 0
        self.db = 0
        self.lr = lr

    def predict(self, x_list):
        return np.array(self.w) * x_list + np.array(self.b)

    def loss(self, x_list, t_list,i):
        y_list = self.predict(x_list)
        diff = y_list - t_list
        loss = np.average(diff ** 2) / 2
        if i % 1000 == 0:
            print('i={0}:'.format(i))
            print('w={0} b={1} loss={2}'.format(self.w, self.b, loss))
        self.dw = np.average(x_list * diff)
        self.db = np.average(diff)
        self.w -= self.lr * self.dw
        self.b -= self.lr * self.db
        return loss

    def train(self, x_list, t_list, max_iter):
        for i in range(max_iter):
            batch_idxs = np.random.choice(len(x_list), size=50)
            x_list_batch = np.array(x_list)[batch_idxs]
            t_list_batch = np.array(t_list)[batch_idxs]
            loss = self.loss(x_list_batch, t_list_batch,i)


def getTrainData(w, b,size=100):
    x_list = np.random.randint(100, size=size) * random.random()
    y_list = w * x_list + b + np.random.randint(-1, 2, size=size) * np.random.random_sample(size)
    return x_list, y_list


if __name__ == '__main__':
    w = random.randint(0, 10) + random.random()
    b = random.randint(0, 5) + random.random()
    x_list, t_list = getTrainData(w, b)
    print('tw={0} tb={1}'.format(w, b))
    LR = linearRegression(lr=0.001)
    LR.train(x_list, t_list, 100000)
