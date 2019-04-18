# coding:utf-8

import numpy as np
from common.activateFunc import sigmoid


class LogisticRegression:
    def __init__(self, lr=0.001):
        self.w = 0
        self.b = 0
        self.dw = 0
        self.db = 0
        self.lr = lr

    def predict(self, x_list):
        return sigmoid(np.array(self.w) * x_list + np.array(self.b))

    def loss(self, x_list, t_list, i):
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

    def loss2(self, x_list, t_list, i):
        y_list = self.predict(x_list)
        diff = y_list - t_list
        J = -t_list * np.log(y_list) - (1 - t_list) * np.log(1 - y_list)
        loss = np.average(J)
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
            # self.loss(x_list_batch, t_list_batch,i)
            self.loss2(x_list_batch, t_list_batch, i)
        return self.w, self.b


def getTrainData(w, b, size=110):
    x_list = np.random.normal(50,40,size)
    t_list = np.array((w * x_list + b) > 0, dtype=int)
    train_idx= np.random.choice(size, size=(size-10),replace=False)
    test_idx=np.setdiff1d(np.arange(size),train_idx)
    x_train = np.array(x_list)[train_idx]
    t_train = np.array(t_list)[train_idx]
    x_test = np.array(x_list)[test_idx]
    t_test = np.array(t_list)[test_idx]
    return (x_train, t_train), (x_test, t_test)


if __name__ == '__main__':
    w = 2
    b = -100
    (x_train, t_train), (x_test, t_test) = getTrainData(w, b,size=110)
    LR = LogisticRegression(lr=0.001)
    predict_w,predict_b=LR.train(x_train, t_train, 10000)
    print('tw={0} tb={1}'.format(w, b))
    predict_y=LR.predict(x_test)
    rlt_pair=np.array((t_test,predict_y)).T
    print('rlt_pair:')
    print(rlt_pair)


