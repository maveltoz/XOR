import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class XOR:
    def __init__(self):
        self.input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.hidden1 = np.random.rand(2, 2)
        self.hidden2 = np.random.rand(2)
        self.output = np.array([0, 1, 1, 0])
        self.lr = 1

    def Predict(self):
        self.hidden_out1 = sigmoid(np.dot(self.input, self.hidden1))
        self.hidden_out2 = sigmoid(np.dot(self.hidden_out1, self.hidden2))
        return self.hidden_out2

    def GetLoss(self, pred):
        return np.sum(0.5 * ((pred - self.output) * (pred - self.output)))

    def GetPred(self):
        self.predicted = np.zeros(4, )
        for i in range(0, 4):
            self.predicted[i] = 1 if self.hidden_out2[i] > 0.5 else 0
        return self.predicted

    def GetAccuracy(self):
        self.accuracy = 0
        for i in range(0, 4):
            if self.predicted[i] == self.output[i]:
                self.accuracy += 1
        return self.accuracy / 4

    def Update(self):
        self.grad2_1 = (self.hidden_out2 - self.output) * (self.hidden_out2 * (1 - self.hidden_out2)) * self.hidden_out1[:, 0]
        self.grad2_2 = (self.hidden_out2 - self.output) * (self.hidden_out2 * (1 - self.hidden_out2)) * self.hidden_out1[:, 1]

        self.hidden2[0] -= self.lr * np.sum(self.grad2_1) / 4
        self.hidden2[1] -= self.lr * np.sum(self.grad2_2) / 4

        self.grad1_1_1 = (self.hidden_out2 - self.output) * (self.hidden_out2 * (1 - self.hidden_out2)) * \
                         (self.hidden2[0]) * (self.hidden_out1[:, 0] * (1 - self.hidden_out1[:, 0])) * self.input[:, 0]
        self.grad1_1_2 = (self.hidden_out2 - self.output) * (self.hidden_out2 * (1 - self.hidden_out2)) * \
                         (self.hidden2[1]) * (self.hidden_out1[:, 1] * (1 - self.hidden_out1[:, 1])) * self.input[:, 0]
        self.grad1_2_1 = (self.hidden_out2 - self.output) * (self.hidden_out2 * (1 - self.hidden_out2)) * \
                         (self.hidden2[0]) * (self.hidden_out1[:, 0] * (1 - self.hidden_out1[:, 0])) * self.input[:, 1]
        self.grad1_2_2 = (self.hidden_out2 - self.output) * (self.hidden_out2 * (1 - self.hidden_out2)) * \
                         (self.hidden2[1]) * (self.hidden_out1[:, 1] * (1 - self.hidden_out1[:, 1])) * self.input[:, 1]

        self.hidden1[0][0] -= self.lr * np.sum(self.grad1_1_1) / 4
        self.hidden1[0][1] -= self.lr * np.sum(self.grad1_1_2) / 4
        self.hidden1[1][0] -= self.lr * np.sum(self.grad1_2_1) / 4
        self.hidden1[1][1] -= self.lr * np.sum(self.grad1_2_2) / 4


xor = XOR()

for i in range(10001):
    pred = xor.Predict()
    xor.Update()

    if i % 1000 == 0:
        print('epoch : ', i + 1)
        loss = xor.GetLoss(pred)
        #print('pred : ', pred)
        print('loss : ', loss)
        print('predict : ', xor.GetPred())
        print('accuracy : ', xor.GetAccuracy())
        print()
