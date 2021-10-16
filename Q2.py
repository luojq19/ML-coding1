# -*- codeing = utf-8 -*-
# @Time:  4:40 下午
# @Author: Jiaqi Luo
# @File: Q2.py
# @Software: PyCharm
import numpy as np

def square_norm(X):
    return np.dot(X.T, X)

# initialize the weights with a normal distribution of mean=0, scale=1
def weight_init(size):
    return np.random.normal(0, 1, size=size)

class Regression:
    # The class Regression is initialized with regularization parameter lamda and initial weight
    def __init__(self, weight, lamda):
        self.w = weight
        self.lamda = lamda

    # The forward function Y = Xw
    def forward(self, X):
        return np.dot(X, self.w)

    # loss_func = 1 / (2N) * (Xw - Y)^2, the basic loss function
    def loss_func(self, X, Y):
        N = Y.size
        return 1 / (2 * N) * square_norm(np.dot(X, self.w) - Y)

    # The gradient of loss = 1 / N * X^T(Xw - Y)
    def loss_grad(self, X, Y):
        N = Y.size
        return 1 / N * np.dot(X.T, (np.dot(X, self.w) - Y))

    # The ridge regression training: at time step t, w_t = w_t_hat * (1 - lr * lamda) i.e. weight decay
    def train_ridge(self, X, Y, num_epochs, lr):
        for i in range(num_epochs):
            grad = self.loss_grad(X, Y)
            self.w -= lr * grad
            self.w *= (1 - lr * self.lamda)
            if i % 10 == 9:
                print('Epoch %d: loss %.3f' % (i + 1, self.loss_func(X, Y)))
        return self.w

    # The Lasso regression training: at time step t,
    # if w_t_hat[i] > lr * lamda, w_t[i] = w_t_hat[i] - lr * lamda
    # if w_t_hat[i] < -lr * lamda, w_t[i] = w_t_hat[i] + lr * lamda
    # else, w_t[i] = 0
    def train_lasso(self, X, Y, num_epochs, lr):
        for i in range(num_epochs):
            grad = self.loss_grad(X, Y)
            self.w -= lr * grad
            self.w[abs(self.w) <= lr * self.lamda] = 0
            self.w += -lr * self.lamda * (self.w > lr * self.lamda) + lr * self.lamda * (self.w < -lr * self.lamda)
            if i % 10 == 9:
                print('Epoch %d: loss %.3f' % (i + 1, self.loss_func(X, Y)))
        return self.w


if __name__ == '__main__':
    X = np.array([[2, 1], [4, 0]])
    Y = np.array([[1], [3]])
    lamda = 1
    weight = weight_init((Y.size, 1))
    print(weight)
    ridge = Regression(weight, lamda)
    lasso = Regression(weight, lamda)
    num_epochs = 100
    learning_rate = 0.05

    print("Ridge Regression: ")
    w1 = ridge.train_ridge(X, Y, num_epochs, lr=learning_rate)
    print("The optimized weight = ", w1)
    print("\nLasso Regression: ")
    w2 = lasso.train_lasso(X, Y, num_epochs, lr=0.05)
    print("The optimized weight = ", w2)














