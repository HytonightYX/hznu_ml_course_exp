'''
基于最小二乘法实现线性回归
@author: husiyuan
'''
import numpy as np


class OLSLinearRegression:

    def _ols(self, X, y):
        '''最小二乘法估计w'''
        tmp = np.linalg.inv(np.matmul(X.T, X))
        tmp = np.matmul(tmp, X.T)
        return np.matmul(tmp, y)

    def _preprocess_data_X(self, X):
        '''数据预处理'''

        # 扩展X, 添加 X0 列并设置为1
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        '''训练模型'''

        # 预处理 X_train (添加 x0 = 1)
        X_train = self._preprocess_data_X(X_train)

        # 使用最小二乘法估算 w
        self.w = self._ols(X_train, y_train)

    def predict(self, X):
        '''预测'''
        # 预处理 X_train (添加 X0 = 1)
        X = self._preprocess_data_X(X)
        return np.matmul(X, self.w)


class GDLinearRegression:

    def __init__(self, n_iter=200, eta=1e-3, tol=None):
        # 训练迭代次数
        self.n_iter = n_iter
        # 学习率
        self.eta = eta
        # 误差变化阈值
        self.tol = tol
        # 模型参数w(训练时初始化)
        self.w = None

    def _loss(self, y, y_pred):
        '''计算损失'''
        return np.sum((y_pred - y) ** 2) / y.size

    def _gradient(self, X,  y, y_pred):
        '''计算梯度'''
        return np.matmul(y_pred - y, X) / y.size

    def _gradient_descent(self, w, X, y):
        '''梯度下降算法'''

        # 若用户指定 tol, 则启用早期停止法
        if self.tol is not None:
            loss_old = np.inf

        # 使用梯度下降,至多迭代 n_iter 次,更新w
        for step_i in range(self.n_iter):
            # 预测
            y_pred = self._predict(X, w)
            # 计算损失
            loss = self._loss(y, y_pred)
            print('%4i Loss: %s' % (step_i, loss))

            # 早期停止法
            if self.tol is not None:
                # 如果损失下降小于阈值,则终止迭代
                if loss_old - loss < self.tol:
                    break
                loss_old = loss

            # 计算梯度
            grad = self._gradient(X, y, y_pred)
            # 更新参数w
            w -= self.eta * grad

    def _preprocess_data_X(self, X):
        '''数据预处理'''

        # 扩展X, 添加 X0 列并设置为1
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        '''训练'''

        # 预处理 X_train (添加x0=1)
        X_train = self._preprocess_data_X(X_train)
        # 初始化参数向量w
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05

        # 执行梯度下降训练 w
        self._gradient_descent(self.w, X_train, y_train)

    def _predict(self, X, w):
        '''预测内部接口,实现函数h(x)'''
        # 预处理 X_train (添加 X0 = 1)
        return np.matmul(X, w)

    def predict(self, X):
        '''预测'''
        # 预处理 X_train (添加 X0 = 1)
        X = self._preprocess_data_X(X)
        return self._predict(X, self.w)

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split

    data = np.genfromtxt('exp1/dataset/winequality-red.csv',
                         delimiter=';', skip_header=True)
    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # 创建模型
    gd_lr = GDLinearRegression(n_iter=3000, eta=0.001, tol=0.00001)

    gd_lr.train(X_train, y_train)