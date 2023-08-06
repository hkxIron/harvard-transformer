import numpy as np
"""
https://spaces.ac.cn/archives/7180
推论1： 高维空间中的任意两个随机向量几乎都是垂直的。

推论2： 从N(0,1/n) 中随机选取n*n个数，组成一个n×n 的矩阵，这个矩阵近似为正交矩阵，且n 越大，近似程度越好
"""
n = 1000
W = np.random.randn(n, n) / np.sqrt(n)
X = np.dot(W.T, W)  # 矩阵乘以自身的转置
print(X)  # 看看是否接近单位阵
print(np.square(X - np.eye(n)).mean())  # 计算与单位阵的mse