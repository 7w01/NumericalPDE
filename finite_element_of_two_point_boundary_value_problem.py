import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from comSimpson import comSimpson

M = 21

#两点边值问题
a = 0
b = 1
p = 1
q = np.pi ** 2 / 4


def f(x):
    return ((np.pi**2)/2)*np.sin(np.pi*x/2)


def U(x):
    return np.sin(np.pi*x/2)


def dU(x):
    return np.pi/2*np.cos(np.pi*x/2)

dif_U = np.zeros(M) #数值解与真解的差值平方的积分
dif_dU = np.zeros(M) #数值解与真解的导数的差值平方的积分
err_H1 = np.zeros(M)
err_L2 = np.zeros(M)
Cond = np.zeros(M)

for ni in range(0, M):
    n = 10 + ni*10

    A = np.zeros((n, n))
    B = np.zeros(n)
    i1 = np.arange(0, n)
    i2 = np.arange(1, n + 1)
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)

    #comSimpson
    xi = np.linspace(0, 1, 101)
    hI = 0.01

    #计算a,b
    temp = -p / h + h * q * (1 - xi) * xi
    a1 = comSimpson(temp, hI)

    temp = -p / h + h * q * (1 - xi) * xi
    a2 = comSimpson(temp, hI)

    temp = p / h + h * q * xi * xi
    a3 = comSimpson(temp, hI)

    temp = p / h + h * q * (1 - xi) * (1 - xi)
    a4 = comSimpson(temp, hI)

    for i in range(1, n):
        x1 = x[i1[i]]
        x2 = x[i2[i]]

        A[i1[i] - 1, i2[i] - 1] += a1
        A[i2[i] - 1, i1[i] - 1] += a2
        A[i2[i] - 1, i2[i] - 1] += a3
        A[i1[i] - 1, i1[i] - 1] += a4

        temp = h * f(x1 + h * xi) * (1 - xi)
        b1 = comSimpson(temp, hI)

        temp = h * f(x1 + h * xi) * xi
        b2 = comSimpson(temp, hI)

        B[i1[i] - 1] += b1
        B[i2[i] - 1] += b2

    #处理左端点
    x1 = i1[0]
    x2 = i2[0]

    temp = h * f(x1 + h * xi) * xi
    b2 = comSimpson(temp, hI)

    A[0][0] += a3
    B[0] += b2

    #解矩阵
    Un = nl.solve(A, B)

    Un = np.insert(Un, 0, 0, axis=0)

    Cond[ni] = nl.cond(A)

    #计算误差
    for i in range(0, n):
        x1 = x[i1[i]]
        x2 = x[i2[i]]

        unit_Un = Un[i1[i]]*(1-xi)+Un[i2[i]]*xi
        unit_U = U(x1 + h*xi)

        unit_dUn = (Un[i2[i]]-Un[i1[i]])/h
        unit_dU = dU(x1 + h*xi)

        dif_U[ni] += comSimpson(((unit_Un-unit_U)**2)*h, hI)
        dif_dU[ni] += comSimpson(((unit_dUn - unit_dU)**2)*h, hI)

    #L2误差
    err_L2[ni] = np.sqrt(dif_U[ni])

    #H1误差
    err_H1[ni] = np.sqrt(dif_U[ni] + dif_dU[ni])

#画图
N = np.linspace(10, 10*M -10, M)

p_L2 = plt.subplot(2, 1, 1)
p_L2.plot(N, err_L2)
p_L2.set_xlabel('区间等分数', fontproperties="SimHei")
p_L2.set_ylabel('error', fontproperties="SimHei")
p_L2.set_title('L2误差', fontproperties="SimHei")

p_H1 = plt.subplot(2, 1, 2)
p_H1.plot(N, err_H1)
p_H1.set_xlabel('区间等分数', fontproperties="SimHei")
p_H1.set_ylabel('error', fontproperties="SimHei")
p_H1.set_title('H1误差', fontproperties="SimHei")

plt.show()
