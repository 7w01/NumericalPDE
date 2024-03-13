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
cond = np.zeros(M)

for ni in range(0, M):
    n = 10 + ni*10

    A = np.zeros((2*n+1, 2*n+1))
    B = np.zeros(2*n+1)
    i = np.zeros((3, n))
    i[0] = np.arange(0, 2 * n, 2)
    i[1] = np.arange(1, 2 * n, 2)
    i[2] = np.arange(2, 2 * n + 1, 2)
    i = i.astype('int')

    h = (b - a) / n
    x = np.linspace(a, b, 2*n + 1)

    #comSimpson
    I_n = 100
    xi = np.linspace(0, 1, I_n + 1)
    hI = 1/I_n

    F = np.zeros((6, I_n + 1))
    F[0] = (2 * xi - 1) * (xi - 1)
    F[2] = (2 * xi - 1) * xi
    F[1] = 4 * xi * (1 - xi)
    F[3] = 4 * xi - 3
    F[5] = 4 * xi - 1
    F[4] = 4 - 8 * xi

    for j in range(0, n):
        x1 = x[i[0][j]]
        x1_5 = x[i[1][j]]
        x2 = x[i[2][j]]

        temp_A = np.zeros((3, 3))
        temp_B = np.zeros(3)

        #计算temp
        for index1 in range(0, 3):
            for index2 in range(0, 3):
                temp_A[index1][index2] = comSimpson(q*F[index1]*F[index2]*h+p*F[index1+3]*F[index2+3]/h, hI)
            temp_B[index1] = comSimpson(f(x1 + h*xi)*F[index1], h*hI)

        #组装刚度矩阵
        for index1 in range(0, 3):
            for index2 in range(0, 3):
                A[i[index1][j]][i[index2][j]] += temp_A[index1][index2]
            B[i[index1][j]] += temp_B[index1]

    #处理左端点
    A = A[1:, 1:]
    B = B[1:]

    #解矩阵
    Un = nl.solve(A, B)

    Un = np.insert(Un, 0, 0, axis=0)

    cond[ni] = nl.cond(A)

    # 计算误差
    for j in range(0, n):
        x1 = x[i[0][j]]

        unit_Un = Un[i[0][j]] * F[0] + Un[i[1][j]] * F[1] + Un[i[2][j]] * F[2]
        unit_U = U(x1 + h * xi)

        unit_dUn = Un[i[0][j]]*F[3]/h + Un[i[1][j]]*F[4]/h + Un[i[2][j]]*F[5]/h
        unit_dU = dU(x1 + h * xi)

        dif_U[ni] += comSimpson(((unit_Un - unit_U) ** 2) * h, hI)
        dif_dU[ni] += comSimpson(((unit_dUn - unit_dU) ** 2) * h, hI)

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
