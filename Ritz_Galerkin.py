import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt
from comSimpson import comSimpson


def U(x):
    return np.sin(x)/np.sin(1)-x


#基底维数
M = 100

#number of nodes
n = 1001

#步长
h = 1/(n - 1)

x = np.linspace(0, 1, n)

y = U(x)

error1 = np.zeros(M)

error2 = np.zeros(M)

cond1 = np.zeros(M)

cond2 = np.zeros(M)

for N in range(1, M):
    A = np.zeros((N, N))
    b = np.zeros((N, 1))

    for i in range(1, N + 1):
        A[i - 1][i - 1] = (pow((i*np.pi), 2)-1)/2
        b[i - 1][0] = -np.cos(i*np.pi)/(i*np.pi)

    C = nl.pinv(A).dot(b)

    Un = np.zeros(n)

    for i in range(1, N + 1):
        Un += C[i - 1]*np.sin(i*np.pi*x)

    d = (Un-y)**2

    error1[N] = pow(comSimpson(d, h), 0.5)

    cond1[N] = nl.cond(A, 2)


for N in range(1, M):
    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            A[i - 1][j - 1] = i * j/(i + j - 1) - (2 * i * j + i + j)/(i + j) \
                      + (i * j + i + j)/(i + j + 1) + 2/(i + j + 2) - 1/(i + j + 3)
        b[i - 1] = 1 / (i + 2) - 1 / (i + 3)

    C = nl.solve(A, b)

    Un = np.zeros(n)

    for i in range(1, N + 1):
        Un += C[i - 1]*x*(1-x)*(x**(i - 1))

    d = (Un-y)**2

    error2[N] = np.sqrt(comSimpson(d, h))

    cond2[N] = nl.cond(A, 2)

M = np.linspace(1, 100, 100)

e1 = plt.subplot(2, 2, 1)
e1.plot(M, error1)
e1.set_xlabel('维数', fontproperties="SimHei")
e1.set_ylabel('error', fontproperties="SimHei")
e1.set_title('基底1的误差', fontproperties="SimHei")

e2 = plt.subplot(2, 2, 2)
e2.plot(M, error2)
e2.set_xlabel('维数', fontproperties="SimHei")
e2.set_ylabel('error', fontproperties="SimHei")
e2.set_title('基底2的误差', fontproperties="SimHei")

e3 = plt.subplot(2, 2, 3)
e3.plot(M, cond1)
e3.set_xlabel('维数', fontproperties="SimHei")
e3.set_ylabel('cond', fontproperties="SimHei")
e3.set_title('基底1的条件数', fontproperties="SimHei")

e4 = plt.subplot(2, 2, 4)
e4.plot(M, cond2)
e4.set_xlabel('维数', fontproperties="SimHei")
e4.set_ylabel('cond', fontproperties="SimHei")
e4.set_title('基底2的条件数', fontproperties="SimHei")

plt.show()
