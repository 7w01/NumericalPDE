import numpy as np
import math
import matplotlib.pyplot as plt


def FuncPhi(x):
    return np.sin(math.pi * x)


def Funcf(x, t):
    return np.sin(math.pi * x) + math.pi ** 2 * t * np.sin(math.pi * x)


l = 1
a = 1
T = 1
tau = 1 / 3200

h = [1 / 10, 1 / 20, 1 / 30, 1 / 40, 1 / 80, 1 / 160]

r = a * tau / np.array(h) ** 2
err0 = np.zeros(len(h))

for hi in range(len(h)):
    NT = math.ceil(T / tau)
    NX = math.ceil(l / h[hi]) + 1

    u = np.zeros((NT + 1, NX))
    x = np.arange(0, l + h[hi], h[hi])[:NX]

    A = np.zeros((NX, NX))
    b = np.zeros(NX)

    # 初始条件
    u[0, :] = FuncPhi(x)

    for i in range(NT):
        # 内点
        for j in range(1, NX - 1):
            A[j, j + 1] = -r[hi]
            A[j, j] = (1 + 2 * r[hi])
            A[j, j - 1] = -r[hi]
            b[j] = u[i, j] + tau * Funcf(x[j], i * tau)

        # 边值条件处理
        A[0, :] = 0
        A[0, 0] = 1
        b[0] = 0
        A[NX - 1, :] = 0
        A[NX - 1, NX - 1] = 1
        b[NX - 1] = 0

        u[i + 1, :] = np.linalg.solve(A, b)

    ## 检验在 t=T 处的 0-范数
    # 真解 u(x, t) = exp(-pi^2 t)*sin(pi*x) + t*sin(pi*x)
    utrueT = np.exp(-math.pi ** 2 * T) * np.sin(math.pi * x) + T * np.sin(math.pi * x)

    erru = utrueT - u[NT, :]
    err0[hi] = np.sqrt(np.dot(erru, erru) * h[hi])

plt.loglog(h, err0, '-*r', label='err0')
plt.loglog(h, np.array(h) ** 2, '-*k', label='O(h^2)')
plt.legend()
plt.show()
