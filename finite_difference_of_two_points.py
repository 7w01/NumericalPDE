import numpy as np
import matplotlib.pyplot as plt
from comSimpson import comSimpson


MK = 100

errC = np.zeros(MK)
errL = np.zeros(MK)
errH = np.zeros(MK)
errH1 = np.zeros(MK)
err2 = np.zeros(MK)
errf = np.zeros(MK)


def CoefQ(x):
    return np.pi**2/4


def CoefP(x):
    return 1


def CoefF(x):
    return np.pi**2*np.sin(np.pi*x/2)/2


for ni in range(0, MK):
    n = 10 + 10 * ni
    a = 0
    b = 1

    A = np.zeros((n, n))
    y = np.zeros(n)
    h = (b - a) / n
    p = a + np.linspace(0, n, n+1)*h

    for i in range(0, n-1):
        tempx1 = (p[i + 1] + p[i]) / 2
        tempx2 = (p[i + 2] + p[i + 1]) / 2

        ai1 = CoefP(tempx1)
        ai2 = CoefP(tempx2)
        di = CoefQ(p[i + 1])
        phii = CoefF(p[i + 1])

        A[i][i] = ai2 / h + ai1 / h + h * di
        A[i][i + 1] = -ai2 / h
        if i != 1:
            A[i][i - 1] = -ai1 / h

        y[i] = h * phii

    A[n-1][n-1] = CoefP(b - h / 2) / h + CoefQ(b) / 2 * h
    A[n-1][n-2] = -CoefP(b - h / 2) / h
    y[n-1] = CoefF(b) * h / 2

    u = np.linalg.solve(A, y)
    u = np.insert(u, 0, 0, axis=0)

    utrue = np.sin(np.pi / 2 * p)

    for i in range(1, n):
        temp = abs(u[i] - utrue[i])
        if temp > errC[ni]:
            errC[ni] = temp

    for i in range(1, n):
        temp = u[i] - utrue[i]
        errL[ni] += h * temp ** 2

    for i in range(1, n + 1):
        temp1 = u[i] - utrue[i]
        temp2 = u[i - 1] - utrue[i - 1]
        errH[ni] = errH[ni] + (temp1 - temp2) ** 2 / h

    errH[ni] += errL[ni]

    errH[ni] = np.sqrt(errH[ni])
    errL[ni] = np.sqrt(errL[ni])

    for i in range(0, n):
        x1 = p[i]
        x2 = p[i + 1]

        M = 10
        hm = 1 / (2 * M)
        xi = np.linspace(0,2*M,2*M+1)*hm

        tempu = u[i] * (1 - xi) + u[i + 1] * xi
        tempv = np.sin(np.pi / 2 * (x1 + h * xi))

        tempdu = (u[i + 1] - u[i]) / h
        tempdv = np.pi / 2 * np.cos(np.pi / 2 * (x1 + h * xi))

        temp = ((tempu - tempv)**2) * h
        err2[ni] = err2[ni] + comSimpson(temp, hm)

        temp = (tempdu - tempdv) ** 2 * h
        errf[ni] = errf[ni] + comSimpson(temp, hm)

    errH1[ni] = np.sqrt(errf[ni] + err2[ni])

N=10+10*np.arange(0,MK)

e1 = plt.subplot(2, 2, 1)
e1.plot(N, errC)
e1.set_xlabel('迭代数', fontproperties="SimHei")
e1.set_ylabel('errC', fontproperties="SimHei")
e1.set_title('C误差', fontproperties="SimHei")

e2 = plt.subplot(2, 2, 2)
e2.plot(N, errL)
e2.set_xlabel('迭代数', fontproperties="SimHei")
e2.set_ylabel('errL', fontproperties="SimHei")
e2.set_title('0误差', fontproperties="SimHei")

e3 = plt.subplot(2, 2, 3)
e3.plot(N, errH)
e3.set_xlabel('迭代数', fontproperties="SimHei")
e3.set_ylabel('errH', fontproperties="SimHei")
e3.set_title('H1误差', fontproperties="SimHei")

e4 = plt.subplot(2, 2, 4)
e4.plot(N, errH1)
e4.set_xlabel('迭代数', fontproperties="SimHei")
e4.set_ylabel('errH1', fontproperties="SimHei")
e4.set_title('区间上的H1误差', fontproperties="SimHei")

plt.show()
