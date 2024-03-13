import numpy as np
import matplotlib.pyplot as plt

a1 = 0
a2 = 1
a3 = 0
a4 = 1

NK = 15
errC = np.zeros(NK)
errL = np.zeros(NK)
H0err = np.zeros(NK)
meshsize = np.zeros(NK)

for ni in range(0, NK):
    N = 5 * (ni+1)
    M = 10 * (ni+1)

    A = np.zeros(((N + 1) * (M + 1), (N + 1) * (M + 1)))
    b = np.zeros((N + 1) * (M + 1))

    hx = (a2 - a1) / M
    hy = (a4 - a3) / N

    Gamma1 = np.linspace(1, M+1, M+1)
    Gamma4 = 1 + np.linspace(0, N, N+1)*(M + 1)

    Gamma2 = (M + 1) * np.linspace(1, N+1, N+1)
    Gamma3 = N * (M + 1) + np.linspace(1, M+1, M+1)

    Gamma1 = Gamma1.astype('int')
    Gamma2 = Gamma2.astype('int')
    Gamma3 = Gamma3.astype('int')
    Gamma4 = Gamma4.astype('int')

    for i in range(1, M):
        for j in range(1, N):
            k = j * (M + 1) + i
            tempx = a1 + i * hx
            tempy = a3 + j * hy

            A[k][k] = -2 / hy ** 2 - 2 / hx ** 2 + 2 * np.pi ** 2
            A[k][k - 1] = 1 / hx ** 2
            A[k][k + 1] = 1 / hx ** 2
            A[k][k + M + 1] = 1 / hy ** 2
            A[k][k - M - 1] = 1 / hy ** 2
            b[k] = 2 * np.pi ** 2 * tempx * tempy

    for i in range(1, len(Gamma2) - 1):
        tempx = a2
        tempy = i * hy + a3
        tempy1 = tempy + 0.5 * hy
        tempy2 = tempy - 0.5 * hy

        A[Gamma2[i]-1][Gamma2[i]-1] = -1 / hy ** 2 - 1 / hx ** 2 + 2 * np.pi ** 2 / 2
        A[Gamma2[i]-1][Gamma2[i] - 2] = 1 / hx ** 2
        A[Gamma2[i]-1][Gamma2[i] + M] = 1 / (2 * hy ** 2)
        A[Gamma2[i]-1][Gamma2[i] - M - 2] = 1 / (2 * hy ** 2)
        b[Gamma2[i]-1] = np.pi ** 2 * tempx * tempy \
            - ((0.5 * tempy1 ** 2 + np.cos(np.pi * tempy1)) - (0.5 * tempy2 ** 2 + np.cos(np.pi * tempy2))) / hx / hy

    for i in range(1, len(Gamma3) - 1):
        tempx = i * hx + a1
        tempy = a4
        tempx1 = tempx + 0.5 * hx
        tempx2 = tempx - 0.5 * hx

        A[Gamma3[i]-1][Gamma3[i]-1] = -1 / (hy ** 2) - 1 / hx ** 2 + 2 * np.pi ** 2 / 2
        A[Gamma3[i]-1][Gamma3[i] - 2] = 1 / (2 * hx ** 2)
        A[Gamma3[i]-1][Gamma3[i]] = 1 / (2 * hx ** 2)
        A[Gamma3[i]-1][Gamma3[i] - M - 2] = 1 / hy ** 2
        b[Gamma3[i]-1] = np.pi ** 2 * tempx * tempy\
        - ((0.5 * tempx1 ** 2 + np.cos(np.pi * tempx1))- (0.5 * tempx2 ** 2 + np.cos(np.pi * tempx2))) / hx / hy

    tempx = a2
    tempy = a4
    tempx2 = a2 - 0.5 * hx
    tempy2 = a4 - 0.5 * hy
    A[(N + 1) * (M + 1)-1][(N + 1) * (M + 1)-1] = -1 / (2 * hx ** 2) - 1 / (2 * hy ** 2) + 2 * np.pi ** 2 / 4
    A[(N + 1) * (M + 1)-1][(N + 1) * (M + 1) - 2] = 1 / (2 * hx ** 2)
    A[(N + 1) * (M + 1)-1][(N + 1) * (M + 1) - M - 2] = 1 / (2 * hy ** 2)
    b[(N + 1) * (M + 1)-1] = 0.5 * np.pi ** 2 * tempx * tempy - ((0.5 * tempx ** 2 + np.cos(np.pi * tempx))
  - (0.5 * tempx2 ** 2 + np.cos(np.pi * tempx2))) / hx / hy-((0.5 * tempy ** 2 + np.cos(np.pi * tempy))
                                                             - (0.5 * tempy2 ** 2 + np.cos(np.pi * tempy2))) / hx / hy

    A[Gamma1-1,:]=0
    for i in range(0, len(Gamma1)):
        A[Gamma1[i]-1][Gamma1[i]-1] = 1
        b[Gamma1[i]-1] = 0

    A[Gamma4-1,:] = 0
    for i in range(0, len(Gamma4)):
        A[Gamma4[i]-1][Gamma4[i]-1] = 1
        b[Gamma4[i]-1] = 0

    u = np.linalg.solve(A,b)

    utrue = np.zeros((N + 1) * (M + 1))
    for i in range(0, M+1):
        tempx = a1 + i * hx
        for j in range(0, N+1):
            k = j * (M + 1) + i
            tempy = a3 + j * hy
            utrue[k] = tempx * tempy + np.sin(np.pi * tempx) * np.sin(np.pi * tempy)

    errC[ni] = np.max(np.abs(utrue - u))

    errL[ni] = (utrue-u).dot((utrue-u)*hx*hy)
    errL[ni] = np.sqrt(errL[ni])

    meshsize[ni] = np.sqrt(hx ** 2 + hy ** 2)

e1 = plt.subplot(2, 1, 1)
e1.plot(meshsize, errL)
e1.set_xlabel('网格边长', fontproperties="SimHei")
e1.set_ylabel('errL', fontproperties="SimHei")
e1.set_title('0-norm error', fontproperties="SimHei")

e2 = plt.subplot(2, 1, 2)
e2.plot(meshsize, meshsize**2)
e2.set_xlabel('网格边长', fontproperties="SimHei")
e2.set_ylabel('errL', fontproperties="SimHei")
e2.set_title('O(h^2)', fontproperties="SimHei")

plt.show()