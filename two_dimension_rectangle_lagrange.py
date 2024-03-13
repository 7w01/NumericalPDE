import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as itg
from comSimpson import comSimpson

M = 21

#两点边值问题
a1 = 0
a2 = 1
a3 = 0
a4 = 1
p = 2 * np.pi ** 2
q1 = -1
q2 = -1


def f(x, y):
    return 2 * x * y * np.pi ** 2


def U(x, y):
    return x * y + np.sin(np.pi * x) * np.sin(np.pi * y)


def dU(x):
    return np.pi/2*np.cos(np.pi*x/2)


for ni in range(0, M):
    n = 10 + ni*10

    A = np.zeros(((n+1)*(n+1), (n+1)*(n+1)))
    B = np.zeros((n+1)*(n+1))
    hx = (a2 - a1) / n
    hy = (a4 - a3) / n
    i = np.zeros((4, n * n))
    x = np.zeros((2, (n+1)*(n+1)))
    x_ = np.linspace(a1, a2, n + 1)
    y_ = np.linspace(a3, a4, n + 1)

    for s in range(0, n + 1):
        for k in range(0, n + 1):
           temp = s * (n + 1) + k
           x[0][temp] = x_[k]
           x[1][temp] = y_[k]

    for s in range(0, n):
        for k in range(0, n):
            temp = s * n + k
            i[0][temp] = s * (n + 1) + k
            i[1][temp] = i[0][temp] + 1
            i[2][temp] = i[0][temp] + n + 1
            i[3][temp] = i[1][temp] + n + 1


    def F_0(x, y):
        return (1-x)*(1-y)


    def F_1(x, y):
        return x*(1-y)


    def F_2(x, y):
        return (1-x)*y


    def F_3(x, y):
        return x*y




    for j in range(0, n * n):
        x1 = x[i[0][j]]
        x2 = x[i[1][j]]
        x3 = x[i[2][j]]
        x4 = x[i[3][j]]

        temp_A = np.zeros((4, 4))
        temp_B = np.zeros(4)

        #计算temp
        temp = lambda x,y: (p*(((1-x)*(1-y))**2) + q1*(y-1)*(y-1)/(hx**2) + q2*(x-1)*(x-1)/(hy**2)) * hx*hy
        A[i[0][j]][i[0][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x,y: (p*(1-x)*(1-y)*x*(1-y)+q1*(y-1)*(1-y)/(hx**2)+q2*x*(1-x)/(hy**2))*hx*hy
        A[i[0][j]][i[1][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)
        A[i[1][j]][i[0][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x,y: (p*(1-x)*(1-y)*y*(1-x) + q1*y*(1-y)/(hx**2) + q2*(x-1)*(1-x)/(hy**2)) * hx*hy
        A[i[0][j]][i[2][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)
        A[i[2][j]][i[0][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x,y: (p*(1-x)*(1-y)*x*y + q1*(y-1)*y/(hx**2) + q2*x*(x-1)/(hy**2)) * hx*hy
        A[i[0][j]][i[3][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)
        A[i[3][j]][i[0][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x, y: (p*((x*(1-y))**2) + q1*(1-y)*(1-y)/(hx**2) + q2*x*x/(hy**2)) * hx*hy
        A[i[1][j]][i[1][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x, y: (p * (((1 - x) * y)**2) + q1 * y*y / (hx ** 2) + q2 * (1-x) * (1-x)/ (hy ** 2)) * hx * hy
        A[i[2][j]][i[2][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x, y: (p * x * y * x*y + q1 * y * y / (hx ** 2) + q2 * x * x/ (hy ** 2)) * hx * hy
        A[i[3][j]][i[3][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x, y: (p * x * (1 - y) * (1-x) * y + q1 * y* y / (hx ** 2) + q2*x*(x-1)/ (hy ** 2)) * hx*hy
        A[i[1][j]][i[2][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)
        A[i[2][j]][i[1][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x, y: (p * x * (1 - y) * x * y + q1 * y* (1-y) / (hx ** 2) - q2 * x * x/ (hy ** 2)) * hx * hy
        A[i[1][j]][i[3][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)
        A[i[3][j]][i[1][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

        temp = lambda x, y: (p * x * y * (1 - x) * y - q1 * y * y / (hx ** 2) + q2*x*(1-x) / (hy ** 2)) * hx * hy
        A[i[2][j]][i[3][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)
        A[i[3][j]][i[2][j]] += itg.dblquad(temp, 0, 1, lambda x: 0, lambda x: 1)

