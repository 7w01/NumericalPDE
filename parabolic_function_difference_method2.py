import numpy as np

def FuncPhi(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

def MatrixStep1(NX, NY, a, r):
    M = NX * NY
    A = np.zeros((M, M))

    for i in range(M):
        A[i, i] = 1 + 2 * a * r
        if i + NX < M:
            A[i, i + NX] = -a * r / 2
        if i - NX >= 0:
            A[i, i - NX] = -a * r / 2

    return A

def MatrixStep2(NX, NY, a, r):
    M = NX * NY
    A = np.zeros((M, M))

    for i in range(M):
        A[i, i] = 1 + 2 * a * r
        if (i + 1) % NX != 0:
            A[i, i + 1] = -a * r / 2
        if i % NX != 0:
            A[i, i - 1] = -a * r / 2

    return A

def DecomposeTriMatrix(A, M, NX):
    L = np.zeros((M, M))
    U = np.zeros((M, M))

    for i in range(M):
        L[i, i] = 1
        U[i, i] = A[i, i]

        if i - NX >= 0:
            L[i, i - NX] = A[i, i - NX] / U[i - NX, i - NX]
            U[i, i - NX] = A[i - NX, i]

    return L, U

def ForwardSolver(L, M, NX, b):
    x = np.zeros(M)
    for i in range(M):
        if i - NX >= 0:
            x[i] = b[i] - L[i, i - NX] * x[i - NX]
        else:
            x[i] = b[i]
    return x

def BackwardSolver(U, M, NX, b):
    x = np.zeros(M)
    for i in reversed(range(M)):
        if i + NX < M:
            x[i] = (b[i] - U[i, i + NX] * x[i + NX]) / U[i, i]
        else:
            x[i] = b[i] / U[i, i]
    return x


# 测试数据
l1 = 1
l2 = 1
T = 1
tau = 1 / 1600
a = 1 / 16
h = 1 / 40
r = tau / h ** 2

NT = int(np.ceil(T / tau))
NX = int(np.ceil(l1 / h) + 1)
NY = int(np.ceil(l2 / h) + 1)

u = np.zeros((NT + 1, NX * NY))
b = np.zeros(NX * NY)
tempu = np.zeros(NX * NY)

x = np.arange(0, l1 + h, h)
y = np.arange(0, l2 + h, h)

# ADI 矩阵: A1, A2 为三对角矩阵，半带宽为 2 和 NX+1
A1 = MatrixStep1(NX, NY, a, r)
A2 = MatrixStep2(NX, NY, a, r)

# Doolittle 分解
L1, U1 = DecomposeTriMatrix(A1, NX * NY, 1)
L2, U2 = DecomposeTriMatrix(A2, NX * NY, NX)

# 初始条件
for i in range(NY):
    u[0, i * NX : (i + 1) * NX] = FuncPhi(x, y[i])

# ADI 迭代
for i in range(NT):
    # ADI Step 1:
    # 内点
    for j in range(1, NX - 1):
        for k in range(1, NY - 1):
            temp1 = j + k * NX
            temp2 = temp1 + 1
            temp3 = temp1 - 1
            temp4 = temp1 + NX
            temp5 = temp1 - NX

            b[temp1] = (1 - a * r) * u[i, temp1] + (a * r / 2) * u[i, temp4] + (a * r / 2) * u[i, temp5]

    # Gamma 1\cup Gamma 3
    for j in range(1, NX - 1):
        temp1 = j
        temp4 = temp1 + NX

        b[temp1] = (1 - a * r) * u[i, temp1] + a * r * u[i, temp4]

        # Gamma 3
        temp1 = j + (NY - 1) * NX
        temp2 = temp1 + 1
        temp3 = temp1 - 1
        temp5 = temp1 - NX

        b[temp1] = (1 - a * r) * u[i, temp1] + a * r * u[i, temp5]

    # Gamma2 \cup Gamma 4
    for k in range(NY):
        temp1 = 1 + k * NX
        b[temp1] = 0

        temp1 = NX - 1 + k * NX
        b[temp1] = 0

    # u_{n+1/2}
    tempb = ForwardSolver(L1, NX * NY, 1, b)
    tempu = BackwardSolver(U1, NX * NY, 1, tempb)

    # ADI Step 2:
    b = np.zeros(NX * NY)
    # 内点
    for j in range(1, NX - 1):
        for k in range(1, NY - 1):
            temp1 = j + k * NX
            temp2 = temp1 + 1
            temp3 = temp1 - 1
            temp4 = temp1 + NX
            temp5 = temp1 - NX

            b[temp1] = (1 - a * r) * tempu[temp1] + (a * r / 2) * tempu[temp2] + (a * r / 2) * tempu[temp3]

    # Gamma 1\cup Gamma 3
    for j in range(1, NX - 1):
        temp1 = j
        temp2 = temp1 + 1
        temp3 = temp1 - 1
        temp4 = temp1 + NX

        b[temp1] = (1 - a * r) * tempu[temp1] + (a * r / 2) * tempu[temp2] + (a * r / 2) * tempu[temp3]

        # Gamma 3
        temp1 = j + (NY - 1) * NX
        temp2 = temp1 + 1
        temp3 = temp1 - 1
        temp5 = temp1 - NX

        b[temp1] = (1 - a * r) * tempu[temp1] + (a * r / 2) * tempu[temp2] + (a * r / 2) * tempu[temp3]

    # Gamma2 \cup Gamma 4
    for k in range(NY):
        temp1 = 1 + k * NX
        b[temp1] = 0

        temp1 = NX - 1 + k * NX
        b[temp1] = 0

    # u_{n+1/2}
    tempb = ForwardSolver(L2, NX * NY, NX, b)
    tempu = BackwardSolver(U2, NX * NY, NX, tempb)
    u[i + 1, :] = tempu

# 真解
utrue = np.zeros_like(u)
for i in range(NY):
    utrue[0, i * NX : (i + 1) * NX] = np.exp(-np.pi ** 2 / 8 * T) * np.sin(np.pi * x) * np.cos(np.pi * y[i])

# 检验
tempNx = [11, 21, 31]
tempNy = [11, 21, 31]

for i in range(3):
    tempk = tempNx[i] + (tempNy[i] - 1) * NX
    print(u[NT, tempk], utrue[tempk])
