import numpy as np


def f1(x):
    y = x*np.sin(x)
    return y


def f2(x):
    y = np.sqrt(x)
    return y


def comSimpson(arr, h):
    result = 0
    for i, v in enumerate(arr):
        if i == 0 or i == len(arr) - 1:
            result += v
        elif i%2 == 1:
            result += 4*v
        else:
            result += 2*v
    return h*result/3


def comSimpson_app():
    I1_r = -2*np.pi

    I2_r = 2/3

    n1 = 10001

    n2 = 20001

    #function1
    h1 = 2*np.pi/(n1 - 1)

    h3 = 2*np.pi/(n2 - 1)

    x1 = np.linspace(0, 2*np.pi, n1)

    x3 = np.linspace(0, 2*np.pi, n2)

    y1 = f1(x1)

    y3 = f1(x3)

    I1 = comSimpson(y1, h1)

    I3 = comSimpson(y3, h3)

    #function2
    h2 = 1/(n1 - 1)

    h4 = 1/(n2 - 1)

    x2 = np.linspace(0, 1, n1)

    x4 = np.linspace(0, 1, n2)

    y2 = f2(x2)

    y4 = f2(x4)

    I2 = comSimpson(y2, h2)

    I4 = comSimpson(y4, h4)

    error1 = np.abs(I1 - I1_r)

    error2 = np.abs(I2 - I2_r)

    error3 = np.abs(I3 - I1_r)

    error4 = np.abs(I4 - I2_r)

    doc1 = (np.log(error1) - np.log(error3))/(np.log(h1) - np.log(h3))

    doc2 = (np.log(error2) - np.log(error4))/(np.log(h2) - np.log(h4))

    print("复化simpson公式")

    print("xsinx")

    print("积分为：", round(I1, 3))

    print("xsinx收敛阶为：", round(doc1, 1))

    print("x^0.5")

    print("积分为：", round(I2, 3))

    print("收敛阶为：", round(doc2, 1))

