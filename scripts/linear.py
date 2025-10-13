import numpy as np
import matplotlib.pyplot as plt

def normalize(array:list):
    #noramalize with std method (standardization)
    arr = np.array(array)
    return (array - np.mean(arr))/np.std(arr)


def linregression_lr(xline: list, yline: list, epochs: int = 100, lr: float = 0.0001):

    #initialize the var
    n = len(xline)
    m = 0
    c = 0
    qcheck: list = []
    ccheck: list = []
    mcheck: list = []
    #need to convert 2 np array
    arrX = np.array(xline)
    arrY = np.array(yline)

    #normalize the arrays
    normX = normalize(arrX)
    normY = normalize(arrY)

    #i personally use a training cicle
    for i in range(epochs):
        #prediction of the y
        yp = (m*normX + c)
        #dm/dy, dc/dy are the gradient
        dm = (-2/n) * sum(normX * (normY - yp))
        dc = (-2/n) * sum(normY - yp)
        #updating the parameters
        m = m - lr * dm
        c = c - lr * dc
        #quadratic error checking

        mse = np.mean((normY- yp) ** 2)
        qcheck.append(mse)
        ccheck.append(c)
        mcheck.append(m)

        if i % 100 == 0:
            print(f"ep n -> {i}: medium sqsuare error -> {mse:.6f}")


    return m, c, qcheck, ccheck, mcheck
