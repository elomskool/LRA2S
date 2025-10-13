import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x, y):

    #dimensional check on the input received

    if len(x) != len(y):
        raise ValueError
    else:
        n = len(x)
        x = np.array(x)
        y = np.array(y)
        pass

    #summatory to find the bias

    sx = sum(x)
    sy = sum(y)
    sxx = sum(i**2 for i in range(n) )
    syy = sum(x[i]*y[i] for i in range(n) )

    #coeff and bias, see theory in readme.rm

    if (n*sxx)-(sxx**2) != 0:

        m = ((n * sxy)-(sx*sy))/((n*sxx)-(sxx**2))
        c = (sy-sx * m)/n

    return m, c
