import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import MinMaxScaler




#debug logging config to analize the process of the file
logging.basicConfig(
    level=logging.DEBUG,
    format=" %(name)s - %(levelname)s - %(message)s",
)

libraries = [
'numpy',
'pandas',
'matplotlib',
'PIL',
'sklearn'
]

for name in libraries:
    logging.getLogger(name).setLevel("WARNING")


logger = logging.getLogger(__name__)

def normalize(array:list):
    #noramalize with std metho (standardization)
    arr = np.array(array)
    return (array - np.mean(arr))/np.std(arr)

def norm_minmax(array: list) -> list:

    arr = np.array(array)
    min = np.min(arr)
    max = np.max(arr)

    if min == max:
        logger.critical('Array of copy of values')
    else:
        return np.array((arr - min) / (max - min))


def linregression_lr(xline: list, yline: list, epochs: int = 100, lr: float = 0.0001):


    n = len(xline)
    m = 0
    c = 0.1
    qcheck: list = []
    ccheck: list = []
    mcheck: list = []
    #need to convert 2 np array
    arrX = np.array(xline)
    arrY = np.array(yline)

    normX = normalize(arrX)
    normY = normalize(arrY)

    #i personally use a training cicle
    for i in range(epochs):

        try:
            yp = (m* normX + c)

            dm = (-2/n) * sum(normX * (normY - yp))
            dc = (-2/n) * sum(normY - yp)


            #updating the parameters
            m = m - lr * dm
            c = c - lr * dc
            mse = np.mean((normY- yp) ** 2)


            qcheck.append(mse)
            ccheck.append(c)
            mcheck.append(m)


            #visualazing the error and the parameters every 100 iteration
            if i % 100 == 0:
                logger.debug(f"epoch: {i}")
                logger.debug(f" medium sqsuare error -> {mse:.6f}")
                logger.debug(f" m param -> {m:.6f}")
                logger.debug(f" c param -> {c:.6f}")

        except Exception as Error:

            logger.critical(f'Caught an error in the processing {Error} ')




    return m, c, qcheck, ccheck, mcheck
