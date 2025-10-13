import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from module import linear_regression

def main():
    data_frame = pd.read_csv('data.csv')

    x = data_frame[['depth']
    y = data_frame[['distance']

    b1, b0 = linear_regression(x, y)

    plt.scatter(x, y, color='blue')
    plt.plot(x, b0 + b1 * x, color='red')
    plt.xlabel('Depth')
    plt.ylabel('Distance')
    plt.title(f'Linear Regression: y = {b0[0]:.2f} + {b1[0][0]:.2f}x')

if __name__ == "__main__":
    main()
