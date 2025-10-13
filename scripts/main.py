import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear import linregression_lr
import os

def main() -> None:

    data = pd.read_csv('../datasource/data.csv',  usecols = ['depth', 'distance'])

    x = data['depth'].to_numpy()
    y = data['distance'].to_numpy()

    ep = 5000
    lr = 0.0001

    b1, b0, mse_debug, c_debug, m_debug = linregression_lr(x, y, ep, lr)

    '''
    #creating a csv file to analize the
    df = pd.DataFrame({
    'epoch': epochs,
    'm': m_debug,
    'm'
    'c': c_debug,
    'mse': mse_debug
    })

    #got the folder 'debug data output'
    output_dir = os.path.join('debug_data_output')
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'regression_debug.csv')
    df.to_csv(csv_path, index=False)

    print(f'✅ File saved in : {csv_path}')
    '''
    plt.scatter(x, y, color='blue')
    plt.plot(x, b0 + b1 * x, color='red')
    plt.xlabel('Depth')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.title(f'Linear Regression: y = {b0:.2f} + {b1:.2f}x')
    plt.tight_layout()

    epochs = [epoch for epoch in range(ep)]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))

    #quadratic error subplot of debugging
    axs[0].plot(epochs, mse_debug, color='green')
    axs[0].set_title('MSE over epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('MSE')
    axs[0].grid(True)

    #m vals subplot of debugging
    axs[1].plot(epochs, m_debug, color='orange')
    axs[1].set_title('m (slope) over epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('m')
    axs[1].grid(True)


    #c vals subplot of debugging
    axs[2].plot(epochs, c_debug, color='purple')
    axs[2].set_title('c (intercept) over epochs')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('c')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


