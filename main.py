# main module developping

#import for the module linear
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from linear import linregression_lr

#import for the main file
import matplotlib.pyplot as plt
import os
import logging
import time
import csv
from datetime import datetime, timezone


#debug logging config to analize the porcess of the program
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

#main func developping
def main() -> None:

    start = time.time()

    logger.info(f"Starting time of the application: {start}")


    data = pd.read_csv('../datasource/data.csv',  usecols = ['depth', 'distance'])

    x = data['depth'].to_numpy()
    y = data['distance'].to_numpy()

    ep = 500
    lr = 0.1

    try:
        b1, b0, mse_debug, c_debug, m_debug = linregression_lr(x, y, ep, lr)
    except Exception as e:

        logger.critical(f'Error in the linear regression phase in linear.py')
        logger.debug(f'End time {time.time() - start}')
        quit()


    try:

        #configuring the file path for graphs
        folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'debug_data_output')
        os.makedirs(folder, exist_ok=True)
        logger.info(f'Currently working in: {os.path.abspath(folder)}')

        plt.scatter(x, y, color='blue')
        plt.plot(x, b0 + b1 * x, color='red')
        plt.xlabel('Depth')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.title(f'Linear Regression: y = {b0:.2f} + {b1:.2f}x')

        path = os.path.join(folder, 'linearRegression2.png')
        logger.info(f'Saving in: {os.path.abspath(path)}')
        plt.tight_layout()
        plt.savefig(path)

        plt.close()

        epochs = [epoch for epoch in range(ep)]
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))

        #quadratic error subplot of debugging
        axs[0].plot(epochs, mse_debug, color='green')
        axs[0].set_title('MSE dbgprah')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('MSE')
        axs[0].grid(True)

        #m vals subplot of debugging
        axs[1].plot(epochs, m_debug, color='orange')
        axs[1].set_title('m dbgprah')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('m')
        axs[1].grid(True)


        #c vals subplot of debugging
        axs[2].plot(epochs, c_debug, color='purple')
        axs[2].set_title('c dbgprah')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('c')
        axs[2].grid(True)

        debug_path = os.path.join(folder, 'debuggingParameters2.png')
        logger.info(f'Saving in: {os.path.abspath(debug_path)}')
        plt.tight_layout()
        plt.savefig(debug_path)


        plt.close()

    except:

        logger.critical("Error in the graphic plotting phase")

    logger.info(f"End time of the application: {time.time()}")
    return None

if __name__ == "__main__":
    main()
