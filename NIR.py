from Algorithms.OptimizationParams.Height.MainAlghorithm import calculate_length_focal_distance
from Data.InputTestDate import DATA_SET_0
from Algorithms.OptimizationParams.Height.MainGradientDescent import visualization_gradient_descent

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    
    '''
    lr = 0.01
    step_h = 0.01
    max_iter = 1000
    detorminator = 1e-4

    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(25,8))
    plt.subplots_adjust(wspace=0.3)
    x1 = np.linspace(0, 2, 100)
    x2 = np.linspace(2, 4, 100)
    x3 = np.linspace(6, 8, 100)
    y1 = np.sin(x1)
    y2 = np.cos(x2)
    y3 = np.tan(x3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), title='Легенда')
    for i, ax in enumerate(axs, start=1):
        ax.set_title(f"График градиентого спуска для {i}-ый высоты")
        ax.set_xlabel("Иттерация")
        ax.set_ylabel(f"Высота {i}-ой линзы, мкм")
        ax.plot(x1, y1, label='lr={}\nstep_h={}\nmax_iter={}\ndetorminator={}'.format(lr, step_h, max_iter, detorminator))
        ax.grid(True)
        ax.legend(frameon=False, fontsize=7)
    fig.savefig("sample.png")
    '''

    visualization_gradient_descent(initial_heights=[7., 7., 7.])
    t = 3
    
