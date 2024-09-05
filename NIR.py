from Algorithms.OptimizationParams.Height.MainAlghorithm import calculate_length_focal_distance
from Algorithms.OptimizationParams.Height.ModifiedAlgorithm import modified_calculate_length_focal_distance
from Algorithms.OptimizationParams.Height.OneDimension import one_dimensional_optimization
from Algorithms.OptimizationParams.Height.GlobalOptimize import differential_evolution_new
from Algorithms.OptimizationParams.Height.TestModule import gradient_descent
from Data.InputTestDate import DATA_SET_0
from Algorithms.OptimizationParams.Height.TestModuleNew import visualizetion_gradient_descent_3_param

import matplotlib.pyplot as plt
import numpy as np
import os
from sys import path


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

    visualizetion_gradient_descent_3_param()
    t = 3


    # начальные приближения высот

    initial_heights = np.array([1., 1., 1.])    

    initial_heights_h1_1_1 = np.array([1.1, 1., 1.])
    initial_heights_h2_1_1 = np.array([1.2, 1., 1.])
    initial_heights_h3_1_1 = np.array([1.3, 1., 1.])
    initial_heights_h4_1_1 = np.array([1.4, 1., 1.])
    initial_heights_h5_1_1 = np.array([2.1, 1., 1.])
    initial_heights_h6_1_1 = np.array([3.4, 1., 1.])
    initial_heights_h7_1_1 = np.array([4.3, 1., 1.])
    initial_heights_h8_1_1 = np.array([5.1, 1., 1.])
    initial_heights_h9_1_1 = np.array([6., 1., 1.])


    initial_heights_1_h1_1 = np.array([1., 1.1, 1.])
    initial_heights_1_h2_1 = np.array([1., 1.2, 1.])
    initial_heights_1_h3_1 = np.array([1., 1.3, 1.])
    initial_heights_1_h4_1 = np.array([1., 1.4, 1.])
    initial_heights_1_h5_1 = np.array([1., 2.1, 1.])
    initial_heights_1_h6_1 = np.array([1., 3.4, 1.])
    initial_heights_1_h7_1 = np.array([1., 4.3, 1.])
    initial_heights_1_h8_1 = np.array([1., 5.1, 1.])
    initial_heights_1_h9_1 = np.array([1., 6., 1.])

    initial_heights_1_1_h1 = np.array([1., 1., 1.1])
    initial_heights_1_1_h2 = np.array([1., 1., 1.2])
    initial_heights_1_1_h3 = np.array([1., 1., 1.3])
    initial_heights_1_1_h4 = np.array([1., 1., 1.4])
    initial_heights_1_1_h5 = np.array([1., 1., 2.1])
    initial_heights_1_1_h6 = np.array([1., 1., 3.4])
    initial_heights_1_1_h7 = np.array([1., 1., 4.3])
    initial_heights_1_1_h8 = np.array([1., 1., 5.1])
    initial_heights_1_1_h9 = np.array([1., 1, 6.])

    initial_heights_h1UP_h1UP_1 = np.array([1.1, 1.1, 1.])
    initial_heights_h2UP_h2UP_1 = np.array([2.1, 1.2, 1.])
    initial_heights_h3UP_h3UP_1 = np.array([1.2, 2.1, 1.])

    # результаты 

    # изменяем высоту 1й линзы
    res_h1_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h1_1_1, True)
    res_h2_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h2_1_1, True)
    res_h3_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h3_1_1, True)
    res_h4_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h4_1_1, True)
    res_h5_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h5_1_1, True)
    res_h6_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h6_1_1, True)
    res_h7_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h7_1_1, True)
    res_h8_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h8_1_1, True)
    res_h9_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h9_1_1, True)

    # изменяем высоту 2й линзы
    res_1_h1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h1_1, True)
    res_1_h2_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h2_1, True)
    res_1_h3_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h3_1, True)
    res_1_h4_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h4_1, True)
    res_1_h5_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h5_1, True)
    res_1_h6_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h6_1, True)
    res_1_h7_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h7_1, True)
    res_1_h8_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h8_1, True)
    res_1_h9_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h9_1, True)

    # изменяем высоту 3й линзы
    res_1_1_h1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h1, True)
    res_1_1_h2 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h2, True)
    res_1_1_h3 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h3, True)
    res_1_1_h4 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h4, True)
    res_1_1_h5 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h5, True)
    res_1_1_h6 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h6, True)
    res_1_1_h7 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h7, True)
    res_1_1_h8 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h8, True)
    res_1_1_h9 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h9, True)

    # изменяем высоты 1й и 2й линзы вверх
    res_h1UP_h1UP_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h1UP_h1UP_1, True)
    res_h2UP_h2UP_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h2UP_h2UP_1, True)
    res_h3UP_h3UP_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h3UP_h3UP_1, True)


    result = gradient_descent(initial_heights=initial_heights)
    print(f"Высоты равыне = {result[0]}",
          f"Время выполнения = {result[1]} с", sep='\n')

    
