from operator import matmul
from functools import reduce
import time 
import warnings
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from Data.InputTestDate import validate_data_set, DATA_SET_0
from MainAlghorithm import calculate_length_focal_distance


LERNING_RATE = 0.01
MAX_ITER = 1000
STEP_H = 0.1 
DENOMINATOR = 1e-1

# m -> фиксированное целое число
# lambda_0 -> идёт подбор этого подбора в зависимости от поданной высоты
# f = (m * lambda_0) / (k * lambda) * f_0 -> формула верная, можно пользоваться
# матрица преломления -> R = [1 0; -D 1]
# матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами
def loss_function(data_set_0: dict,
                  heights_optimize : list[float] | np.ndarray) -> float:
    
    '''
    heights_optimize приходит не в [мкм] !!!
    '''
    #if not isinstance(heights_optimize, np.ndarray): heights_optimize = np.array(height_optimize) * 1e-6
    #else: heights_optimize *= 1e-6

    validate_data_set(data_set_0)

    lambda_massive = np.linspace(data_set_0['lower_lambda'],
                                 data_set_0['upper_lambda'],
                                 1000) 
    
    focus_lambda_dict = {}
    for lmbd in lambda_massive:
        Matrix_Mults_List = []

        for num_linse in range(1, data_set_0['count_linse'] + 1):
            
            harmonica = data_set_0['harmonica'][num_linse]
            focus_0 = data_set_0['focus_0'][num_linse]
            height_optimize = heights_optimize[num_linse-1] * 1e-6
            refractive_index = data_set_0['refractive_index'][num_linse]

            lambda_0 = height_optimize * (refractive_index - 1) / harmonica
            if (lambda_0 > data_set_0["upper_lambda"] or lambda_0 < data_set_0['lower_lambda']): 
                continue
            k = round((lambda_0 / (lmbd)) * harmonica)
            if k == 0: k = 1
            focus = ((harmonica * lambda_0) / (k * lmbd)) * focus_0
            optic_power = 1 / focus 

            Refractive_Matrix = np.array(
                [
                    [1, 0],
                    [-optic_power, 1]
                ]
            )
            Matrix_Mults_List.append(Refractive_Matrix)
            
            if num_linse != data_set_0['count_linse']:

                refractive_area = data_set_0['refractive_area']['{}-{}'.format(num_linse, num_linse + 1)]
                dist = data_set_0['distance']['{}-{}'.format(num_linse, num_linse + 1)]

                reduce_dist = dist / refractive_area

                Transfer_Matrix = np.array(
                    [
                        [1, reduce_dist],
                        [0, 1]
                    ]
                )
                Matrix_Mults_List.append(Transfer_Matrix)
                
        if not Matrix_Mults_List: return        
        Matrix_Mults_List.reverse()
        Matrix_Mults_List = np.array(Matrix_Mults_List)

        mult_res = reduce(matmul, Matrix_Mults_List)

        focus_lambda_dict[lmbd] = - 1 / mult_res[1, 0]
    
    all_focus = list(focus_lambda_dict.values())
    length_focus_distance = max(all_focus) - min(all_focus)
    
    return length_focus_distance


def numerical_gradient(heights : np.ndarray,
                       loss_func = loss_function, 
                       step_h = STEP_H, denominator = DENOMINATOR) -> np.ndarray[float]:
    
    grad = np.zeros_like(heights)
    for i in range(len(heights)):
        # Сохранение текущего значения параметра
        original_value = heights[i]

        # Вычесление f(h + step_h)
        heights[i] = original_value + step_h
        loss_func_plus_h = loss_func(DATA_SET_0, heights)

        # Вычесление f(h)
        heights[i] = original_value
        loss_function_original = loss_func(DATA_SET_0, heights)

        # Градиент по i-координате
        grad[i] = (loss_func_plus_h - loss_function_original) / denominator

        # Востановление параметра
        heights[i] = original_value

    return grad


def gradient_descent_simple_3(initial_params : list[float] | np.ndarray,
                           max_iter = MAX_ITER, learning_rate = LERNING_RATE, momentum = 0.8, acc = 1e-5,
                           loss_func = loss_function,
                           grad_func = numerical_gradient):
    
    #if not isinstance(initial_params, np.ndarray): initial_params = np.array(initial_params) * 1e-6
    #else: initial_params *= 1e-6

    h1_list = []
    h2_list = []
    #h3_list = []

    h1 = None
    h2 = None
    #h3 = None

    h1_list.append(initial_params[0])
    h2_list.append(initial_params[1])
    #h3_list.append(initial_params[2])

    h1 = initial_params[0]
    h2 = initial_params[1]
    #h3 = initial_params[2]

    h_general_list = np.array([h1, h2])

    start_time = time.time()
    for i in range(max_iter):

        grad = grad_func(h_general_list, loss_func)

        # Обновляем параметры
        h1 -= learning_rate * grad[0]
        h2 -= learning_rate * grad[1]
        #h3 -= learning_rate * grad[2]

        # Нужны списки для последующей визуализации град.спуска для каждой высоты
        h1_list.append(h1)
        h2_list.append(h2)
        #h3_list.append(h3)

        h_general_list = np.array([h1, h2])

    end_time = time.time()
    diff_time = end_time - start_time
    print("Время работы град.спуска = {} м {} с".format(diff_time // 60, diff_time % 60))
    return (h1, h2), (h1_list, h2_list)


def visualizetion_gradient_descent_3_param(initial_heights : list[float] | np.ndarray = None):
    if initial_heights is None: 
        initial_heights = [2., 2.]
    
    height_optimize_list, h = gradient_descent_simple_3(initial_heights)
    focus_dist = calculate_length_focal_distance(DATA_SET_0, height_optimize_list)
    optimize_param_name = "h_" + "_".join(str(initial_heights).
                                          replace("[", "").
                                          replace("]", "").
                                          replace(",", "").
                                          split() + \
                                          ['lr={}_step_h={}_max_iter={}_detorminator={}'.format(LERNING_RATE, STEP_H, MAX_ITER, DENOMINATOR)])

    fig, axs = plt.subplots(nrows=1, ncols=len(initial_heights), figsize=(15,7))
    plt.subplots_adjust(wspace=0.3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), title=f"Фокальный отрезок = {focus_dist * 100} см")
    
    for i, ax in enumerate(axs, start=1):
        ax.set_title(f"График градиентого спуска для {i}-ый высоты")
        ax.set_xlabel("Иттерация")
        ax.set_ylabel(f"Высота {i}-ой линзы, мкм")
        ax.plot(h[i-1], label='lr={}\nstep_h={}\nmax_iter={}\ndetorminator={}'.format(LERNING_RATE, STEP_H, MAX_ITER, DENOMINATOR))
        ax.grid(True)
        ax.legend(frameon=False, fontsize=7)

    fname = os.path.join("NIR_Update", "Result", "GradientDescent", "NewAttempts", "2_Linse", f"{optimize_param_name}.png")
    fig.savefig(fname=fname)
    plt.close()


