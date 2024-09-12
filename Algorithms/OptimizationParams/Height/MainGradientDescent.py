"""
--------------------------------------------------------------------
Данный модуль хранит в себе главный алгоритм оптимизации, 
а имеено градиентный спуск для параметра оптимизации - `высоты`
-----------------------------------------------------------------
"""


from operator import matmul
from functools import reduce
import time 
import os
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt

from Data.InputTestDate import validate_data_set, DATA_SET_0
from MainAlghorithm import calculate_length_focal_distance
from Utils.Converter import custom_converter_tolist

LERNING_RATE = 0.01
MAX_ITER = 1000
STEP_H = 0.1 
DENOMINATOR = 1e-2

# m -> фиксированное целое число
# lambda_0 -> идёт подбор этого подбора в зависимости от поданной высоты
# f = (m * lambda_0) / (k * lambda) * f_0 -> формула верная, можно пользоваться
# матрица преломления -> R = [1 0; -D 1]
# матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами
def loss_function(data_set_0: dict,
                  heights_optimize : list[float] | np.ndarray):
    
    '''
    heights_optimize приходит не в [мкм] !!!
    '''

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
                      
        Matrix_Mults_List.reverse()
        Matrix_Mults_List = np.array(Matrix_Mults_List)

        mult_res = reduce(matmul, Matrix_Mults_List)
        focus_lambda_dict[lmbd] = - 1 / mult_res[1, 0]
    
    all_focus = list(focus_lambda_dict.values())
    length_focus_distance = max(all_focus) - min(all_focus)
    
    return length_focus_distance
    

def collect_min_max_h_range(data_set_0: dict) -> dict[int, tuple[float, float]]:
    '''
        Возвращяет минимальные и максимальные высоты каждой линзы,
        пригодные для алгоритма оптимизации в [мкм]  
    '''

    lower_lambda = data_set_0['lower_lambda']
    upper_lambda = data_set_0['upper_lambda']

    collect_data = {}
    for num_linse in range(1, data_set_0['count_linse'] + 1):   

        harmonica = data_set_0['harmonica'][num_linse]
        refractive_index = data_set_0['refractive_index'][num_linse]

        min_h = harmonica / (refractive_index - 1) * lower_lambda * 1e6
        max_h = harmonica / (refractive_index - 1) * upper_lambda * 1e6

        collect_data[num_linse] = (min_h, max_h)

    return collect_data    


def numerical_gradient(heights : np.ndarray | list,
                       loss_func = loss_function, 
                       step_h = STEP_H, denominator = DENOMINATOR) -> np.ndarray[float]:
    '''
        Description:
        ---------------
        Численный градиент, т.к целевая функция имеет неявный вид

        Return:
        ----------
        Возвращяет численный градиент

        Params:
        ---------

        `heights`: Параметр оптимизации (высоты)
        `loss_func`: Функция потерь (или целевая функция)
        `step_h` : Шаг для параметра оптимизации (высоты)
        `denominator` : Знаменатель, для выбора масштаба градиента

    '''
    
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


def gradient_descent(initial_params : list[float] | np.ndarray,
                     max_iter = MAX_ITER, learning_rate = LERNING_RATE,
                     loss_func = loss_function,
                     grad_func = numerical_gradient):
    
    '''
        
    '''
    
    # Проверка на выход из допустимого диапазона высоты
    collect_data = collect_min_max_h_range(DATA_SET_0)
    for i, param in enumerate(initial_params, start=1):
        min_h = collect_data[i][0]
        max_h = collect_data[i][1]

        if (param < min_h):
            initial_params[i-1] = min_h
        elif (param > max_h):
            initial_params[i-1] = max_h

    h = None
    h_list = []

    h = np.array(initial_params)
    h_list.append(h.copy())

    start_time = time.time()
    for _ in range(max_iter):

        grad = grad_func(h, loss_func)

        h -= learning_rate * grad
        h_copy = h.copy()
        h_list.append(h_copy)
        
    end_time = time.time()
    diff_time = end_time - start_time
    print(Fore.BLUE, "Время работы град.спуска = {} м {} с".format(diff_time // 60, diff_time % 60))
    return h, h_list


def visualization_gradient_descent(initial_heights : list[float] | np.ndarray = None):
    if initial_heights is None: 
        initial_heights = []
        [initial_heights.append(1.) for _ in range(DATA_SET_0['count_linse'])]
    
    height_optimize_list, h = gradient_descent(initial_heights)
    h_convert = custom_converter_tolist(size=DATA_SET_0['count_linse'], current_list=h)
    focus_dist = calculate_length_focal_distance(DATA_SET_0, height_optimize_list)
    collect_min_max_h = collect_min_max_h_range(DATA_SET_0)

    optimize_param_name = "h_" + "_".join(str(initial_heights).
                                          replace("[", "").
                                          replace("]", "").
                                          replace(",", "").
                                          split() + \
                                          ['lr={}_step_h={}_max_iter={}_detorminator={}'.format(LERNING_RATE, STEP_H, MAX_ITER, DENOMINATOR)])
    
    count_linse = DATA_SET_0['count_linse']
    figsize = (15, 7)
    if (count_linse == 4): figsize = (20, 7)
    elif (count_linse == 5): figsize = (25, 7)

    fig, axs = plt.subplots(nrows=1, ncols=len(initial_heights), figsize=figsize)
    plt.subplots_adjust(wspace=0.3)
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), title=f"Фокальный отрезок = {focus_dist * 100} см")
    
    iter_massive = list(range(1, MAX_ITER + 1))
    for i, ax in enumerate(axs, start=1):
        min_h_masssive = [ collect_min_max_h[i][0] ] * MAX_ITER
        max_h_massive = [ collect_min_max_h[i][1] ] * MAX_ITER

        harmonica = DATA_SET_0['harmonica'][i]
        ref_index = DATA_SET_0['refractive_index'][i]
        focus_0 = DATA_SET_0['focus_0'][i] 
        label = 'lr={}\nstep_h={}\nmax_iter={}\ndetorminator={}\n'.format(
                LERNING_RATE,
                STEP_H,
                MAX_ITER,
                DENOMINATOR
                ) + \
                'harmonica={}\nref_index={}\nfocus_0={} см'.format(
                harmonica,
                ref_index,
                focus_0 * 100
                )

        ax.set_title(f"График град.спуска для {i}-ый высоты")
        ax.set_xlabel("Итерация")
        ax.set_ylabel(f"Высота {i}-ой линзы, мкм")
        ax.plot(h_convert[i-1], label=label)
        ax.plot(iter_massive, min_h_masssive, color='green')
        ax.plot(iter_massive, max_h_massive, color='red')
        ax.grid(True)
        ax.legend(frameon=False, fontsize=7)

    
    fname = os.path.join("NIR_Update", "Result", "GradientDescent", "NewAttempts", f"{count_linse}_Linse", f"{optimize_param_name}.png")
    fig.savefig(fname=fname)
    plt.close()


