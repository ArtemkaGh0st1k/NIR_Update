"""
-----------------------------------------------------
Данный модуль хранит в себе основной алгоритм расчёта 
длины фокального отрезка для системы гаромнических линз
---------------------------------------------------------
"""


from operator import matmul
from functools import reduce
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

from Data.InputTestDate import validate_data_set

#warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='warn')

# матрица преломления -> R = [1 0; -Ф 1]
# матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами
# формула для фокусн.раст f(k,lamda) = (m * lamda_0) / (k * lamda) * f_0
# итоговая матрица -> перемножение всех матриц с конца


#TODO: подумать насчёт правильности/логичности алгоритма
#FIXME: посмотреть значения данных, их размерность
#FIXME: пофиксить функцию, добавить новые параметры для дальнейших оптимизаций
def calculate_length_focal_distance(data_set_0: dict,
                                    height_optimize_list: list[float] = None,
                                    show_plot = False) -> float:
    
    '''
    Description
    -----------
    Основная функция для графической и расчётной интерпретации
    зависимости длины фок.расстояния от длины волны
    
    Return
    ------
    Функция возвращает длину фокального отрезка

    Params
    ------
    `data_set_0`: Исходные данные
    `heightOptimizeList`: Параметр оптимизации; в данном случае список высот
    `show_plot`: Если true -> Выводит график зависимсоти длины волны от фокусного расстояния
    '''
    
    validate_data_set(data_set_0)

    lambda_massive = np.linspace(data_set_0['lower_lambda'] * 1e9,
                                data_set_0['upper_lambda'] * 1e9,
                                1000) # без [нм]
    
    # Идёт заполнение массива гармоник,
    # если не был задан массив высот
    if height_optimize_list is None:
        harmonics = []
        [harmonics.append(5+i) for i in range(1, data_set_0['count_linse'] + 1)]

    focus_lambda_dict = {}
    for lmbd in lambda_massive:
        Matrix_Mults_List = []
        for num_linse in range(1, data_set_0['count_linse'] + 1):

            remove_degree_to_lambda_0 = (data_set_0['lambda_0'][num_linse]) * 1e9 # [нм -> 10^(-9)]
            focus_0 = data_set_0['focus_0'][num_linse]

            if height_optimize_list is None:
                harmonica = harmonics[num_linse - 1]

                k = round((remove_degree_to_lambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?
                focus = ((harmonica * remove_degree_to_lambda_0) / (k * lmbd)) * focus_0
            else:
                height_optimize = height_optimize_list[num_linse-1]
                refractive_index = data_set_0['refractive_index'][num_linse]

                harmonica = (height_optimize * (refractive_index - 1) / remove_degree_to_lambda_0) * 1e3
                k = round((remove_degree_to_lambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?
                if k == 0: k = 1
                focus = ((harmonica * remove_degree_to_lambda_0) / (k * lmbd)) * focus_0

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
                        [-optic_power, 1]
                    ]
                )
                Matrix_Mults_List.append(Transfer_Matrix)
                
        Matrix_Mults_List.reverse()
        Matrix_Mults_List = np.array(Matrix_Mults_List)

        mult_res = reduce(matmul, Matrix_Mults_List)

        focus_lambda_dict[lmbd] = - 1 / mult_res[1, 0]

    all_focus = list(focus_lambda_dict.values())
    max_val = max(all_focus)
    min_val = min(all_focus)
    length_focus_distance = max(all_focus) - min(all_focus)

    if show_plot: 
        optimize_param_name = "h_" + "_".join(str(height_optimize_list).replace("[", "").replace("]", "").split())
        save_path = os.path.join("NIR_Update", "Result", f"{optimize_param_name}.png")
        all_lamdba = list(focus_lambda_dict.keys())

        x = np.array(all_lamdba)
        y = np.array(all_focus) * 100

        plt.figure(figsize=(6,5))
        plt.title('Зависимость длины волны от фокусного расстояния')
        plt.xlabel('Длина волны, нм')
        plt.ylabel('Фокусное расстояние, см')
        plt.grid(True)
        plt.plot(x, y)
        plt.savefig(save_path)
        #plt.show() # не работает!

        print('Длина фокального отрезка: {} см'.format(length_focus_distance * 100))
    
    return length_focus_distance