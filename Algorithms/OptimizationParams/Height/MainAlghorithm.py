from operator import matmul
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt

from Validate.InputTestDate import validateDataSet


# матрица преломления -> R = [1 0; -Ф 1]
# матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами
# формула для фокусн.раст f(k,lamda) = (m * lamda_0) / (k * lamda) * f_0
# итоговая матрица -> перемножение всех матриц с конца


#TODO: подумать насчёт правильности/логичности алгоритма
#FIXME: посмотреть значения данных, их размерность
#FIXME: пофиксить функцию, добавить новые параметры для дальнейших оптимизаций
def calculate_length_focal_distance(data_set_0: dict,
                                    heightOptimizeList: list[float] = None,
                                    show_plot = False) -> float:
    
    '''
    return: Функция возвращает длину фокального отрезка

    params: 
    heightOptimizeList: Параметр оптимизации; в данном случае список высот
    show_plot: Если true -> Выводит график зависимсоти длины волны от фокусного расстояния
    '''
    
    validateDataSet(data_set_0)

    lambdaMassive = np.linspace(data_set_0['lower_lambda'] * 1e9,
                                data_set_0['upper_lambda'] * 1e9,
                                1000) # без [нм]
    
    # Идёт заполнение массива гармоник,
    # если не был задан массив высот
    if heightOptimizeList is None:
        harmonics = []
        [harmonics.append(5+i) for i in range(1, data_set_0['count_linse'] + 1)]

    focus_labmda_dict = {}
    for lmbd in lambdaMassive:
        MatrixMultsList = []

        for currentNumLinse in range(1, data_set_0['count_linse'] + 1):

            removeDegreeToLambda_0 = (data_set_0['lambda_0'][currentNumLinse]) * 1e9 # [нм -> 10^(-9)]

            if heightOptimizeList is None:
                k = round((removeDegreeToLambda_0 / (lmbd)) * harmonics[currentNumLinse-1]) #FIXME: Правильно ли округляю число k?
                focus = ((harmonics[currentNumLinse-1] * removeDegreeToLambda_0) / (k * lmbd)) * data_set_0['focus_0'][currentNumLinse]
            else:
                harmonica = (heightOptimizeList[currentNumLinse-1] * (data_set_0['refractive_index'][currentNumLinse] - 1) / removeDegreeToLambda_0) * 1e3
                k = round((removeDegreeToLambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?
                focus = ((harmonica * removeDegreeToLambda_0) / (k * lmbd)) * data_set_0['focus_0'][currentNumLinse]

            Optic_Power = pow(focus , -1)            
            Refractive_Matrix = np.array(
                [
                    [1, 0],
                    [-Optic_Power, 1]
                ]
            )
            MatrixMultsList.append(Refractive_Matrix)
            
            if currentNumLinse != data_set_0['count_linse']:
                reduce_dist = data_set_0['distance']['{}-{}'.format(currentNumLinse, currentNumLinse + 1)] / data_set_0['refractive_index'][currentNumLinse]
                Transfer_Matrix = np.array(
                    [
                        [1, reduce_dist],
                        [0, 1]
                    ]
                )
                MatrixMultsList.append(Transfer_Matrix)
                
        MatrixMultsList.reverse()
        MatrixMultsList = np.array(MatrixMultsList)

        mult_res = reduce(matmul, MatrixMultsList)

        focus_labmda_dict[lmbd] = pow( -mult_res[1,0], -1)

    all_focus = list(focus_labmda_dict.values())
    length_focus_distance = max(all_focus) - min(all_focus)

    if show_plot: 
        all_lamdba = list(focus_labmda_dict.keys())

        x = np.array(all_lamdba)
        y = np.array(all_focus) * 100

        plt.figure(figsize=(6,5))
        plt.title('Зависимость длины волны от фокусного расстояния')
        plt.xlabel('Длина волны, нм')
        plt.ylabel('Фокусное расстояние, см')
        plt.grid(True)
        plt.scatter(x, y, linewidths=1, marker=3)
        plt.show()

        print('Длина фокального отрезка: {} см'.format(length_focus_distance * 100))
    
    return length_focus_distance