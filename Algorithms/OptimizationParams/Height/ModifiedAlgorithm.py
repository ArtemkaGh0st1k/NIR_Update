from functools import reduce
from operator import matmul

import numpy as np
from numpy import ceil

from Validate.InputTestDate import validateDataSet


def modified_calculate_length_focal_distance(heightOptimize,
                                            heightsConstList: list,
                                            currentLinse: int,
                                            data_set_0: dict) -> float:
    
    '''
    Description: Модифицированная функция для подачи 
    её в аругемент функции оптимизации из модуля SciPy

    return: Функция возвращает длину фокального отрезка

    params: 
    `heightOptimize`: Высота, которая подбирается
    `heightsConstList`: Список фиксированных высот
    `currentLinse`: Текущая линза, для которой идёт оптимизация
    '''
    
    validateDataSet(data_set_0)

    lambdaMassive = np.linspace(data_set_0['lower_lambda'] * 1e9,
                                data_set_0['upper_lambda'] * 1e9,
                                100) # без [нм]

    focus_labmda_dict = {}
    for lmbd in lambdaMassive:
        MatrixMultsList = []

        for currentNumLinse in range(1, data_set_0['count_linse'] + 1):

            removeDegreeToLambda_0 = (data_set_0['lambda_0'][currentNumLinse] * 1e9) # [нм -> 10^(-9)]

            if currentNumLinse == currentLinse:
                harmonica = ((heightOptimize * (data_set_0['refractive_index'][currentNumLinse] - 1)) / removeDegreeToLambda_0) * 1e3
            else:
                harmonica = ((heightsConstList[currentNumLinse-1] * (data_set_0['refractive_index'][currentNumLinse] - 1)) / removeDegreeToLambda_0) * 1e3
              

            k = ceil((removeDegreeToLambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?

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
    return length_focus_distance