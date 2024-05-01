from math import ceil
from functools import reduce
from operator import matmul

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar

from InputTestDate import data_set_0, validateDataSet

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


#TODO: реализовать покоординатный метод поиска
#TODO: подумать над самой идеей метода
def test_method(data_set_0: dict, eps = 1e-4, bounds: tuple[int, int] = None, step: int = None):

    '''
    Алгоритм перебирает все значение и находит минимум
    '''

    if bounds is None:
        minHeight = 1
        maxHeight = 10
        step_ = 100 if step is None else step
        heightsMassive = np.linspace(minHeight, maxHeight, step_)
    else:
        minHeight = bounds[0]
        maxHeight = bounds[1]
        step_ = 100 if step is None else step
        heightsMassive = np.linspace(minHeight, maxHeight, step_)


    HeightOptimizeMassive = [] # добавление без [мкм] 
    [HeightOptimizeMassive.append(1.) for _ in range(1, data_set_0['count_linse'] + 1)]

    # словарь рассчитан макс для 500 элементов?
    globalFocalDist = {}
    #globalHeightValues = {}
    globalCountIter = 0

    while(globalCountIter < 100):

        if globalCountIter > 1:
            if (globalFocalDist['{}_{}'.format('Step', str(globalCountIter))] - globalFocalDist['{}_{}'.format('Step', str(globalCountIter-1))]) < eps:
                return [ f'Дипазон оптимизации высот: от {minHeight} мкм до {maxHeight} мкм',
                        f'Шаг оптимизации высот: 1 / {len(heightsMassive)} мкм',
                        'Фок.отрезок : {} см'.format((globalFocalDist[f'Step_{globalCountIter}'] * 100).__round__(4)), 
                        f'Число иттераций(по системе): {globalCountIter}',
                        f'Высоты равны = {HeightOptimizeMassive} мкм'
                        ]
        

        localFocalDist = {}
        localCountIter = 0  
        while(True):

            if localCountIter > 1:
                if (localFocalDist['{}_{}'.format('Step', str(localCountIter))] - localFocalDist['{}_{}'.format('Step', str(localCountIter-1))]) < eps:
                    break       
                
            for count_linse in range(1, data_set_0['count_linse'] + 1):
                
                # макс. размер словарей 500 объектов???
                HelperFocalDist_Dict = {}
                HelperHeight_Dict = {} 
                for index, height in enumerate(heightsMassive):

                    HeightOptimizeMassive[count_linse-1] = height
                    focal_dist = calculate_length_focal_distance(HeightOptimizeMassive, data_set_0)

                    # трюк для того, чтобы достать высоту, при которой был мин.фок.отрезок
                    HelperFocalDist_Dict[index] = focal_dist
                    HelperHeight_Dict[str(focal_dist)] = height
                
                min_focal_dist = min(HelperFocalDist_Dict.values())
                HeightOptimizeMassive[count_linse-1] = HelperHeight_Dict[str(min_focal_dist)]
                #globalHeightValues[count_linse] = HelperHeight_Dict[str(min_focal_dist)]

                HelperHeight_Dict.clear()
                HelperFocalDist_Dict.clear()
            
            localCountIter += 1
            #for count_linse in range(1, data_set_0['count_linse'] + 1):
                #HeightOptimizeMassive[count_linse-1] = globalHeightValues[count_linse]  
            localFocalDist[f'Step_{localCountIter}'] = calculate_length_focal_distance(HeightOptimizeMassive, data_set_0)
        
        globalCountIter +=1
        globalFocalDist[f'Step_{globalCountIter}'] = calculate_length_focal_distance(HeightOptimizeMassive, data_set_0)
            

def OneDimensionalOptimization(data_set_0: dict, eps = 1e-5
                               ) -> tuple[list[float], list[float], float]:

    '''
    Description: Одномерная оптимизация

    Return: Возвращает кортеж ->
            1) Начальное приближение\n
            2) Список высот в [мкм]\n
            3) Фокальный отрезок в [м]

    Warning: Данный метод очень сильно зависит
            от начального приближения

    `data_set_0`: начальный датасет
    `eps`: точность
    '''

    initial_height = [1., 1., 1.]
    retuned_initial_heiht = initial_height
    options = {'maxiter': 100, 'xatol': eps}

    globalIter = 0
    globalFocalDit_Dict = {}

    while(globalIter < 100):

        if globalIter > 1:
            if globalFocalDit_Dict['Iter_{}'.format(str(globalIter))] - globalFocalDit_Dict['Iter_{}'.format(str(globalIter-1))] < eps:
                break

        localIter = 0
        localFocalDit_Dict = {}
        while(True):
            if localIter > 1:
                if localFocalDit_Dict['Iter_{}'.format(str(localIter))] - localFocalDit_Dict['Iter_{}'.format(str(localIter-1))] < eps:
                    break
            # для всех лизн, кроме i высоты фиксированы
            # это нужно для поиска минимума для каждой линзы т.е одномерная оптимизация
            for numLinse in range(1, data_set_0['count_linse'] + 1):

                result_scipy = minimize_scalar(modified_calculate_length_focal_distance,
                                                method='bounded', bounds=(1, 30),
                                                args=(initial_height,numLinse, data_set_0), options=options)
                if result_scipy.success:
                    height = result_scipy.x
                    initial_height[numLinse-1] = height
                else:
                    raise TypeError('Внимательно подавайте аргументы в функцию!')
            
            localIter += 1
            localFocalDit_Dict[f'Iter_{localIter}'] =  calculate_length_focal_distance(data_set_0, initial_height)

        localFocalDit_Dict.clear()
        globalIter +=1
        globalFocalDit_Dict[f'Iter_{globalIter}'] = calculate_length_focal_distance(data_set_0, initial_height)  

    return retuned_initial_heiht, initial_height, globalFocalDit_Dict[f'Iter_{globalIter}']
      