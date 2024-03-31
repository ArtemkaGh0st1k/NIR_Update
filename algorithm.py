import matplotlib.pyplot as plt
import numpy as np
from math import ceil
from functools import reduce
from operator import matmul

from InputTestDate import data_set_0, validateDataSet

# матрица преломления -> R = [1 0; -Ф 1]
# матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами
# формула для фокусн.раст f(k,lamda) = (m * lamda_0) / (k * lamda) * f_0
# итоговая матрица -> перемножение всех матриц с конца


#TODO: подумать насчёт правильности/логичности алгоритма
#FIXME: посмотреть значения данных, их размерность
#FIXME: пофиксить функцию, добавить новые параметры для дальнейших оптимизаций
def calculate_length_focal_distance(heightOptimizeList: list[float],
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
                                100) # без [нм]

    focus_labmda_dict = {}
    focus_labmda_reduce_test= []
    for lmbd in lambdaMassive:
        MatrixMultsList = []

        for currentNumLinse in range(1, data_set_0['count_linse'] + 1):

            #removeDegreeToHeight = float(heightOptimizeList[currentNumLinse-1] * 1e6) # [мкм -> 10^(-6)]
            #removeDegreeToLambda = lmbd * 1e9 # [нм -> 10^(-9)]
            removeDegreeToLambda_0 = (data_set_0['lambda_0'][currentNumLinse] * 1e9) # [нм -> 10^(-9)]
             
            harmonica = ((heightOptimizeList[currentNumLinse-1] * (data_set_0['refractive_index'][currentNumLinse] - 1)) / removeDegreeToLambda_0) * 1e3

            k = ceil((removeDegreeToLambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?

            focus = ((harmonica * removeDegreeToLambda_0) / (k * lmbd)) * data_set_0['focus_0'][currentNumLinse]
            Optic_Power = pow(focus , -1)

            Refractive_Matrix = np.array( [ [1, 0], 
                                            [-Optic_Power, 1] ])
            MatrixMultsList.append(Refractive_Matrix)
            
            if currentNumLinse != data_set_0['count_linse']:
                reduce_dist = data_set_0['distance']['{}-{}'.format(currentNumLinse, currentNumLinse + 1)] / data_set_0['refractive_index'][currentNumLinse]
                Transfer_Matrix   = np.array( [ [1, reduce_dist ],
                                                [0,1] ])
                MatrixMultsList.append(Transfer_Matrix)
                
        MatrixMultsList.reverse()
        MatrixMultsList = np.array(MatrixMultsList)

        mult_res_reduce = reduce(matmul, MatrixMultsList)

        mult_res = MatrixMultsList[0] @ MatrixMultsList[1]
        for j in range(2, len(MatrixMultsList)):
            mult_res = mult_res @ MatrixMultsList[j]
        #mult_res = mult_res @ MatrixMultsList[-1]

        focus_labmda_dict[lmbd] = pow( -mult_res[1,0], -1)
        focus_labmda_reduce_test.append(pow(-mult_res_reduce[1, 0], -1))

    all_focus = list(focus_labmda_dict.values())
    all_focus_reduce = list(focus_labmda_reduce_test)
    length_focus_distance_reduce = max(all_focus_reduce) - min(all_focus_reduce)
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
        print('Длина фокального отрезка reduce: {} см'.format(length_focus_distance_reduce * 100))
    
    return length_focus_distance, length_focus_distance_reduce

#TODO: реализовать покоординатный метод поиска
#TODO: подумать над самой идеей метода
def test_method(eps = 1e-4):

    '''
    Алгоритм перебирает все значение и находит минимум
    '''

    minHeight = 1
    maxHeight = 10
    heightsMassive = np.linspace(minHeight, maxHeight, 100)

    HeightOptimizeMassive = [] # добавление без [мкм] 
    [HeightOptimizeMassive.append(1.) for _ in range(1, data_set_0['count_linse'] + 1)]

    # словарь рассчитан макс для 500 элементов
    globalFocalDist = {}
    globalHeightValues = {}
    globalCountIter = 0

    while(globalCountIter < 100):

        if globalCountIter > 1:
           if (globalFocalDist['{}_{}'.format('Step', str(globalCountIter-1))] - globalFocalDist['{}_{}'.format('Step', str(globalCountIter))]) < eps:
               return [ f'Дипазон оптимизации высот: от {minHeight} мкм до {maxHeight} мкм',
                        f'Шаг оптимизации высот: 1 / {len(heightsMassive)} мкм',
                        'Фок.отрезок : {} см'.format((globalFocalDist[f'Step_{globalCountIter}'] * 100).__round__(4)), 
                        f'Число иттераций(по системе): {globalCountIter}',
                        [f'Линза {numLinse} : {height.__round__(4)} мкм' for numLinse, height in globalHeightValues.items()]
                      ]
           
        
        for count_linse in range(1, data_set_0['count_linse'] + 1):
            
            '''[x_0]
            # начальное значение для сравнения
            #tempH = heightsMassive[0]
            #HeightOptimizeMassive[count_linse-1] = heightsMassive[0]
            #focal_dist_next = calculate_length_focal_distance(HeightOptimizeMassive)
            '''

            # макс. размер словарей 500 объектов???
            HelperFocalDist_Dict = {}
            HelperHeight_Dict = {} 
            for index, height in enumerate(heightsMassive):

                '''[prev&next]
                #focal_dist_prev = focal_dist_next
                #focal_dist = calculate_length_focal_distance(HeightOptimizeMassive)

                #diff = focal_dist_next - focal_dist_prev
                #if fabs(focal_dist_next - focal_dist_prev) < eps:
                '''
                
                HeightOptimizeMassive[count_linse-1] = height
                focal_dist = calculate_length_focal_distance(HeightOptimizeMassive)

                # трюк для того, чтобы достать высоту, при которой был мин.фок.отрезок
                HelperFocalDist_Dict[index] = focal_dist
                HelperHeight_Dict[str(focal_dist)] = height
            
            min_focal_dist = min(HelperFocalDist_Dict.values())
            HeightOptimizeMassive[count_linse-1] = HelperHeight_Dict[str(min_focal_dist)]
            globalHeightValues[count_linse] = HelperHeight_Dict[str(min_focal_dist)]

            HelperHeight_Dict.clear()
            HelperFocalDist_Dict.clear()
        
        globalCountIter += 1
        #for count_linse in range(1, data_set_0['count_linse'] + 1):
            #HeightOptimizeMassive[count_linse-1] = globalHeightValues[count_linse]  
        globalFocalDist[f'Step_{globalCountIter}'] = calculate_length_focal_distance(HeightOptimizeMassive)
            

        
