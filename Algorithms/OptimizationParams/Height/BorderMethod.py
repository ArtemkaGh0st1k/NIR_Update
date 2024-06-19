import numpy as np

from MainAlghorithm import calculate_length_focal_distance


#TODO: реализовать покоординатный метод поиска
#TODO: подумать над самой идеей метода
def border_method(data_set_0: dict, eps = 1e-4, bounds: tuple[int, int] = None, step: int = None):

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
            