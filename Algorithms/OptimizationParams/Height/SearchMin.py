import numpy as np
import matplotlib.pyplot as plt

from MainAlghorithm import calculate_length_focal_distance


def search_min(data_set_0: dict, 
            step: int | float = 0.1,     
            accuracy = 1e-6):
    
    '''
    Description:
    ------------
    Данный алгоритм перебирает диапазон высот с некоторым шагом,
    в результате для каждой линзы ищется минимум и ,после его нахождения,
    найденный минимум используется на следующем шаге для другой линзы.
    Пройдя всю систему первый раз мы проходимся по ней еще раз, чтобы 
    удостовериться, что мы нашли те значения высот, который дают нам 
    минимальный фокальный отрезок.

    Params:
    -------
    `data_set_0`: Данные о системе
    `step`: Шаг заданного диапазона высот
    `accuracy`: Точность для высот

    '''
    
    minH = 1e-6
    maxH = 20e-6
    num = (20 - 1) / step
    range_heights = np.linspace(minH, maxH, num=num, dtype=float) # Получили наш диапазон высот
    
    initial_heights = []
    [initial_heights.append(1.) for _ in range(data_set_0['count_linse'])]

    fig, axes = plt.subplots(2, 2)


    for numLinse in range(1, data_set_0['count_linse'] + 1):

        for height in range_heights:
            
            focal_dist = calculate_length_focal_distance(data_set_0, )



    