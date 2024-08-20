from Algorithms.OptimizationParams.Height.MainAlghorithm import calculate_length_focal_distance
from Algorithms.OptimizationParams.Height.ModifiedAlgorithm import modified_calculate_length_focal_distance
from Algorithms.OptimizationParams.Height.OneDimension import one_dimensional_optimization
from Algorithms.OptimizationParams.Height.GlobalOptimize import differential_evolution_new
from Algorithms.OptimizationParams.Height.TestModule import gradient_descent
from Data.InputTestDate import DATA_SET_0

import numpy as np
from os.path import join
from sys import path

if __name__ == '__main__':

    
    #with open(join(path[0], "Result", "results.txt"), 'a') as file:
    #    list_str = [
    #        "Привет, это 1 строка\n",
    #        "Привет, это 2 строка\n",
    #        "Привет, это 3 строка\n",
    #        "Конец строки\n"
    #        ] 
    #    file.writelines(list_str)
    #        file.write(str(elem) + '\n')
    #    file.write('___________________________________________________\n\n')

    #s = 1
    #print(*res, sep='\n')
 
        
    #heights, initial_heights, focal_dist = one_dimensional_optimization(DATA_SET_0)

    #print(f'Начальное приближение = {initial_heights} мкм',
    #      f'Список высот = {heights} мкм',
    #      f'Фокальный отрезок = {focal_dist * 100} см',
    #      sep='\n')
    
    #res = calculate_length_focal_distance(DATA_SET_0, None, True)
    #result = list(differential_evolution_new())
    #optimized_heights, min_focus_length = result[-1]
    #print(f"Optimized Heights: {optimized_heights}")
    #print(f"Minimum Focus Length: {min_focus_length}")


    try:
        a = 3
        zero = 0
        ab = a / zero
    except ZeroDivisionError:
        e = 3
        c = 4
        pass
    finally:
        ac = 1
        an = 2


    # начальные приближения высот

    initial_heights = np.array([1., 1., 1.])

    initial_heights_h1_1_1 = np.array([1.1, 1., 1.])
    initial_heights_h2_1_1 = np.array([1.2, 1., 1.])
    initial_heights_h3_1_1 = np.array([1.3, 1., 1.])
    initial_heights_h4_1_1 = np.array([1.4, 1., 1.])
    initial_heights_h5_1_1 = np.array([2.1, 1., 1.])
    initial_heights_h6_1_1 = np.array([3.4, 1., 1.])

    initial_heights_1_h1_1 = np.array([1., 1.1, 1.])
    initial_heights_1_h2_1 = np.array([1., 1.2, 1.])
    initial_heights_1_h3_1 = np.array([1., 1.3, 1.])
    initial_heights_1_h4_1 = np.array([1., 1.4, 1.])
    initial_heights_1_h5_1 = np.array([1., 2.1, 1.])
    initial_heights_1_h6_1 = np.array([1., 3.4, 1.])
    initial_heights_1_h7_1 = np.array([1., 4.2, 1.])

    initial_heights_1_1_h1 = np.array([1., 1., 1.1])
    initial_heights_1_1_h2 = np.array([1., 1., 1.2])
    initial_heights_1_1_h3 = np.array([1., 1., 1.3])
    initial_heights_1_1_h4 = np.array([1., 1., 1.4])
    initial_heights_1_1_h5 = np.array([1., 1., 2.1])
    initial_heights_1_1_h6 = np.array([1., 1., 3.4])
    initial_heights_1_1_h7 = np.array([1., 1., 4.2])

    initial_heights_h1UP_h1UP_1 = np.array([1.1, 1.1, 1.])
    initial_heights_h2UP_h2UP_1 = np.array([1.2, 1.2, 1.])
    initial_heights_h3UP_h3UP_1 = np.array([1.3, 1.3, 1.])

    # результаты 

    # изменяем высоту 1й линзы
    res_h1_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h1_1_1, True)
    res_h2_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h2_1_1, True)
    res_h3_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h3_1_1, True)
    res_h4_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h4_1_1, True)
    res_h5_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h5_1_1, True)
    res_h6_1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h6_1_1, True)

    # изменяем высоту 2й линзы
    res_1_h1_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h1_1)
    res_1_h2_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h2_1)
    res_1_h3_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h3_1)
    res_1_h4_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h4_1)
    res_1_h5_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h5_1)
    res_1_h6_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h6_1)
    res_1_h7_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_h7_1)

    # изменяем высоту 3й линзы
    res_1_1_h1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h1)
    res_1_1_h2 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h2)
    res_1_1_h3 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h3)
    res_1_1_h4 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h4)
    res_1_1_h5 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h5)
    res_1_1_h6 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h6)
    res_1_1_h7 = calculate_length_focal_distance(DATA_SET_0, initial_heights_1_1_h7)


    # изменяем высоты 1й и 2й линзы вверх
    res_h1UP_h1UP_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h1UP_h1UP_1)
    res_h2UP_h2UP_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h2UP_h2UP_1)
    res_h3UP_h3UP_1 = calculate_length_focal_distance(DATA_SET_0, initial_heights_h3UP_h3UP_1)


    result = gradient_descent(initial_heights=initial_heights)
    print(f"Высоты равыне = {result[0]}",
          f"Время выполнения = {result[1]} с", sep='\n')

    
