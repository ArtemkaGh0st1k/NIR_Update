# Функции для глобальной оптимизации
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping
from scipy.optimize import shgo
from scipy.optimize import dual_annealing
from scipy.optimize import OptimizeResult

import numpy as np

from operator import matmul
from functools import reduce

from Data.InputTestDate import validate_data_set, DATA_SET_0


def test_function(heights_optimize_list: list[float] | np.ndarray) -> float:
    
    '''
    Description
    -----------
    Тестовая функция, которая будет использоваться далее
    для глобальных оптимизации 
    
    Return
    ------
    Функция возвращает длину фокального отрезка

    Params
    ------
    `heights_optimize_list`: Параметр оптимизации; в данном случае список высот в [мкм]
    '''
    
    validate_data_set(DATA_SET_0)

    lambda_massive = np.linspace(DATA_SET_0['lower_lambda'] * 1e9,
                                DATA_SET_0['upper_lambda'] * 1e9,
                                1000) # без [нм]
    
    focus_lambda_dict = {}
    for lmbd in lambda_massive:
        Matrix_mults_list = []

        for current_linse in range(1, DATA_SET_0['count_linse'] + 1):

            remove_degree_to_lambda_0 = (DATA_SET_0['lambda_0'][current_linse]) * 1e9 # [нм -> 10^(-9)]
            focus_0 = DATA_SET_0['focus_0'][current_linse]

            height_optimize = heights_optimize_list[current_linse-1]
            refractive_index = DATA_SET_0['refractive_index'][current_linse]

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
            Matrix_mults_list.append(Refractive_Matrix)
            
            if current_linse != DATA_SET_0['count_linse']:

                dist = DATA_SET_0['distance']['{}-{}'.format(current_linse, current_linse + 1)]
                refractive_area = DATA_SET_0['refractive_area']['{}-{}'.format(current_linse, current_linse + 1)]

                reduce_dist = dist / refractive_area

                Transfer_Matrix = np.array(
                    [
                        [1, reduce_dist],
                        [0, 1]
                    ]
                )
                Matrix_mults_list.append(Transfer_Matrix)
                
        Matrix_mults_list.reverse()
        Matrix_mults_list = np.array(Matrix_mults_list)

        mult_res = reduce(matmul, Matrix_mults_list)

        focus_lambda_dict[lmbd] = 1 / -mult_res[1, 0]
    
    all_focus = list(focus_lambda_dict.values())
    length_focus_distance = max(all_focus) - min(all_focus)
    return length_focus_distance


def optimize_function(heights_optimize_list: list):
    return test_function(heights_optimize_list)

bounds = [(1, 6) for _ in range(DATA_SET_0['count_linse'])]
def differential_evolution_new(fobj = test_function, bounds=bounds, mut=0.8, crossp=0.7, popsize=20, its=1000):
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            f = fobj(trial_denorm)
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
        yield best, fitness[best_idx]


def get_optimization(method: str) -> tuple[list, float]:

    '''
    Description
    -----------
    Пробная функция для глобальной оптимизации
    
    Return
    ------
    Функция возвращает кортеж из списка высот и фокального отрезка

    Params
    ------
    `methods'\n
    - differential_evolution
    - basinhopping
    - shgo
    - dual_annealing
    '''

    if method == 'differential_evolution':
        bounds = []
        [bounds.append((1, 6)) for _ in range(DATA_SET_0['count_linse'])]
        result = differential_evolution(func=test_function,
                                        bounds=bounds,
                                        tol=1e-5)
        return result.x.tolist(), result.fun

    elif method == 'basinhopping':
        x0 = []
        [x0.append(1.) for _ in range(DATA_SET_0['count_linse'])]
        result = basinhopping(func=optimize_function,
                              x0=x0,
                              stepsize=0.1)
        return result.x, result.fun

    elif method == 'shgo':
        result = shgo(func=optimize_function,
                      bounds=(1,6),
                      iters=3)
        return result.x, result.fun
    
    elif method == 'dual_annealing':
        result = dual_annealing(func=optimize_function,
                                bounds=(1, 6))
        return result.x, result.fun
    else: raise TypeError('Укажите тип оптимизации')