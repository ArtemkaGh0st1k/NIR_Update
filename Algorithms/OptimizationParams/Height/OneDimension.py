import pytest
from scipy.optimize import minimize, minimize_scalar
from scipy.optimize._optimize import (fmin, fmin_ncg, fmin_bfgs, 
                                      fmin_cg, fmin_powell, fminbound)
from scipy.optimize.minpack2 import (dcsrch, dcstep)
from scipy.optimize._minpack import (_hybrd, _hybrj, _chkder)
from scipy.optimize._bracket import (_bracket_root, _bracket_minimum,
                                     _bracket_minimum_iv, _bracket_root_iv)
from scipy.optimize._basinhopping import (MinimizerWrapper, inspect, 
                                          Storage, basinhopping, Metropolis)
import scipy.optimize.tests.test__basinhopping as TestBasinhoping
import scipy.optimize.tests.test_optimize as testOptimize
import scipy.optimize.tests.test_bracket as TestBracket

from Data.InputTestDate import validate_data_set
from MainAlghorithm import calculate_length_focal_distance
from ModifiedAlgorithm import modified_calculate_length_focal_distance

# матрица преломления -> R = [1 0; -Ф 1]
# матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами
# формула для фокусн.раст f(k,lamda) = (m * lamda_0) / (k * lamda) * f_0
# итоговая матрица -> перемножение всех матриц с конца


def one_dimensional_optimization(data_set_0: dict, 
                                 initial_heights : list = None,
                                 eps = 1e-5
                                ) -> tuple[list[float], list[float], float]:

    '''
    Description
    -----------
    Одномерная оптимизация

    Return
    ------
    Возвращает кортеж ->
            1) Список высот [мкм]\n
            2) Начальное приближение [мкм]\n
            3) Фокальный отрезок в [м]

    Warning
    -------
    Данный метод очень сильно зависит
    от начального приближения

    Params
    ------
    `data_set_0`: начальный датасет
    `eps`: точность
    '''

    validate_data_set(data_set_0)

    if initial_heights is None:
        initial_heights = []
        [initial_heights.append(1.) for _ in range(data_set_0['count_linse'])]
    
    retuned_initial_heihts = initial_heights.copy()
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
                                                method='bounded', bounds=(1, 6),
                                                args=(retuned_initial_heihts, numLinse, data_set_0), options=options)
                if result_scipy.success:
                    height = float(result_scipy.x)
                    retuned_initial_heihts[numLinse-1] = height
                else:
                    raise TypeError('Внимательно подавайте аргументы в функцию!')
            
            localIter += 1
            localFocalDit_Dict[f'Iter_{localIter}'] =  calculate_length_focal_distance(data_set_0, retuned_initial_heihts)

        localFocalDit_Dict.clear()
        globalIter +=1
        globalFocalDit_Dict[f'Iter_{globalIter}'] = calculate_length_focal_distance(data_set_0, retuned_initial_heihts)  

    return retuned_initial_heihts, initial_heights, globalFocalDit_Dict[f'Iter_{globalIter}']
      