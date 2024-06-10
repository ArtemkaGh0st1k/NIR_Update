from scipy.optimize import minimize, minimize_scalar

from Validate.InputTestDate import validateDataSet
from MainAlghorithm import calculate_length_focal_distance
from ModifiedAlgorithm import modified_calculate_length_focal_distance

# матрица преломления -> R = [1 0; -Ф 1]
# матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами
# формула для фокусн.раст f(k,lamda) = (m * lamda_0) / (k * lamda) * f_0
# итоговая матрица -> перемножение всех матриц с конца


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

    validateDataSet(data_set_0)

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
      