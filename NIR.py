from algorithm import (test_method, modified_calculate_length_focal_distance,
                       calculate_length_focal_distance)
from InputTestDate import data_set_0
from scipy.optimize import minimize_scalar

if __name__ == '__main__':

 
    res = test_method()
    #
    #with open('results.txt', 'a') as file:
    #    for elem in res:
    #        file.write(str(elem) + '\n')
    #    file.write('___________________________________________________\n\n')

    print(*res, sep='\n')

    eps = 1e-6
    initial_height = [1., 1., 1.]
    options = {'maxiter': 100, 'xatol': eps}

    globalIter = 0
    globalFocalDit_Dict = {}
    while(True):

        if globalIter > 1:
            if globalFocalDit_Dict['Iter_{}'.format(str(globalIter-1))] - globalFocalDit_Dict['Iter_{}'.format(str(globalIter))] < eps:
                break
        for numLinse in range(1, data_set_0['count_linse'] + 1):

            result_scipy = minimize_scalar(modified_calculate_length_focal_distance,
                                            method='bounded', bounds=(1, 10),
                                            args=(initial_height,numLinse), options=options)
            get_focal_length = result_scipy.x
            initial_height[numLinse-1] = get_focal_length
        
        globalIter += 1
        globalFocalDit_Dict[f'Iter_{globalIter}'] =  calculate_length_focal_distance(initial_height)
    
    print('Фок.отрезок = ', {globalFocalDit_Dict[f'Iter_{globalIter}']},
          f'Высоты равны = {initial_height}',
          f'Кол-во итераций = {globalIter}',
            sep='\n')
    #calculate_length_focal_distance(heightOptimizeList=[6.6364, 4.0101, 6.0303, 3.0], show_plot=True)


