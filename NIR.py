from Algorithms.OptimizationParams.Height.BorderMethod import border_method
from Algorithms.OptimizationParams.Height.MainAlghorithm import calculate_length_focal_distance
from Algorithms.OptimizationParams.Height.ModifiedAlgorithm import modified_calculate_length_focal_distance
from Algorithms.OptimizationParams.Height.OneDimension import one_dimensional_optimization
from Data.InputTestDate import DATA_SET_0


if __name__ == '__main__':


    #with open('results.txt', 'a') as file:
    #    for elem in res:
    #        file.write(str(elem) + '\n')
    #    file.write('___________________________________________________\n\n')

    #print(*res, sep='\n')
    #calculate_length_focal_distance(data_set_0, None, True)

    
    initial_heights, heights, focal_dist = one_dimensional_optimization(DATA_SET_0)

    print(f'Начальное приближение = {initial_heights}',
          f'Список высот = {heights}',
          f'Фокальный отрезок = {focal_dist}',
          sep='\n')
    

    
