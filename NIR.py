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
 
        
    heights, initial_heights, focal_dist = one_dimensional_optimization(DATA_SET_0)

    print(f'Начальное приближение = {initial_heights} мкм',
          f'Список высот = {heights} мкм',
          f'Фокальный отрезок = {focal_dist * 100} см',
          sep='\n')
    

    
