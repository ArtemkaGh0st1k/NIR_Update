from algorithm import (test_method, modified_calculate_length_focal_distance,
                       calculate_length_focal_distance, OneDimensionalOptimization)
from InputTestDate import data_set_0


if __name__ == '__main__':

 
    #res = chatGPT_test_method(data_set_0)
    #
    #with open('results.txt', 'a') as file:
    #    for elem in res:
    #        file.write(str(elem) + '\n')
    #    file.write('___________________________________________________\n\n')

    #print(*res, sep='\n')

    calculate_length_focal_distance(data_set_0, None, True)

    
    initial_heights, heights, focal_dist = OneDimensionalOptimization(data_set_0)

    print(f'Начальное приближение = {initial_heights}',
          f'Список высот = {heights}',
          f'Фокальный отрезок = {focal_dist}',
          sep='\n')
    
    
    #print(f'Длина фок. отрезка = {calculate_length_focal_distance(data_set_0, initial_height) * 100} см')
    #calculate_length_focal_distance(heightOptimizeList=[6.6364, 4.0101, 6.0303, 3.0], show_plot=True)

    
