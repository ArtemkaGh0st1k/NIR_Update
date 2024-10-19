from Data.InputTestDate import DATA_SET_0
from Algorithms.MainAlgorithm import Calc
from Algorithms.MainAlgorithm_Ver_2_0 import Calc_Ver_2_0
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import os
from colorama import Fore
import sys
from Utils.Unit import Unit
from Utils.Calculate import find_common_focus

if __name__ == '__main__':

    # Для срабатывания алгоритма необходимо подобрать гиперпараметры
    # lr, denominator и т.д

    
    initial_data_set = \
    {
        'height' : \
        {   
            1: 6,
            2: 6 
        }
    }

    calc = Calc()

    h0 = 6
    for i in range(5):
        initial_data_set = \
        {
            'height' : \
            {
                1: h0 + i,
                2: h0 + i
            }
        } 

        calc.visualization_gradient_descent(initial_data_set=initial_data_set)
        print(f'Найденные высоты_1 равны = {calc.get_h}')