from Data.InputTestDate import DATA_SET_0
from Algorithms.MainAlgorithm import Calc
from Algorithms.MainAlgorithm_Ver_2_0 import Calc_Ver_2_0
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import os
from colorama import Fore


if __name__ == '__main__':

    initial_data_set = \
    {
        'height' : \
        {   # в [мкм]
            1: 6,
            2: 6
        }
    }
    calc = Calc()
    calc_ver_2_0 = Calc_Ver_2_0()

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