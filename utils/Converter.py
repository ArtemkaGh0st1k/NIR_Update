import numpy as np


def custom_converter_tolist(size : int, current_list):
    
    h_global = []
    h_local = []
    for i in range(size):
        for elem in current_list:
            h_local.append(elem[i])
        h_global.append(h_local.copy())
        h_local.clear()
    return h_global