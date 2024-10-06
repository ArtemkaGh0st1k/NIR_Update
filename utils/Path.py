import os
from os import getcwd
from os.path import isdir, join

from Data.InputTestDate import DATA_SET_0

def create_dir(is_params : dict[str, bool]) -> str:
    count_linse = DATA_SET_0['count_linse']

    path = join(getcwd(), "NIR_Update", "Result")
    if not isdir(path):
        os.mkdir(path)

    if (is_params["h"] and
        is_params["d"] and 
        is_params["f_0"]):
        path = join(path,"H_D_F0")
        if not isdir(path): os.mkdir(path)
        path = join(path, f"{count_linse}_Linse")
        if not isdir(path): os.mkdir(path)
    elif (is_params["h"] and
            is_params["d"]):
        path = join(path, "H_D")
        if not isdir(path): os.mkdir(path)
        path = join(path, f"{count_linse}_Linse")
        if not isdir(path): os.mkdir(path)
    elif (is_params["h"] and
            is_params["f_0"]):
        path = join(path, "H_F0")
        if not isdir(path): os.mkdir(path)
        path = join(path, f"{count_linse}_Linse")
        if not isdir(path): os.mkdir(path)
    elif (is_params["f_0"] and
            is_params["d"]):
        path = join(path, "D_F0")
        if not isdir(path): os.mkdir(path)
        path = join(path, f"{count_linse}_Linse")
        if not isdir(path): os.mkdir(path)
    elif (is_params["h"]):
        path = join(path, "H")
        if not isdir(path): os.mkdir(path)
        path = join(path, f"{count_linse}_Linse")
        if not isdir(path): os.mkdir(path)
    elif (is_params["d"]):
        path = join(path, "D")
        if not isdir(path): os.mkdir(path)
        path = join(path, f"{count_linse}_Linse")
        if not isdir(path): os.mkdir(path)
    elif (is_params["f_0"]):
        path = join(path, "F0")
        if not isdir(path): os.mkdir(path)
        path = join(path, f"{count_linse}_Linse")
        if not isdir(path): os.mkdir(path)

    return path