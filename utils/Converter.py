import numpy as np

def custom_converter_tolist(size : int, current_list):
    
    param_global = []
    param_local = []
    for i in range(size):
        for param in current_list:
            param_local.append(param[i])
        param_global.append(param_local.copy())
        param_local.clear()
    return param_global


def custom_converter_tostr(initial_data_set : dict,
                           is_params : dict[str, bool]) -> str:
    
    optimize_param_name = "h_" + "_".join(str(np.round(list(initial_data_set['height'].values()), 3)).
                                                replace("[", "").
                                                replace("]", "").
                                                replace(",", "").
                                                split()) + \
                                                ('_|d_' + '_'.join(str(np.round(list(initial_data_set['distance'].values()), 3)).
                                                replace("[", "").
                                                replace("]", "").
                                                replace(",", "").
                                                split()) 
                                                if is_params['d']
                                                else "") + \
                                                ('_|f0_' + '_'.join(str(np.round(list(initial_data_set['focus_0'].values()), 3)).
                                                replace("[", "").
                                                replace("]", "").
                                                replace(",", "").
                                                split())
                                                if is_params['f_0']
                                                else "")
    return optimize_param_name