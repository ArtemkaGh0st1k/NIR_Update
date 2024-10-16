import numpy as np

from Data.InputTestDate import validate_data_set, DATA_SET_0


def calc_2_linse_focal_dist(data_set_0: dict,
                            heights_optimize : list[float] | np.ndarray) -> float:
    validate_data_set(data_set_0)

    lambda_massive = np.linspace(data_set_0['lower_lambda'],
                                 data_set_0['upper_lambda'],
                                 1000)
    
    focus_lambda_dict = {}
    for lmbd in lambda_massive:
        Optics_Power_List = []

        for num_linse in range(1, data_set_0['count_linse'] + 1):
            harmonica = data_set_0['harmonica'][num_linse]
            focus_0 = data_set_0['focus_0'][num_linse]
            height_optimize = heights_optimize[num_linse-1] * 1e-6
            refractive_index = data_set_0['refractive_index'][num_linse]

            lambda_0 = height_optimize * (refractive_index - 1) / harmonica            
            k = round((lambda_0 / (lmbd)) * harmonica)
            if k == 0: k = 1

            focus = ((harmonica * lambda_0) / (k * lmbd)) * focus_0
            optic_power = 1 / focus

            Optics_Power_List.append(optic_power)
    
        optic_power_1 = Optics_Power_List[0]
        optic_power_2 = Optics_Power_List[1]
        distance = data_set_0['distance']['1-2']

        Optic_Power = optic_power_1 + optic_power_2 - \
                      distance * optic_power_1 * optic_power_2
        focus_lambda_dict[lmbd] = 1 / Optic_Power

    all_focus = list(focus_lambda_dict.values())
    length_focus_distance = max(all_focus) - min(all_focus)
    
    return length_focus_distance


def find_common_focus(initial_data_set : dict = None) -> float:
    '''
    Description:
    ------------
    Функция находит общий фокус системы, который должен
    оставаться постоянным, когда идёт подбор параметров 

    :Params:\n
    -------
    `initial_data_set`:\n
    Если `None`, то обращается к базовому словараю\n
    Иначе ищет фокус по заданным значениям в поданном словаре
    '''

    if initial_data_set is None:
        initial_data_set = DATA_SET_0.copy()

    try:
        get_focus = initial_data_set['focus_0']
        get_distance = initial_data_set['distance']
        pass
    except KeyError as ke:
        raise KeyError(f'Не найдены значения в словаре по такому ключу: {ke.args}')

    count_linse = initial_data_set['count_linse']
   
    focus = ( (initial_data_set['focus_0'][1] * initial_data_set['focus_0'][2]) / 
            (initial_data_set['focus_0'][1] + initial_data_set['focus_0'][2]) - initial_data_set['distance']['1-2'] )
    if count_linse == 2: return focus
    
    for i in range(2, count_linse):
        f0 = initial_data_set['focus_0'][i+1]
        distance = initial_data_set['distance'][f'{i}-{i+1}']

        focus = (f0 * focus) / (f0 + focus - distance)

    return focus