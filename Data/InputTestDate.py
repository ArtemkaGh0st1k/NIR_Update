# Словарь начальных данных

data_set_0 = {

    'count_linse': 3,
    'lower_lambda': 400 * 1e-9,
    'upper_lambda': 1000 * 1e-9,
    'refractive_index': {   # Показатель преломления каждой линзы 
        1: 1.5,
        2: 1.5,
        3: 1.5
    },
    'distance': {           # Расстояние м/у линзами в [см]
        '1-2': 20 * 1e-3,
        '2-3': 20 * 1e-3,
    },
    'refractive_area' : {   # Показатели преломления пространства
                            # м/у линзами
        '1-2': 1.,
        '2-3': 1.
    },
    'lambda_0': {           # Базовая длина волны для каждой линзы в [нм]
        1: 500 * 1e-9,
        2: 500 * 1e-9,
        3: 500 * 1e-9,
    },
    'focus_0': {            # Базовый фокус для каждой линзы в [см]
        1: 50 * 1e-3,
        2: 50 * 1e-3,
        3: 50 * 1e-3,
    }  
}


def validateDataSet(data_set: dict):

    '''
    `Description`: Проверяет корректность входных данных
    '''

    count_linse = data_set['count_linse']

    if data_set_0['count_linse'] <= 1: 
        raise ValueError('Количество линз должно быть больше 1')

    for key in data_set:

        match key:
            case 'refractive_index':
                if len(data_set[key]) != count_linse: 
                    raise ValueError('Число элементов не совпадает с числом линз в системе')
            case 'distance':
                if len(data_set[key]) != count_linse-1:
                    raise ValueError('Число элементов не совпадает с числом линз в системе')
            case 'refractive_area':
                if len(data_set[key]) != count_linse-1:
                    raise ValueError('Число элементов не совпадает с числом линз в системе')
            case 'lambda_0':
                if len(data_set[key]) != count_linse:
                    raise ValueError('Число элементов не совпадает с числом линз в системе')
            case 'focus_0':
                if len(data_set[key]) != count_linse:
                    raise ValueError('Число элементов не совпадает с числом линз в системе')