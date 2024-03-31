# Словарь начальных данных
data_set_0 = {

    'count_linse': 4,
    'lower_lambda': 400 * 1e-9,
    'upper_lambda': 1000 * 1e-9,
    'refractive_index': {
        1: 1.5,
        2: 1.5,
        3: 1.5,
        4: 1.5
    },
    'distance': {
        '1-2': 150 * 1e-2,
        '2-3': 150 * 1e-2,
        '3-4': 150 * 1e-2
    },
    'lambda_0': {
        1: 500 * 1e-9,
        2: 500 * 1e-9,
        3: 500 * 1e-9,
        4: 500 * 1e-9
    },
    'focus_0': {
        1: 50 * 1e-3,
        2: 50 * 1e-3,
        3: 50 * 1e-3,
        4: 50 * 1e-3
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
            case 'lambda_0':
                if len(data_set[key]) != count_linse:
                    raise ValueError('Число элементов не совпадает с числом линз в системе')
            case 'focus_0':
                if len(data_set[key]) != count_linse:
                    raise ValueError('Число элементов не совпадает с числом линз в системе')