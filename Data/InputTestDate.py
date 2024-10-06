# Словарь начальных данных

DATA_SET_0 = \
    {
    'count_linse': 2,
    'lower_lambda': 400 * 1e-9,
    'upper_lambda': 1000 * 1e-9,
    'refractive_index': \
    {                       # Показатель преломления каждой линзы 
        1: 1.5,
        2: 1.5
    },
    'harmonica' : \
    {                       # Набор гармоник
        1: 7,
        2: 7
    },
    'distance': \
    {                       # Расстояние м/у линзами в [см]
        '1-2': 10 * 1e-2
    },
    'refractive_area' : \
    {                       # Показатели преломления пространства
                            # м/у линзами
        '1-2': 1.
    },
    'lambda_0': \
    {                       # Базовая длина волны для каждой линзы в [нм]
        1: 550 * 1e-9,
        2: 550 * 1e-9
    },
    'focus_0': \
    {                       # Базовый фокус для каждой линзы в [см]
        1: 100 * 1e-2,
        2: 50 * 1e-2
    }  
}


def validate_data_set(data_set: dict):

    '''
    `Description`: Проверяет корректность входных данных
    '''

    count_linse = data_set['count_linse']

    if count_linse <= 1: 
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
            case 'harmonica':
                if len(data_set[key]) != count_linse:
                    raise ValueError('Число элементов не совпадает с числом линз в системе')