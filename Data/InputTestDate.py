from Utils.Unit import Unit

"""
Данный модуль хранит информацию,
необходимую для расчёта
"""

# Словарь данных по умолчанию
# Данный словарь представляет из себя набо постоянных параметров,
# которые будут браться, если параметр оптимизации не совпадает с ключом данного словаря!

DATA_SET_0 = \
    {
    'count_linse': 2,
    'lower_lambda': 400 * Unit.NANOMETER,
    'upper_lambda': 1000 * Unit.NANOMETER,
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
            '1-2': 30 * Unit.CENTIMETR
        },
    'refractive_area' : \
        {                       # Показатели преломления пространства
                                # м/у линзами
            '1-2': 1.
        },
    'lambda_0': \
        {                       # Базовая длина волны для каждой линзы в [нм]
            1: 550 * Unit.NANOMETER,
            2: 550 * Unit.NANOMETER
        },
    'focus_0': \
        {                       # Базовый фокус для каждой линзы в [см]
            1: 50 * Unit.CENTIMETR,
            2: 50 * Unit.CENTIMETR
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