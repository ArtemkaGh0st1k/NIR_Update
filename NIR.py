from Algorithms.MainAlgorithm import Calc


if __name__ == '__main__':

    # Для срабатывания алгоритма необходимо подобрать гиперпараметры
    # lr, denominator и т.д

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