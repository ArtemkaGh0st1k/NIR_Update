import numpy as np
from colorama import Fore
import time

from Data.InputTestDate import DATA_SET_0


def __gradient_descent(self, initial_params: dict) -> dict[str, tuple[np.ndarray[float], list[float]] | None]:
    
        '''
        Description:
        ------------
        Основной алгоритм градиентного спуска

        Return:
        -------
        Возвращяет кортеж из `списка найденных высот` и\n
        `список всех высот, которые вычислялись в алгоритме`

        :Params:\n
        `initial_params`: Начальное приближение
        '''
        
        # Проверка на выход из допустимого диапазона параметров оптимизации
        collect_data = self.__collect_min_max_params_range(initial_params)

        h = None
        d = None
        f_0 = None
        d_list = None
        f_0_list = None
        h_list = []

        for i, key in enumerate(initial_params, start=1):
            for j in range(1, DATA_SET_0['count_linse']  + 1):
                match key:
                    case 'height':
                        min_h = collect_data[j][key][0]
                        max_h = collect_data[j][key][1]

                        if initial_params[key][j] < min_h: initial_params[key][j] = min_h
                        elif initial_params[key][j] > max_h: initial_params[key][j] = max_h

                    case 'focus_0':
                        initial_params[key][j] = collect_data[j][key]

        initial_params_copy = initial_params.copy()
        if initial_params['distance']:
            d = np.array(list(initial_params_copy['distance'].values())) * 1e2
            d_list = [d.copy()]
        if initial_params['focus_0']:
            f_0 = np.array(list(initial_params['focus_0'].values())) * 1e2
            f_0_list = [f_0.copy()]
        
        h = np.array(list(initial_params_copy['height'].values())) * 1e6
        h_list.append(h.copy())

        start_time = time.time()
        for _ in range(self._MAX_ITER):

            grad = self.__numerical_gradient(initial_params_copy)

            h -= self._LEARNING_RATE['h'] * grad[0]
            h_list.append(h.copy())
            
            initial_params_copy['height'] = \
            {
                key : value * 1e-6
                for key, value
                in zip(list(range(1, len(h) + 1)), h)
            }

            if d is not None:
                d -= self._LEARNING_RATE['d'] * grad[1]
                d_list.append(d.copy())

                initial_params_copy['distance'] = \
                {
                    '{}-{}'.format(key, key + 1) : value * 1e-2
                    for key, value 
                    in zip(list(range(1, len(d) + 1)) , d)
                }
            if f_0 is not None:
                f_0 -= self._LEARNING_RATE['f_0'] * grad[2]
                f_0_list.append(f_0.copy())

                initial_params_copy['focus_0'] = \
                {
                    key: value * 1e-2
                    for key, value
                    in zip(list(range(1, len(f_0) + 1)), f_0)
                }
            
        time_grad_desc = time.time() - start_time

        self._h = h
        self._d = d 
        self._f_0 = f_0
        self._time_grad_desc = time_grad_desc

        print(Fore.BLUE, "Время работы град.спуска = {} м {} с".format(time_grad_desc // 60, time_grad_desc % 60))
        return \
            {
                'h'     : (h, h_list),
                'd'     : (d, d_list),
                'f_0'   : (f_0, f_0_list)
            }

def __numerical_gradient(self, initial_data_set : dict) -> tuple[np.ndarray[float],
                                                                    np.ndarray[float] | None,
                                                                    np.ndarray[float] | None]:
    '''
    Description:
    ---------------
    Используется численный градиент, т.к целевая функция имеет неявный вид

    Return:
    ----------
    Возвращяет численный градиент

    :Params:\n
    '''
    
    is_params = self.__collect_optimize_params(initial_data_set)
    
    grad_h = np.zeros_like(list(initial_data_set['height'].values()))
    grad_d = (np.zeros_like(list(initial_data_set['distance'].values())) 
                if is_params['d']
                else None)
    grad_f_0 = (np.zeros_like(list(initial_data_set['focus_0'].values()))
                if is_params['f_0']
                else None)

    for i in range(1, DATA_SET_0['count_linse'] + 1):
        orig_val_h = initial_data_set['height'][i]

        initial_data_set['height'][i] = orig_val_h + self._STEP['h']
        loss_func_plus_h = self.__loss_function(initial_data_set)

        initial_data_set['height'][i] = orig_val_h
        loss_func_orig_h = self.__loss_function(initial_data_set)
                
        grad_h[i-1] = (loss_func_plus_h - loss_func_orig_h) / self._DENOMINATOR['h']

        initial_data_set['height'][i] = orig_val_h

        if is_params['d'] and i != DATA_SET_0['count_linse']:
            orig_val_d = initial_data_set['distance'][f'{i}-{i+1}'] 

            initial_data_set['distance'][f'{i}-{i+1}'] = orig_val_d + self._STEP['d']
            loss_func_plus_d = self.__loss_function(initial_data_set)

            initial_data_set['distance'][f'{i}-{i+1}'] = orig_val_d
            loss_func_orig_d = self.__loss_function(initial_data_set)
                    
            grad_d[i-1] = (loss_func_plus_d - loss_func_orig_d) / self._DENOMINATOR['d']

            initial_data_set['distance'][f'{i}-{i+1}'] = orig_val_d
        
        if is_params['f_0']:
            orig_val_f_0 = initial_data_set['focus_0'][i]

            initial_data_set['focus_0'][i] = orig_val_f_0 + self._STEP['f_0']
            loss_func_plus_f_0 = self.__loss_function(initial_data_set)

            initial_data_set['focus_0'][i] = orig_val_f_0
            loss_func_orig_f_0 = self.__loss_function(initial_data_set)
                    
            grad_f_0[i-1] = (loss_func_plus_f_0 - loss_func_orig_f_0) / self._DENOMINATOR['f_0']

            initial_data_set['focus_0'][i] = orig_val_f_0

    return (grad_h, grad_d, grad_f_0)

    '''
    @staticmethod
    def calculate_length_focal_distance(self,
                                        initial_params : dict,
                                        save_plot = False) -> float:
    
        """
        Description
        -----------
        Основная функция для графической и расчётной интерпретации
        зависимости длины фок.расстояния от длины волны\n
        Формулы и методы используемые в алгоритме:\n
        Матрица преломления -> R = [1 0; -Ф 1]\n
        Матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами\n
        Формула для фокусн.раст f(k,lamda) = (m * lamda_0) / (k * lamda) * f_0\n
        Итоговая матрица -> перемножение всех матриц с конца
        
        Return
        ------
        Функция возвращает длину фокального отрезка в [м]

        :Params:\n
        ------
        `data_set_0`: Исходные данные в виде словаря\n
        `height_optimize_list`: Параметр оптимизации; в данном случае список высот\n
        `save_plot`: Если true -> Выводит график зависимсоти длины волны от фокусного расстояния
        """
        
        validate_data_set(DATA_SET_0)

        lambda_massive = np.linspace(DATA_SET_0['lower_lambda'] * 1e9,
                                     DATA_SET_0['upper_lambda'] * 1e9,
                                     1000)
        
        focus_lambda_dict = {}
        for lmbd in lambda_massive:
            Matrix_Mults_List = []
            for i in range(1, DATA_SET_0['count_linse'] + 1):

                remove_degree_to_lambda_0 = (data_set_0['lambda_0'][i]) * 1e9 # [нм -> 10^(-9)]
                focus_0 = data_set_0['focus_0'][i]

                if height_optimize_list is None:
                    harmonica = DATA_SET_0['harmonica'][i]

                    k = round((remove_degree_to_lambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?
                    focus = ((harmonica * remove_degree_to_lambda_0) / (k * lmbd)) * focus_0
                else:
                    height_optimize = height_optimize_list[i-1]
                    refractive_index = DATA_SET_0['refractive_index'][i]

                    harmonica = (height_optimize * (refractive_index - 1) / remove_degree_to_lambda_0) * 1e3
                    k = round((remove_degree_to_lambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?
                    if k == 0: k = 1
                    focus = ((harmonica * remove_degree_to_lambda_0) / (k * lmbd)) * focus_0

                optic_power = 1 / focus          
                Refractive_Matrix = np.array(
                    [
                        [1, 0],
                        [-optic_power, 1]
                    ]
                )
                Matrix_Mults_List.append(Refractive_Matrix)
                
                if i != DATA_SET_0['count_linse']:

                    refractive_area = DATA_SET_0['refractive_area']['{}-{}'.format(i, i + 1)]
                    dist = DATA_SET_0['distance']['{}-{}'.format(i, i + 1)]

                    reduce_dist = dist / refractive_area

                    Transfer_Matrix = np.array(
                        [
                            [1, reduce_dist],
                            [0, 1]
                        ]
                    )
                    Matrix_Mults_List.append(Transfer_Matrix)
                    
            Matrix_Mults_List.reverse()
            Matrix_Mults_List = np.array(Matrix_Mults_List)

            mult_res = reduce(matmul, Matrix_Mults_List)

            focus_lambda_dict[lmbd] = - 1 / mult_res[1, 0]

        all_focus = list(focus_lambda_dict.values())
        length_focus_distance = max(all_focus) - min(all_focus)

        if save_plot: 
            if height_optimize_list is None:
                optimize_param_name = "m_" + "_".join(str(list(data_set_0['harmonica'].values())).
                                                    replace("[", "").
                                                    replace("]", "").
                                                    replace(",", "").
                                                    split())
            else:
                optimize_param_name = "h_" + "_".join(str(height_optimize_list).
                                                    replace("[", "").
                                                    replace("]", "").
                                                    replace(",", "").
                                                    split())

            save_path = os.path.join("NIR_Update", "Result", "NewHeights", f"{optimize_param_name}.png")
            all_lamdba = list(focus_lambda_dict.keys())

            lambda_list = np.array(all_lamdba)
            focus_list = np.array(all_focus) * 100

            plt.figure(figsize=(7,6))
            plt.title('Зависимость длины волны от фокусного расстояния')
            plt.xlabel('Длина волны, нм')
            plt.ylabel('Фокусное расстояние, см')
            plt.grid(True)
            plt.plot(lambda_list, focus_list)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), title="Длина фокального отрезка: {} см".format(length_focus_distance * 100))
            plt.savefig(save_path)
            plt.close()

            print(Fore.GREEN,
                'Длина фокального отрезка: {} см'.format(length_focus_distance * 100),
                "Высоты равны = {}".format(height_optimize_list), 
                sep=" | ")
    
        return length_focus_distance
    '''
    