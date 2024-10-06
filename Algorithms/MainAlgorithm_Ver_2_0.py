from operator import matmul
from functools import reduce
import time 
import os
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from accessify import private, protected

from Data.InputTestDate import validate_data_set, DATA_SET_0
from Utils.Converter import (custom_converter_tolist,
                             custom_converter_tostr)
from Utils.Plot import create_subplots

class Calc_Ver_2_0():
    '''
    `Description`: Основной класс расчёта
    '''
    def __init__(self, 
                 max_iter=1000,
                 lr : dict[str, float] = { 'h':0.05, 'd':0.01, 'f_0':0.01 },
                 step : dict[str, float] = { 'h':0.1, 'd': 5, 'f_0': 5 },
                 denominator : dict[str, float] = { 'h':1e-3, 'd':1e-3, 'f_0':1e-3 }) -> None:
        '''
        Descroption: 
        ------------
            Инициализирует такие параметры как ->\n
            Макс.число итераций\n
            Словарь обучающего шага для каждого из параметров\n
            Словарь шага суммирования у каждого параметра\n
            Словарь знаменателей для каждого параметра
        '''
        
        self._LEARNING_RATE = lr
        self._STEP = step
        self._MAX_ITER = max_iter
        self._DENOMINATOR = denominator

        self._time_grad_desc = 0
        self._h = None
        self._d = None
        self._f_0 = None

    
    @property
    def get_h(self) -> np.ndarray[float]:
        return self._h
    
    @property
    def get_d(self) -> np.ndarray[float]:
        return self._d
    
    @property
    def get_f_0(self) -> np.ndarray[float]:
        return self._f_0
    
    @property
    def get_time_grad_desc(self) -> tuple[int, float]:
        '''
        Возвращает время, затраченное на алгоритм град.спуска
        в минутах и секундах
        '''
        return (self._time_grad_desc // 60, self._time_grad_desc % 60)
    
# Приватные методы, которые являются как вспомгательные к основным метода расчета и визуализации

    def __collect_optimize_params(self, initial_data_set : dict) -> dict[str, bool]:
            is_h = False
            is_distance = False
            is_focus_0 = False

            for key in initial_data_set:
                match key:
                    case "height":
                        is_h = not is_h
                    case "distance":
                        is_distance = not is_distance
                    case "focus_0":
                        is_focus_0 = not is_focus_0 
            
            return {
                    'h' : is_h,
                    'd' : is_distance,
                    'f_0' : is_focus_0
                    }


    def __collect_min_max_params_range(self, initial_data_set: dict) -> dict:

        '''
        Description:
        ------------
        Возвращяет минимальные и максимальные высоты каждой линзы,
        пригодные для алгоритма оптимизации в [мкм]  

        Return:
        -------
        Возвращает словарь, в котором ключ - номер линзы,\n
        а значение - кортеж мин и макс допустимый высоты для данной линзы
        '''

        is_params = self.__collect_optimize_params(initial_data_set)
        #FIXME: Учесть, то что h может быть не обязательный параметр


        lower_lambda = DATA_SET_0['lower_lambda']
        upper_lambda = DATA_SET_0['upper_lambda']
        if is_params['d'] and is_params['f_0']: 
            max_distance = max(initial_data_set['distance'].values())

        collect_data = {}
        for i in range(1, DATA_SET_0['count_linse'] + 1):   

            if is_params['h']:
                harmonica = DATA_SET_0['harmonica'][i]
                refractive_index = DATA_SET_0['refractive_index'][i]
                min_h = harmonica / (refractive_index - 1) * lower_lambda
                max_h = harmonica / (refractive_index - 1) * upper_lambda

                collect_data[i] = \
                {
                    'height' : (min_h, max_h)
                }

                if is_params['d'] and is_params['f_0']:
                    min_focus_0 = max(max_distance, initial_data_set['focus_0'][i])
                    collect_data[i] = \
                    {
                        'height' : (min_h, max_h),
                        'focus_0': min_focus_0
                    }   

        return collect_data 
    

    def loss_function(self,
                      initial_data_set: dict,
                      save_plot = False) -> float:
        
        '''
        Warning:
        --------
        По умолчанию высота является обязательным параметром оптимизации.\n
        В противном случае алгоритм не пойдёт дальше и выдаст ошибку! 

        Description:
        -----------
        Целевая функция\n
        Формулы и методы используемые в алгоритме:\n
        m -> гармоника, целое фиксированное число\n
        lambda_0 -> базовая длина волны, идёт подбор этой длины в зависимости от поданной высоты\n
        f = (m * lambda_0) / (k * lambda) * f_0 \n
        матрица преломления -> R = [1 0; -D 1], где D - оптич.сила\n
        матрица переноса -> T = [1 d/n; 0 1], где d - расстояние м/у линзами

        Return: 
        -------
        Возвращяет фокальный отрезок

        :Params:\n
        -------
        `data_set_0`: Исходные данные в виде словаря\n
        `heights_optimize`: Массив высот (приходит не в [мкм]!!!)
        '''

        is_params = self.__collect_optimize_params(initial_data_set)

        validate_data_set(DATA_SET_0)
        lambda_massive = np.linspace(DATA_SET_0['lower_lambda'],
                                     DATA_SET_0['upper_lambda'],
                                     1000) 
        
        focus_lambda_dict = {}
        for lmbd in lambda_massive:
            Matrix_Mults_List = []

            for i in range(1, DATA_SET_0['count_linse'] + 1):
                
                harmonica = DATA_SET_0['harmonica'][i]
                refractive_index = DATA_SET_0['refractive_index'][i]

                focus_0 = (initial_data_set['focus_0'][i] * 1e-2
                           if is_params['f_0']
                           else DATA_SET_0['focus_0'][i]) 
                
                height = initial_data_set['height'][i] * 1e-6
                lambda_0 = height * (refractive_index - 1) / harmonica            
                k = round((lambda_0 / (lmbd)) * harmonica)
                if k == 0: k = 1

                focus = ((harmonica * lambda_0) / (k * lmbd)) * focus_0
                optic_power = 1 / focus 

                Refractive_Matrix = np.array(
                    [
                        [1, 0],
                        [-optic_power, 1]
                    ]
                )
                Matrix_Mults_List.append(Refractive_Matrix)
                
                if i != DATA_SET_0['count_linse']:

                    refractive_area = DATA_SET_0['refractive_area']['{}-{}'.format(i, i+1)]
                    dist = (initial_data_set['distance']['{}-{}'.format(i, i+1)] * 1e-2
                            if is_params['d']
                            else DATA_SET_0['distance']['{}-{}'.format(i, i+1)])

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
            optimize_param_name = custom_converter_tostr(initial_data_set,
                                                         is_params)


            save_path = os.path.join("NIR_Update",
                                     "Result",
                                     "NewHeights",
                                     f"{optimize_param_name}.png")
            all_lamdba = list(focus_lambda_dict.keys())

            lambda_list = np.array(all_lamdba)
            focus_list = np.array(all_focus) * 100

            plt.figure(figsize=(7,6))
            plt.title('Зависимость длины волны от фокусного расстояния')
            plt.xlabel('Длина волны, нм')
            plt.ylabel('Фокусное расстояние, см')
            plt.grid(True)
            plt.plot(lambda_list, focus_list)
            plt.legend(loc='upper center',
                       bbox_to_anchor=(0.5, 1.0),
                       title="Длина фокального отрезка: {} см".format(length_focus_distance * 100))
            plt.savefig(save_path)
            plt.close()

            print(Fore.GREEN,
                'Длина фокального отрезка: {} см'.format(length_focus_distance * 100),
                'Высоты равны = {}'.format(initial_data_set['height'].values()),
                'Базовые фокусы равны = {}'.format(list(initial_data_set['focus_0'].values())) if is_params['f_0'] else "",
                'Расстояния м/у линзами равны = {}'.format(list(initial_data_set['distance'].values())) if is_params['d'] else "",
                sep=' | ')
        
        return length_focus_distance
    

    def __numerical_gradient(self, 
                             initial_data_set : dict,
                             key : str) -> np.ndarray[float]:
        
        grad = np.zeros_like(list(initial_data_set[key].values())).astype('float64')
        for i in range(1, len(list(initial_data_set[key].values())) + 1):
            if key is not 'distance':
                orig_val = initial_data_set[key][i]

                initial_data_set[key][i] = orig_val + self._STEP[key[0]]
                loss_func_plus_step = self.loss_function(initial_data_set)

                initial_data_set[key][i] = orig_val
                loss_func_plus_orig = self.loss_function(initial_data_set)

                grad[i-1] = (loss_func_plus_step - loss_func_plus_orig) / self._DENOMINATOR[key[0]]

                initial_data_set[key][i] -= self._LEARNING_RATE[key[0]] * grad[i-1]
            else:
                orig_val = initial_data_set[key][f'{i}-{i+1}']

                initial_data_set[key][f'{i}-{i+1}'] = orig_val + self._STEP[key[0]]
                loss_func_plus_step = self.loss_function(initial_data_set)

                initial_data_set[key][f'{i}-{i+1}'] = orig_val
                loss_func_plus_orig = self.loss_function(initial_data_set)

                grad[i-1] = (loss_func_plus_step - loss_func_plus_orig) / self._DENOMINATOR[key[0]]

                initial_data_set[key][f'{i}-{i+1}'] -= self._LEARNING_RATE[key[0]] * grad[i-1]

        return grad
    
    
    def __gradient_descent(self, initial_data_set : dict) -> tuple[ list[list[float]] | None,
                                                                    list[list[float]] | None,
                                                                    list[list[float]] | None]: 
        is_params = self.__collect_optimize_params(initial_data_set)
        collect_data = self.__collect_min_max_params_range(initial_data_set)
        for key in enumerate(initial_data_set, start=1):
            for j in range(1, DATA_SET_0['count_linse']  + 1):
                match key:
                    case 'height':
                        min_h = collect_data[j][key][0] # TODO: Нужно домножить на 1e6???
                        max_h = collect_data[j][key][1]

                        if initial_data_set[key][j] < min_h: initial_data_set[key][j] = min_h
                        elif initial_data_set[key][j] > max_h: initial_data_set[key][j] = max_h

        initial_data_set_copy = initial_data_set.copy()
        list_h, list_d, list_f0 = None, None, None
        if is_params['h']:
            list_h = [[initial_data_set_copy['height'][i]] for i in range(1, DATA_SET_0['count_linse'] + 1)]   
            h = list(initial_data_set_copy['height'].values())
            grad_h = 0
        if is_params['d']:
            list_d = [[initial_data_set_copy['distance'][f'{i}-{i+1}']] for i in range(1, DATA_SET_0['count_linse'])]
            d = list(initial_data_set_copy['distance'].values())
        if is_params['f_0']:
            list_f0 = [[initial_data_set_copy['focus_0'][i]] for i in range(1, DATA_SET_0['count_linse'] + 1)]
            f0 = list(initial_data_set_copy['focus_0'].values())
            grad_f0 = 0


        #FIXME: Вопрос с размерностью параметров!
        start_time = time.time()
        for _ in range(self._MAX_ITER):
        
            if is_params['h']:
                grad_h = self.__numerical_gradient(initial_data_set_copy, key='height')   

                h -= self._LEARNING_RATE['h'] * grad_h
                [list_h[i].append(h[i]) for i in range(len(h))]

                initial_data_set_copy['height'] = \
                {
                    key : value
                    for key, value
                    in zip(list(range(1, len(h) + 1)), h)
                }

            if is_params['d']:
                grad_d = self.__numerical_gradient(initial_data_set_copy, key='distance')

                d -= self._LEARNING_RATE['d'] * grad_d
                list_d.append(d.copy())

                initial_data_set_copy['distance'] = \
                {
                    '{}-{}'.format(key, key + 1) : value
                    for key, value 
                    in zip(list(range(1, len(d) + 1)) , d)
                }

            if is_params['f_0']:
                grad_f0 = self.__numerical_gradient(initial_data_set_copy, key='focus_0')

                f0 -= self._LEARNING_RATE['f_0'] * grad_f0
                list_f0.append(f0.copy())

                initial_data_set_copy['focus_0'] = \
                {
                    key: value 
                    for key, value
                    in zip(list(range(1, len(f0) + 1)), f0)
                }
        
        time_grad_desc = time.time() - start_time

        self._h = [h[-1] for h in list_h] if list_h else list_h
        self._d = [d[-1] for d in list_d] if list_d else list_d
        self._f_0 = [f0[-1] for f0 in list_f0] if list_f0 else list_f0

        print(Fore.YELLOW, "Время работы град.спуска =",
              Fore.RED, "{} мин".format(time_grad_desc // 60),
              Fore.GREEN, "{} сек".format(time_grad_desc % 60),
              Fore.WHITE)

        return (list_h, list_d, list_f0)
        

    def visualization_gradient_descent(self, 
                                       initial_data_set: dict,
                                       fname = None):
        
        '''
        Description:
        ------------
        Основной алгоритм для вывода и визуализации градиентного спуска

        :Params:\n
        `initial_heights`: Начальное приближение для высот (подаётся в [мкм]!)\n
        `fname`: Полный путь для сохранения результата визуализации
        '''

        count_linse = DATA_SET_0['count_linse']
        common_list = []
        
        is_params = self.__collect_optimize_params(initial_data_set) # Проверка: какие параметры будем оптимизировать
        borders = self.__collect_min_max_params_range(initial_data_set) # Рассчитываем допустимый диапазон параметров
                                                                      # #TODO: Нужно будет подкорректировать, учесть что h - необязат.парам 
        optimize_params = self.__gradient_descent(initial_data_set) # Получаем результат и важно, 
                                                                      # что initial_params должен поменяться,
                                                                      # с учётом новых параметров 
        focus_dist = self.loss_function(initial_data_set) # Как раз можно проверить, что на вход
                                                        # идут уже обновленные парамеры

        nrow_h = 0
        ncols_h = 0
        if is_params['h']:
            nrow_h += 1
            ncols_h += count_linse
            common_list.append({ 'h' : optimize_params[0]})

        nrow_d = 0
        ncols_d = 0
        if is_params['d']:
            nrow_d += 1
            ncols_d += count_linse - 1
            common_list.append({ 'd' : optimize_params[1]})
            
        nrow_f_0 = 0
        ncols_f_0 = 0
        if is_params['f_0']:
            nrow_f_0 += 1
            ncols_f_0 += count_linse
            common_list.append({ 'f_0' : optimize_params[2]})

        # Постройка графиков
        nrows = nrow_h + nrow_d + nrow_f_0
        ncols = max(ncols_h, ncols_d, ncols_f_0)
        
        # Возможно упростить выражение
        optimize_param_name = custom_converter_tostr(initial_data_set,
                                                     is_params)
        
        create_subplots(self, 
                       focus_dist,
                       borders,
                       list_convert=common_list,
                       rows=nrows,
                       cols=ncols,
                       optimize_param_name=optimize_param_name,
                       is_params=is_params,
                       ver2="_ver_2_0")

        



        
        