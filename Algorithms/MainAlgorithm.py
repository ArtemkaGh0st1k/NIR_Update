"""
Основной модуль расчёта
"""

from operator import matmul
from functools import reduce
import time 
import os
from colorama import Fore
from typing import TypeVar

import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from Data.InputTestDate import validate_data_set, DATA_SET_0
from Utils.Converter import custom_converter_tostr
from Utils.Plot import create_subplots


SelfCalc = TypeVar("SelfCalc", bound="Calc")
class Calc():
    def __init__(self : SelfCalc, 
                 max_iter=1000,
                 lr : dict[str, float] = { 'h':0.01, 'd':0.01, 'f_0':0.01 },
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
        self._INITIAL_DATA_SET = None

        self._time_grad_desc = 0
        self._h = None
        self._d = None
        self._f_0 = None

    
    @property
    def get_h(self : SelfCalc) -> np.ndarray[float, 1]:
        '''
        Возвращает найденные высоты для каждой линзы в виде списка
        '''
        return self._h
    
    @property
    def get_d(self : SelfCalc) -> np.ndarray[float, 1]:
        '''
        Возвращает найденные расстояния для каждой пары линз в виде списка
        '''
        return self._d
    
    @property
    def get_f_0(self : SelfCalc) -> np.ndarray[float, 1]:
        '''
        Возвращает найденные базовые фокусы для каждой линзы в виде списка
        '''
        return self._f_0
    
    @property
    def get_time_grad_desc(self : SelfCalc) -> tuple[int, float]:
        '''
        Возвращает время, затраченное на алгоритм град.спуска
        в минутах и секундах
        '''
        return (self._time_grad_desc // 60, self._time_grad_desc % 60)
    

    def __collect_optimize_params(self : SelfCalc) -> dict[str, bool]:
            '''
            Description:
            ------------
            Собирает информацию о том, какие параметры будут
            учавствовать в алгоритме

            Return:
            -------
            Возвращает словарь в виде:\n
            key - имя параметра\n
            value - bool параметр, если он учавствует в оптимизации

            :Params:\n
            `initial_data_set`: Исходные данные в виде словаря
            '''
            
            is_h = False
            is_distance = False
            is_focus_0 = False

            for key in self._INITIAL_DATA_SET:
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


    def loss_function(self : SelfCalc,
                      initial_data_set: dict,
                      save_plot = False) -> float:
        
        '''
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
        `initial_data_set`: Исходные данные в виде словаря\n
        `save_plot`: Сохранить ли график
        '''
        if self._INITIAL_DATA_SET is None: self._INITIAL_DATA_SET = initial_data_set.copy()
        is_params = self.__collect_optimize_params()

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
                
                height = (initial_data_set['height'][i] * 1e-6
                          if is_params['h']
                          else 
                          (DATA_SET_0['harmonica'][i] * DATA_SET_0['lambda_0'][i] / (DATA_SET_0['refractive_index'][i] - 1)))
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
                'Высоты равны = {}'.format(list(initial_data_set['height'].values())) if is_params['h'] else "",
                'Базовые фокусы равны = {}'.format(list(initial_data_set['focus_0'].values())) if is_params['f_0'] else "",
                'Расстояния м/у линзами равны = {}'.format(list(initial_data_set['distance'].values())) if is_params['d'] else "",
                sep=' | ')
        
        return length_focus_distance
    

    def __collect_min_max_params_range(self : SelfCalc) -> dict:

        '''
        Description:
        ------------
        Возвращяет минимальные и максимальные высоты каждой линзы,
        пригодные для алгоритма оптимизации в [мкм]  

        Return:
        -------
        Возвращает словарь, в котором ключ - номер линзы,\n
        а значение - кортеж мин и макс допустимый высоты для данной линзы

        :Params:\n
        `initial_data_set`: Исходные данные в виде словаря
        '''

        is_params = self.__collect_optimize_params()

        lower_lambda = DATA_SET_0['lower_lambda']
        upper_lambda = DATA_SET_0['upper_lambda']
        if is_params['d'] and is_params['f_0']: 
            max_distance = max(self._INITIAL_DATA_SET['distance'].values())

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
                    min_focus_0 = max(max_distance, self._INITIAL_DATA_SET['focus_0'][i])
                    collect_data[i] = \
                    {
                        'height' : (min_h, max_h),
                        'focus_0': min_focus_0
                    }   

        return collect_data 
    

    def __numerical_gradient(self : SelfCalc, initial_data_set : dict) -> tuple[np.ndarray[float],
                                                                                np.ndarray[float] | None,
                                                                                np.ndarray[float] | None]:
        '''
        Description:
        ---------------
        Используется численный градиент, т.к целевая функция имеет неявный вид

        Return:
        ----------
        Возвращяет численный градиент, в виде кортежа для каждого параметра оптимизации
        
        :Params:\n
        `initial_data_set`: Исходные данные в виде словаря
        '''

        initial_data_set_copy = initial_data_set.copy()
        is_params = self.__collect_optimize_params()
        
        grad_h = (np.zeros_like(list(initial_data_set_copy['height'].values())).astype('float64')
                  if is_params['h']
                  else None)
        grad_d = ((np.zeros_like(list(initial_data_set_copy['distance'].values())).astype('float64')) 
                    if is_params['d']
                    else None)
        grad_f_0 = ((np.zeros_like(list(initial_data_set_copy['focus_0'].values())).astype('float64'))
                    if is_params['f_0']
                    else None)

        for i in range(1, DATA_SET_0['count_linse'] + 1):
            if is_params['h']:
                orig_val_h = initial_data_set_copy['height'][i]

                initial_data_set_copy['height'][i] = orig_val_h + self._STEP['h']
                loss_func_plus_h = self.loss_function(initial_data_set_copy)

                initial_data_set_copy['height'][i] = orig_val_h
                loss_func_orig_h = self.loss_function(initial_data_set_copy)
                        
                grad_h[i-1] = (loss_func_plus_h - loss_func_orig_h) / self._DENOMINATOR['h']

                initial_data_set_copy['height'][i] = orig_val_h

            if is_params['d'] and i != DATA_SET_0['count_linse']:
                orig_val_d = initial_data_set_copy['distance'][f'{i}-{i+1}'] 

                initial_data_set_copy['distance'][f'{i}-{i+1}'] = orig_val_d + self._STEP['d']
                loss_func_plus_d = self.loss_function(initial_data_set_copy)

                initial_data_set_copy['distance'][f'{i}-{i+1}'] = orig_val_d
                loss_func_orig_d = self.loss_function(initial_data_set_copy)
                        
                grad_d[i-1] = (loss_func_plus_d - loss_func_orig_d) / self._DENOMINATOR['d']

                initial_data_set_copy['distance'][f'{i}-{i+1}'] = orig_val_d
            
            if is_params['f_0']:
                orig_val_f_0 = initial_data_set_copy['focus_0'][i]

                initial_data_set_copy['focus_0'][i] = orig_val_f_0 + self._STEP['f_0']
                loss_func_plus_f_0 = self.loss_function(initial_data_set_copy)

                initial_data_set_copy['focus_0'][i] = orig_val_f_0
                loss_func_orig_f_0 = self.loss_function(initial_data_set_copy)
                        
                grad_f_0[i-1] = (loss_func_plus_f_0 - loss_func_orig_f_0) / self._DENOMINATOR['f_0']

                initial_data_set_copy['focus_0'][i] = orig_val_f_0

        return (grad_h, grad_d, grad_f_0)
        
    
    def __gradient_descent(self : SelfCalc, initial_data_set : dict) -> tuple[ list[list[float]] | None,
                                                                               list[list[float]] | None,
                                                                               list[list[float]] | None]: 
        
        '''
        Description:
        ------------
        Оснвоной метод градиентного спуска

        Return:
        -------
        Возвращает кортеж из списка всех значений, которые были найдены
        в ходе алгоритма для каждого параметра оптимизацииы

        :Params:\n
        `initial_data_set`: Исходные данные в виде словаря
        '''

        is_params = self.__collect_optimize_params()
        collect_data = self.__collect_min_max_params_range()
        for _, key in enumerate(initial_data_set, start=1):
            for j in range(1, DATA_SET_0['count_linse']  + 1):
                match key:
                    case 'height':
                        min_h = collect_data[j][key][0] * 1e6
                        max_h = collect_data[j][key][1] * 1e6

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
            grad_d = 0
        if is_params['f_0']:
            list_f0 = [[initial_data_set_copy['focus_0'][i]] for i in range(1, DATA_SET_0['count_linse'] + 1)]
            f0 = list(initial_data_set_copy['focus_0'].values())
            grad_f0 = 0

        start_time = time.time()
        for _ in range(self._MAX_ITER):
        
            if is_params['h']:

                grad = self.__numerical_gradient(initial_data_set_copy)
                grad_h = grad[0]   
                h -= self._LEARNING_RATE['h'] * grad_h
                [list_h[i].append(h[i]) for i in range(len(h))]

                initial_data_set_copy['height'] = \
                {
                    key : value
                    for key, value
                    in zip(list(range(1, len(h) + 1)), h)
                }

            if is_params['d']:
                grad = self.__numerical_gradient(initial_data_set_copy)
                grad_d = grad[1]
                d -= self._LEARNING_RATE['d'] * grad_d
                [list_d[i].append(d[i]) for i in range(len(d))]

                initial_data_set_copy['distance'] = \
                {
                    '{}-{}'.format(key, key + 1) : value
                    for key, value 
                    in zip(list(range(1, len(d) + 1)) , d)
                }

            if is_params['f_0']:
                grad = self.__numerical_gradient(initial_data_set_copy)
                grad_f0 = grad[2]
                f0 -= self._LEARNING_RATE['f_0'] * grad_f0
                [list_f0[i].append(f0[i]) for i in range(len(f0))]

                initial_data_set_copy['focus_0'] = \
                {
                    key: value 
                    for key, value
                    in zip(list(range(1, len(f0) + 1)), f0)
                }
        
        time_grad_desc = time.time() - start_time

        self._h = [float(h[-1]) for h in list_h] if list_h else list_h
        self._d = [float(d[-1]) for d in list_d] if list_d else list_d
        self._f_0 = [float(f0[-1]) for f0 in list_f0] if list_f0 else list_f0

        print(Fore.YELLOW, "Время работы град.спуска =",
              Fore.RED, "{} мин".format(time_grad_desc // 60),
              Fore.GREEN, "{} сек".format(time_grad_desc % 60),
              Fore.WHITE)

        return (list_h, list_d, list_f0)
        

    def visualization_gradient_descent(self : SelfCalc, 
                                       initial_data_set: dict):
        
        '''
        Description:
        ------------
        Основной алгоритм для вывода и визуализации градиентного спуска

        :Params:\n
        `initial_data_set`: Исходные данные в виде словаря (все подаётся не в СИ!!!)\n
        '''
        if self._INITIAL_DATA_SET is None: self._INITIAL_DATA_SET = initial_data_set.copy() 
        count_linse = DATA_SET_0['count_linse']
        common_list = []
        
        is_params = self.__collect_optimize_params() # Проверка: какие параметры будем оптимизировать
        borders = self.__collect_min_max_params_range() # Рассчитываем допустимый диапазон параметров
                                                                      # #TODO: Нужно будет подкорректировать, учесть что h - необязат.парам 
        optimize_params = self.__gradient_descent(initial_data_set) # Получаем результат и важно, 
                                                                      # что initial_params должен поменяться,
                                                                      # с учётом новых параметров
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

        global nrows
        global ncols
        # Постройка графиков
        nrows = nrow_h + nrow_d + nrow_f_0
        ncols = max(ncols_h, ncols_d, ncols_f_0)

        
        # Возможно упростить выражение
        optimize_param_name = custom_converter_tostr(initial_data_set,
                                                     is_params)
        
        create_subplots(self,
                       borders,
                       list_convert=common_list,
                       rows=nrows,
                       cols=ncols,
                       optimize_param_name=optimize_param_name,
                       is_params=is_params)

        



        
        