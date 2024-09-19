
from operator import matmul
from functools import reduce
import time 
import os
from colorama import Fore

import numpy as np
import matplotlib.pyplot as plt
from accessify import private, protected

from Data.InputTestDate import validate_data_set, DATA_SET_0
from Utils.Converter import custom_converter_tolist


class Calc():
    def __init__(self, lr=0.01, step_h=0.1, max_iter=1000, denominator=1e-3) -> None:
        self._LEARNING_RATE = lr
        self._STEP_H = step_h
        self._MAX_ITER = max_iter
        self._DENOMINATOR = denominator

        self._time_grad_desc = None
        self._h = None

    
    @property
    def get_h(self) -> np.ndarray[float]:
        return self._h
    
    @property
    def get_time_grad_desc(self) -> tuple[int, float]:
        '''
        Возвращает время, затраченное на алгоритм град.спуска
        в минутах и секундах
        '''
        return (self._time_grad_desc // 60, self._time_grad_desc % 60)
    

    @staticmethod
    def calculate_length_focal_distance(data_set_0: dict,
                                        height_optimize_list: list[float] = None,
                                        save_plot = False) -> float:
    
        '''
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
        '''
        
        validate_data_set(data_set_0)

        lambda_massive = np.linspace(data_set_0['lower_lambda'] * 1e9,
                                    data_set_0['upper_lambda'] * 1e9,
                                    1000)
        
        focus_lambda_dict = {}
        for lmbd in lambda_massive:
            Matrix_Mults_List = []
            for i in range(1, data_set_0['count_linse'] + 1):

                remove_degree_to_lambda_0 = (data_set_0['lambda_0'][i]) * 1e9 # [нм -> 10^(-9)]
                focus_0 = data_set_0['focus_0'][i]

                if height_optimize_list is None:
                    harmonica = data_set_0['harmonica'][i]

                    k = round((remove_degree_to_lambda_0 / (lmbd)) * harmonica) #FIXME: Правильно ли округляю число k?
                    focus = ((harmonica * remove_degree_to_lambda_0) / (k * lmbd)) * focus_0
                else:
                    height_optimize = height_optimize_list[i-1]
                    refractive_index = data_set_0['refractive_index'][i]

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
                
                if i != data_set_0['count_linse']:

                    refractive_area = data_set_0['refractive_area']['{}-{}'.format(i, i + 1)]
                    dist = data_set_0['distance']['{}-{}'.format(i, i + 1)]

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


    def __loss_function(self, data_set_0: dict,
                        heights_optimize : list[float] | np.ndarray) -> float:
        
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
        `data_set_0`: Исходные данные в виде словаря\n
        `heights_optimize`: Массив высот (приходит не в [мкм]!!!)
        '''

        validate_data_set(data_set_0)

        lambda_massive = np.linspace(data_set_0['lower_lambda'],
                                    data_set_0['upper_lambda'],
                                    1000) 
        
        focus_lambda_dict = {}
        for lmbd in lambda_massive:
            Matrix_Mults_List = []

            for i in range(1, data_set_0['count_linse'] + 1):
                
                harmonica = data_set_0['harmonica'][i]
                focus_0 = data_set_0['focus_0'][i]
                height_optimize = heights_optimize[i-1] * 1e-6
                refractive_index = data_set_0['refractive_index'][i]

                lambda_0 = height_optimize * (refractive_index - 1) / harmonica            
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
                
                if i != data_set_0['count_linse']:

                    refractive_area = data_set_0['refractive_area']['{}-{}'.format(i, i + 1)]
                    dist = data_set_0['distance']['{}-{}'.format(i, i + 1)]

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
        
        return length_focus_distance
    

    def __collect_min_max_h_range(self, data_set_0: dict) -> dict[int, tuple[float, float]]:

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
        ----------
        `data_set_0`: Исходные данные в виде словаря
        '''

        validate_data_set(data_set_0)

        lower_lambda = data_set_0['lower_lambda']
        upper_lambda = data_set_0['upper_lambda']

        collect_data = {}
        for i in range(1, data_set_0['count_linse'] + 1):   

            harmonica = data_set_0['harmonica'][i]
            refractive_index = data_set_0['refractive_index'][i]

            min_h = harmonica / (refractive_index - 1) * lower_lambda * 1e6
            max_h = harmonica / (refractive_index - 1) * upper_lambda * 1e6

            collect_data[i] = (min_h, max_h)

        return collect_data 
    

    def __numerical_gradient(self, heights : np.ndarray | list) -> np.ndarray[float]:
        '''
        Description:
        ---------------
        Используется численный градиент, т.к целевая функция имеет неявный вид

        Return:
        ----------
        Возвращяет численный градиент

        :Params:\n
        `heights`: Параметр оптимизации (высоты)\n
        `loss_func`: Функция потерь (или целевая функция)\n
        '''
        
        grad = np.zeros_like(heights)
        for i in range(len(heights)):
            # Сохранение текущего значения параметра
            original_value = heights[i]

            # Вычесление f(h + step_h)
            heights[i] = original_value + self._STEP_H
            loss_func_plus_h = self.__loss_function(DATA_SET_0, heights)

            # Вычесление f(h)
            heights[i] = original_value
            loss_function_original = self.__loss_function(DATA_SET_0, heights)
                    
            # Градиент по i-координате
            grad[i] = (loss_func_plus_h - loss_function_original) / self._DENOMINATOR

            # Востановление параметра
            heights[i] = original_value

        return grad
    

    def __gradient_descent(self, initial_params : list[float] | np.ndarray) -> tuple[np.ndarray[float], list[float]]:
    
        '''
        Description:
        ------------
        Основной алгоритм градиентного спуска

        :Params:\n
        `initial_params`: Начальное приближение\n
        `loss_func`: Функция потерь (целевая функция)\n 
        `grad_func`: Градиент\n

        Return:
        -------
        Возвращяет кортеж из `списка найденных высот` и\n
        `список всех высот, которые вычислялись в алгоритме`
        '''
        
        # Проверка на выход из допустимого диапазона высоты
        collect_data = self.__collect_min_max_h_range(DATA_SET_0)
        for i, param in enumerate(initial_params, start=1):
            min_h = collect_data[i][0]
            max_h = collect_data[i][1]

            if (param < min_h):
                initial_params[i-1] = min_h
            elif (param > max_h):
                initial_params[i-1] = max_h

        h = None
        h_list = []

        h = np.array(initial_params)
        h_list.append(h.copy())

        start_time = time.time()
        for _ in range(self._MAX_ITER):

            grad = self.__numerical_gradient(h)

            h -= self._LEARNING_RATE * grad
            h_list.append(h.copy())
            
        time_grad_desc = time.time() - start_time

        self._h = h
        self._time_grad_desc = time_grad_desc
        print(Fore.BLUE, "Время работы град.спуска = {} м {} с".format(time_grad_desc // 60, time_grad_desc % 60))
        return h, h_list
    

    def visualization_gradient_descent(self, 
                                       initial_heights : list[float] | np.ndarray = None,
                                       fname = None):
        
        '''
        Description:
        ------------
        Основной алгоритм для вывода и визуализации градиентного спуска

        :Params:\n
        `initial_heights`: Начальное приближение для высот (подаётся в [мкм]!)
        '''

        if initial_heights is None: 
            initial_heights = []
            [initial_heights.append(1.) for _ in range(DATA_SET_0['count_linse'])]
        
        height_optimize_list, h = self.__gradient_descent(initial_heights)
        h_convert = custom_converter_tolist(size=DATA_SET_0['count_linse'], current_list=h)
        focus_dist = self.calculate_length_focal_distance(DATA_SET_0, height_optimize_list)
        collect_min_max_h = self.__collect_min_max_h_range(DATA_SET_0)

        optimize_param_name = "h_" + "_".join(str(initial_heights).
                                            replace("[", "").
                                            replace("]", "").
                                            replace(",", "").
                                            split() + \
                                            ['lr={}_step_h={}_max_iter={}_detorminator={}'.format(self._LEARNING_RATE,
                                                                                                  self._STEP_H,
                                                                                                  self._MAX_ITER,
                                                                                                  self._DENOMINATOR)])
        
        count_linse = DATA_SET_0['count_linse']
        figsize = (15, 7)
        if (count_linse == 4): figsize = (20, 7)
        elif (count_linse == 5): figsize = (25, 7)

        fig, axs = plt.subplots(nrows=1, ncols=len(initial_heights), figsize=figsize)
        plt.subplots_adjust(wspace=0.3)
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), title=f"Фокальный отрезок = {focus_dist * 100} см")
        
        iter_massive = list(range(1, self._MAX_ITER + 1))
        for i, ax in enumerate(axs, start=1):
            min_h_masssive = [ collect_min_max_h[i][0] ] * self._MAX_ITER
            max_h_massive = [ collect_min_max_h[i][1] ] * self._MAX_ITER

            harmonica = DATA_SET_0['harmonica'][i]
            ref_index = DATA_SET_0['refractive_index'][i]
            focus_0 = DATA_SET_0['focus_0'][i] 
            label = 'lr={}\nstep_h={}\nmax_iter={}\ndetorminator={}\n'.format(
                    self._LEARNING_RATE,
                    self._STEP_H,
                    self._MAX_ITER,
                    self._DENOMINATOR
                    ) + \
                    'harmonica={}\nref_index={}\nfocus_0={} см'.format(
                    harmonica,
                    ref_index,
                    focus_0 * 100
                    )

            ax.set_title(f"График град.спуска для {i}-ый высоты")
            ax.set_xlabel("Итерация")
            ax.set_ylabel(f"Высота {i}-ой линзы, мкм")
            ax.plot(h_convert[i-1], label=label)
            ax.plot(iter_massive, min_h_masssive, color='green')
            ax.plot(iter_massive, max_h_massive, color='red')
            ax.grid(True)
            ax.legend(frameon=False, fontsize=7)

        
        fname = os.path.join("NIR_Update", "Result", "GradientDescent", "NewAttempts", f"{count_linse}_Linse", f"{optimize_param_name}.png")
        fig.savefig(fname=fname)
        plt.close()



        
        