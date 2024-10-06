import os
from os.path import (join, isdir)
import time
from colorama import Fore

import matplotlib.pyplot as plt
import numpy as np

from Data.InputTestDate import DATA_SET_0
from Utils.Path import create_dir


def create_subplots(self,
                    focus_dist : float,
                    borders : dict,
                    list_convert : list[ dict[str, list] ],
                    rows : int,
                    cols : int,
                    optimize_param_name : str,
                    is_params : dict[str, bool],
                    ver2 : str ="") -> None:
            global rows_copy
            global cols_copy
            rows_copy = rows
            cols_copy = cols

            iter_massive = list(range(1, self._MAX_ITER + 1))
            xlabel = "Итерация"
            
            fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10 + 1.5 * rows_copy, 10 + 1.5 * cols_copy))

            fig.subplots_adjust(wspace=0.4, hspace=0.4)
            fig.legend(loc='upper center',
                      bbox_to_anchor=(0.5, 1.0),
                      title=f"Фокальный отрезок = {focus_dist * 100} см")
            fig.legend(labels=\
                       [
                          'step_h={}'.format(self._STEP['h']),
                          'step_d={}'.format(self._STEP['d']),
                          'step_f_0={}'.format(self._STEP['f_0']),
                          'lr_h={}'.format(self._LEARNING_RATE['d']),
                          'lr_d={}'.format(self._LEARNING_RATE['d']),
                          'lr_f_0={}'.format(self._LEARNING_RATE['f_0']),
                          'denom_h={}'.format(self._DENOMINATOR['h']),
                          'denom_d={}'.format(self._DENOMINATOR['d']),
                          'denom_f_0={}'.format(self._DENOMINATOR['f_0'])
                       ], loc='upper left')
            
            if len(axs.shape) == 1:
              for i in range(1, axs.shape[0] + 1):
                selected_list = list_convert[0]

                _key = None
                for key in selected_list:
                     match key:
                          case 'h':
                            _key = key
                            break
                          case 'd':
                            _key = key
                            break
                          case 'f_0':
                            _key = key
                            break
                               
                harmonica = DATA_SET_0['harmonica'][i]
                ref_index = DATA_SET_0['refractive_index'][i]

                #FIXME: Пофиксить шаг (занимает дохуя места т.к мал величина с плав точкой)
                #FIXME: Учесть, что для разных параметров разные шаги
                #FIXME: Подрефакторить
                title = None
                ylabel = None
                if _key == 'h':
                  title = f"Град.спуск для {i}-ый высоты"
                  ylabel = f"Высота {i}-ой линзы, мкм"
                  label = 'harmonica={}\nref_index={}'.format(
                          harmonica,
                          ref_index
                          )
                  min_h_masssive = [ borders[i]['height'][0] * 1e6 ] * self._MAX_ITER
                  max_h_massive = [ borders[i]['height'][1] * 1e6] * self._MAX_ITER
                  axs[i-1].plot(iter_massive, min_h_masssive, color='green')
                  axs[i-1].plot(iter_massive, max_h_massive, color='red')
                  axs[i-1].plot(iter_massive + [iter_massive[-1] + 1], selected_list[_key][i-1], label=label)
                  axs[i-1].set_title(title)
                  axs[i-1].set_xlabel(xlabel)
                  axs[i-1].set_ylabel(ylabel)
                  axs[i-1].grid(True)
                  axs[i-1].legend(frameon=False, fontsize=7)
                   
                if _key == 'd':
                  title = f"Град.спуск для {i}-{i+1} раст-ий"
                  ylabel = f"Раст-ия м/у {i}-{i+1} линзой, см"
                  if i != DATA_SET_0['count_linse']:
                    axs[i-1].plot(iter_massive + [iter_massive[-1] + 1], selected_list[_key][i-1])
                    axs[i-1].set_title(title)
                    axs[i-1].set_xlabel(xlabel)
                    axs[i-1].set_ylabel(ylabel)
                    axs[i-1].grid(True)
                    axs[i-1].legend(frameon=False, fontsize=7)
                  else:
                    axs[i-1].remove() 
                  
                if _key == 'f_0':
                  title = f"Град.спуск для {i} фокуса"
                  ylabel = f"Фокус {i}-ой линзы, см"
                  label = 'harmonica={}\nref_index={}'.format(
                          harmonica,
                          ref_index
                          )
                  axs[i-1].plot(iter_massive + [iter_massive[-1] + 1], selected_list[_key][i-1], label=label)  
                  axs[i-1].set_title(title)
                  axs[i-1].set_xlabel(xlabel)
                  axs[i-1].set_ylabel(ylabel)
                  axs[i-1].grid(True)
                  axs[i-1].legend(frameon=False, fontsize=7)
                 
            else:
              for row in range(1, rows + 1):
                for i in range(1, cols + 1):  
                  selected_list = list_convert[row-1]

                  _key = None
                  for key in selected_list:
                      match key:
                            case 'h':
                              _key = key
                              break
                            case 'd':
                              _key = key
                              break
                            case 'f_0':
                              _key = key
                              break
                                
                  harmonica = DATA_SET_0['harmonica'][i]
                  ref_index = DATA_SET_0['refractive_index'][i]

                  #FIXME: Пофиксить шаг (занимает дохуя места т.к мал величина с плав точкой)
                  #FIXME: Учесть, что для разных параметров разные шаги
                  #FIXME: Подрефакторить
                  title = None
                  ylabel = None
                  if _key == 'h':
                    title = f"Град.спуск для {i}-ый высоты"
                    ylabel = f"Высота {i}-ой линзы, мкм"
                    label = 'harmonica={}\nref_index={}'.format(
                            harmonica,
                            ref_index
                            )
                    min_h_masssive = [ borders[i]['height'][0] * 1e6 ] * self._MAX_ITER
                    max_h_massive = [ borders[i]['height'][1] * 1e6] * self._MAX_ITER
                    axs[row - 1, i - 1].plot(iter_massive, min_h_masssive, color='green')
                    axs[row - 1, i - 1].plot(iter_massive, max_h_massive, color='red')
                    axs[row - 1, i - 1].plot(iter_massive + [iter_massive[-1] + 1], selected_list[_key][i-1], label=label)
                    axs[row - 1, i - 1].set_title(title)
                    axs[row - 1, i - 1].set_xlabel(xlabel)
                    axs[row - 1, i - 1].set_ylabel(ylabel)
                    axs[row - 1, i - 1].grid(True)
                    axs[row - 1, i - 1].legend(frameon=False, fontsize=7)
                    
                  if _key == 'd':
                    title = f"Град.спуск для {i}-{i+1} раст-ий"
                    ylabel = f"Раст-ия м/у {i}-{i+1} линзой, см"
                    if i != DATA_SET_0['count_linse']:
                      axs[row - 1, i - 1].plot(iter_massive + [iter_massive[-1] + 1], selected_list[_key][i-1])
                      axs[row - 1, i - 1].set_title(title)
                      axs[row - 1, i - 1].set_xlabel(xlabel)
                      axs[row - 1, i - 1].set_ylabel(ylabel)
                      axs[row - 1, i - 1].grid(True)
                      axs[row - 1, i - 1].legend(frameon=False, fontsize=7)
                    else:
                      axs[row - 1, i - 1].remove() 
                    
                  if _key == 'f_0':
                    title = f"Град.спуск для {i} фокуса"
                    ylabel = f"Фокус {i}-ой линзы, см"
                    label = 'harmonica={}\nref_index={}'.format(
                            harmonica,
                            ref_index
                            )
                    axs[row - 1, i - 1].plot(iter_massive + [iter_massive[-1] + 1], selected_list[_key][i-1], label=label)  
                    axs[row - 1, i - 1].set_title(title)
                    axs[row - 1, i - 1].set_xlabel(xlabel)
                    axs[row - 1, i - 1].set_ylabel(ylabel)
                    axs[row - 1, i - 1].grid(True)
                    axs[row - 1, i - 1].legend(frameon=False, fontsize=7)

            path = create_dir(is_params)                  
            fname = join(path, f"{optimize_param_name}{ver2}.png")
            fig.savefig(fname=fname)
            plt.close()