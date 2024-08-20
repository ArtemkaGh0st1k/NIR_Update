from typing import Callable, Any
from operator import matmul
from functools import reduce
import time 
import warnings

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, expon

from Data.InputTestDate import validate_data_set, DATA_SET_0
from MainAlghorithm import calculate_length_focal_distance

def test_smth():
    # load the ...
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target


    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    param_grid = {
        'C' : [0.1, 1, 10],
        'kernel' : ['linear', 'rbf', 'poly'],
        'gamma' : [0.1, 1, 'scale', 'auto'],
    }

    svm = SVC()

    grid_search = GridSearchCV(
        estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1
    )

    grid_search.fit(X_train, Y_train)

    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Accuracy Score: {:.2f}%".format(grid_search.best_score_ * 100))
    
    # Evaluate the model on the test set
    best_svm = grid_search.best_estimator_
    test_accuracy = best_svm.score(X_test, Y_test)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))


def loss_function(data_set_0: dict,
                  heights_optimize : np.ndarray) -> float:
    
    
    validate_data_set(data_set_0)

    lambda_massive = np.linspace(data_set_0['lower_lambda'] * 1e9,
                                data_set_0['upper_lambda'] * 1e9,
                                1000) # без [нм]
    

    focus_lambda_dict = {}
    for lmbd in lambda_massive:
        Matrix_Mults_List = []

        for num_linse in range(1, data_set_0['count_linse'] + 1):
            
            remove_degree_to_lambda_0 = (data_set_0['lambda_0'][num_linse]) * 1e9 # [нм -> 10^(-9)]
            focus_0 = data_set_0['focus_0'][num_linse]

            height_optimize = heights_optimize[num_linse-1]
            refractive_index = data_set_0['refractive_index'][num_linse]

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
            
            if num_linse != data_set_0['count_linse']:

                refractive_area = data_set_0['refractive_area']['{}-{}'.format(num_linse, num_linse + 1)]
                dist = data_set_0['distance']['{}-{}'.format(num_linse, num_linse + 1)]

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


# Образец численного градиента 
# Т.к найти производную явно тяжело
# loss_func - это функция рассчёта фок.отрезка
def numerical_gradient(heights : np.ndarray, 
                        loss_func = loss_function, step_h=0.1, eps=1e-8): # eps
    gradient = np.zeros_like(heights)
    for i in range(len(heights)):
        old_heights = heights.copy()
        new_heights = old_heights.copy()
        new_heights[i] += step_h
        next_func = loss_func(DATA_SET_0, new_heights)
        prev_func = loss_func(DATA_SET_0, heights)
        gradient[i] = (loss_func(DATA_SET_0, new_heights) - loss_func(DATA_SET_0, heights)) / eps # ~10^2 порядок
    return gradient 


# Реализация градиентого спуска
def gradient_descent(initial_heights : np.ndarray, 
                     learning_rate=0.01, max_iter=1000, tol=1e-4): # tol и learnin_rate
    start_time = time.time()
    heights = initial_heights
    for _ in range(max_iter):
        gradient = numerical_gradient(heights)
        if gradient.any() < 0: print("Есть < 0")
        new_heights = heights - learning_rate * gradient
        if np.linalg.norm(new_heights - heights) < tol:
            break
        heights = new_heights
    end_time = time.time()
    time_spent = end_time - start_time
    return heights, time_spent


