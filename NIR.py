from Data.InputTestDate import DATA_SET_0
from Algorithms.OptimizationParams.Height.MainAlgorithm import Calc


if __name__ == '__main__':

    calc = Calc()
    calc.visualization_gradient_descent(initial_heights=[8., 8.])
    print(f"Найденные высоты равны = {calc.get_h}")    
