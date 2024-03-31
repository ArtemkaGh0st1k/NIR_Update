#from tkinter import *

'''Без excel файла

# начальные данные

count_linse = 4 # кол-во линз
m = 10 # гармоника 
n = 1.5 # показатель преломления
lambda0 = 550 * 1e-9 # базовая длина волны
h_relef = m * lambda0 / (n - 1) # высота микрорельефа
upper_lambda = 1000 * 1e-9 # верхння граница длины волны
lower_lambda = 400 * 1e-9 # нижняя граница длины волны
focus_0 = 200 * 1e-3 # начальный фокус
list_distanceBetweenLinse = [20 * 1e-3,
                             30 * 1e-3,
                             40 * 1e-3]
                              # расстояния м/у линзами,[см]
                             
dict_focus_labmda = dict()

local_lower_lambda = int(lower_lambda * 1e9)
local_upper_lambda = int(upper_lambda * 1e9)

for lmbd in range(local_lower_lambda, local_upper_lambda):
    list_MatrixMults = []                            
    
    k = round(m * lambda0 / (lmbd * 1e-9))
    optic_power = pow( ((m * lambda0) / (k * lmbd * 1e-9) * focus_0) , -1)
    first_Matrix = np.array( [ [1, 0], 
                               [-optic_power, 1] ] )
    
    list_MatrixMults.append(first_Matrix)

    if count_linse == 2:
        second_Matrix = np.array( [ [1, list_distanceBetweenLinse[0]],
                                    [0, 1] ] )
        list_MatrixMults.append(second_Matrix)
    else:
        for i in range(1, 2 * count_linse-1):   # до -1 означает, что последний матрицы переноса не будет!
            if i % 2 == 0:
                list_MatrixMults.append( [ [1, 0], 
                                           [-optic_power, 1] ] )
            if i % 2 == 1:
                list_MatrixMults.append( [ [1, list_distanceBetweenLinse[(i//2)] / n],
                                           [0, 1] ] )

    list_MatrixMults.reverse() # перевернул список, чтобы умножение было по правилу

    mult_res = np.matmul(list_MatrixMults[0], list_MatrixMults[1])

    if count_linse > 2:
        for j in range(2, len(list_MatrixMults)):
            mult_res = np.matmul(mult_res, list_MatrixMults[j])
        mult_res = np.matmul(mult_res, list_MatrixMults[-1])

    # в словарь делаем сцепку: длина волны(ключ) -> фокус расстояние(значение)
    dict_focus_labmda[lmbd * 1e-9] = pow( -mult_res[1,0], -1) # нужен ли здесь минус или нет???
'''

'''GUI 
window = Tk()
window.title("Научная работа")
window.geometry('500x350')

def comand_clear():
    text1.delete(0, END) #-> очистил поле ввода

def get_value():
    return text2.configure(text = 'Привет')

label1 = Label(window, text="Количество линз(больше 1):").grid(column=0,row=0)
text1 = Entry(window, width=10).grid(column=1,row=0)

button1 = Button(window, text="Подтвердить", command=get_value).grid(column=2,row=0)
button2 = Button(window, text="Отменить", command=comand_clear).grid(column=3,row=0)

label2 = Label(window, text="Нижняя граница длины волны, нм").grid(column=0,row=1)
text2 = Entry(window, width=10,show=get_value).grid(column=1,row=1)

label3 = Label(window, text="Верхняя граница длины волны, нм").grid(column=0,row=2)
text3 = Entry(window, width=10).grid(column=1,row=2)


window.mainloop()
'''

'''Попытка создания алгоритма, основанного на 
    поиске общей формулы для дальнешего применения
    этой формулы с использованием модуля SciPy и метода optimize
    
def get_commonFunction(count_linse: int):

    checkZeroMult = lambda elem1, elem2: True if (elem1 == '0' or elem2 == '0') else False

    R_Matrix = np.array( [ [str(1), str(0)],
                                    [str(f'-F_{count_linse}'), str(1)] ] )
    if i != count_linse:
        T_Matrix = np.array( [ [str(1), str(f'd_{i}{i+1}'/f'n_{i}')],
                                      [str(0), str(1)] ] )

    MatrixG: np.ndarray[str] = np.array([ [ f'({R_Matrix[0][0]}*{T_Matrix[1][0]})+({R_Matrix[0][1]+T_Matrix[1][1]})',
                                            f'()'] ], dtype=str)
    
    MatrixG = np.array( [ [f'({R_Matrix[0][0]}*{T_Matrix[1][0]})' if checkZeroMult(R_Matrix[0][0], T_Matrix[1][1]) else '',] ] )

    MatrixG

    for i in range(1, count_linse+1):
        Refractive_Matrix = np.array( [ [str(1), str(0)],
                                        [str(f'-F_{count_linse}'), str(1)] ] )
        if i != count_linse:
            Transfer_Matrix = np.array( [ [str(1), str(f'd_{i}{i+1}'/f'n_{i}')],
                                          [str(0), str(1)] ] )
            
        

    pass

def getElem(row: int, column: int) -> str:
    CheckZeroMult: bool = lambda elem1, elem2, row1, column1, row2, column2: True if (elem1[row1][column1] == '0' or elem2[row2][column2] == '0') else False
    DoOperation: str = lambda checkZeroMult, elem1, elem2: (('0' if checkZeroMult(elem1,elem2,row1=row,column1=0,row2=0,column2=column) else '{}*{}'.format(elem1[row][0], elem2[0][column])) +  
                                                      ('0' if checkZeroMult(elem1,elem2,row1=row,column1=1,row2=1,column2=column) else '+ {}*{}'.format(elem1[row][1], elem2[1][column])))

    return DoOperation()

'''

''' Явное перемножение матриц
mult_res_reduce = reduce(matmul, MatrixMultsList)

        mult_res = MatrixMultsList[0] @ MatrixMultsList[1]
        for j in range(2, len(MatrixMultsList)):
            mult_res = mult_res @ MatrixMultsList[j]
'''