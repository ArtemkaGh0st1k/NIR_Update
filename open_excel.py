from openpyxl import load_workbook

def open_DataSet():
    
    workbook = load_workbook("dataset.xlsx")

    sheet = workbook.active
    
    local_lower_lambda = float(sheet[1][1].value * 1e-9)
    local_upper_lambda = float(sheet[2][1].value * 1e-9)

    local_count_linse = int(sheet[4][1].value)
    
    iterator_row = 6

    local_list_distanceBetweenLinse = list()
    for i in range(1,local_count_linse):
        local_list_distanceBetweenLinse.append(( float(sheet[iterator_row + i][0].value) * 1e-3))
        

    iterator_row += (local_count_linse-1) + 2

    local_numLinse_dataset = dict()
    for i in range(1, local_count_linse + 1):
        local_numLinse_dataset[i] = [int(sheet[iterator_row+i][1].value), 
                               float(sheet[iterator_row+i][2].value),
                               float(sheet[iterator_row+i][3].value) * 1e-9,
                               float(sheet[iterator_row+i][4].value) * 1e-3]
    
    return [local_count_linse,
            local_lower_lambda,
            local_upper_lambda,
            local_list_distanceBetweenLinse,
            local_numLinse_dataset]