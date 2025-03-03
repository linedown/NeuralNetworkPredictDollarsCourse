import openpyxl
import numpy as np

def load_dataset(filename, count):
    wb = openpyxl.load_workbook(filename)

    sheet = wb['Лист1']
    
    inp = np.ndarray((count,3),dtype=np.float64)
    otp = np.ndarray((count,1),dtype=np.float64)

    for i in range(count):
        cellX = sheet['A'+str(i+2)].value
        cellY = sheet['B'+str(i+2)].value
        cellZ = sheet['C'+str(i+2)].value
        cellFunc = sheet['D'+str(i+2)].value

        inp[i] = np.array([cellX,cellY,cellZ])
        otp[i] = cellFunc
        
    return (inp,otp)
