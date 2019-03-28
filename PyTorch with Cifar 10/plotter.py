import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import os

#File to be copied
__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
wb = openpyxl.load_workbook(os.path.join(__location__, "VGG_data.xlsx")) #Add file name
sheet = wb["Sheet1"] #Add Sheet name

def copyRange(startCol, startRow, endCol, endRow, sheet):
    rangeSelected = []
    #Loops through selected Rows
    for i in range(startRow,endRow + 1,1):
        #Appends the row to a RowSelected list
        rowSelected = []
        for j in range(startCol,endCol+1,1):
            rowSelected.append(sheet.cell(row = i, column = j).value)
        #Adds the RowSelected List and nests inside the rangeSelected
        rangeSelected.append(rowSelected)

    return np.transpose(np.asarray(rangeSelected))

x = list(range(10))
np.asarray(x)

tab_value = 0 # or 37

plt.figure(1)
plt.subplot(2,3,1)
row_start = 36 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv1 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,2)
row_start = 30 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv2 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,3)
row_start = 23 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv3 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,4)
row_start = 54 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv4 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,5)
row_start = 48 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv5 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,6)
row_start = 42 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("FC Layer Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

##### LEARNING RATE = 0.1

tab_value = 37
plt.figure(2)
plt.subplot(2,3,1)
row_start = 36 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv1 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,2)
row_start = 30 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv2 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,3)
row_start = 23 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv3 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,4)
row_start = 54 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv4 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,5)
row_start = 48 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("Conv5 Layers Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")

plt.subplot(2,3,6)
row_start = 42 + tab_value
Conv3_1 = copyRange(2, row_start,11,row_start, sheet)
Conv3_2 = copyRange(2, row_start+1,11,row_start+1, sheet)
Conv3_3 = copyRange(2, row_start+2,11,row_start+2, sheet)
Conv3_4 = copyRange(2, row_start+3,11,row_start+3, sheet)
plt.plot(x, Conv3_1, '-*', label='1bit')
plt.plot(x, Conv3_2, '-*', label='2bit')
plt.plot(x, Conv3_3, '-*', label='3bit')
plt.plot(x, Conv3_4, '-*', label='4bit')
plt.title("FC Layer Quantization")
plt.xlabel("# Epochs")
plt.ylabel("% Accuracy")
plt.legend(loc="lower right")


plt.show()
