

import numpy as np
import matplotlib.pyplot as plt

TestSetImageFile = open("/Users/moritzgruss/Desktop/Neural net/TrainingData/t10k-images-idx3-ubyte","rb")
TestSetLabels = open("/Users/moritzgruss/Desktop/Neural net/TrainingData/t10k-labels-idx1-ubyte","rb")

TrainingSetImageFile = open("/Users/moritzgruss/Desktop/Neural net/TrainingData/train-images-idx3-ubyte","rb")
TrainingSetLabels = open("/Users/moritzgruss/Desktop/Neural net/TrainingData/train-labels-idx1-ubyte","rb")
#we have 28 rows and 28 columns, 28 * 28 = 784
#function returns pixelmatrix for n'th digit in set
def getPixelMatrix(n,set):
    data = set.read()[(16+784*n):(800+784*n)]
    vector = []
    z = 0
    for x in range(28):
        row = []
        for y in range(28):
            row.append(data[z])
            z = z+1
        vector.append(row)
    return vector

#function extracts the label of the image
def getLabel(n,set):
    data = set.read()[(8+n):(9+n)]
    return int.from_bytes(data, byteorder='big')

#takes index of training example from MNIST and displays
#while also printing label
def outputTrainingExam(n):      
    PixelMatrix = getPixelMatrix(n,TrainingSetImageFile)
    print(getLabel(n,TrainingSetLabels))
    plt.imshow(PixelMatrix, cmap='gray')
    plt.colorbar()
    plt.show()