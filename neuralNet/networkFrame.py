import numpy as np

class net:
    def __init__(self,layers):
        # creates an array of layer objects
        self.net = np.empty(len(layers),dtype=layer)
        for x in range(len(layers)):
            self.net[x] = layer(layers[x-1],layers[x])
        
    def getBias(self,layer):
        if layer == 0:
            print("input layer contains no Bias")
        else:
            print(self.net[layer].bias)
    
    def getWeights(self,layer):
        if layer == 0:
            print("input layer contains no Weights")
        else:
            print(self.net[layer].weights)
    
class layer:
    def __init__(self,prevsize,size):
        self.bias = np.random.rand(size)
        self.weights = np.random.rand(size, prevsize)
        self.sum = np.empty(size, dtype=float)
        self.activations = np.empty(size, dtype=float)

list = net([3,3,2])
#print(list.net[0].sum)



