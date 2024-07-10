
import numpy as np
import networkFrame as frame

class ForewardProp:

    #calculates the weighted sum of current layer based on previous
    #activations and current weights and biases
    #prev is an activation vector
    #current is the layer object of current layer
    def propagate(self,prev,current):
        return np.add(np.matmul(current.weight,prev),current.bias)

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))
    
    def forwardprop(self,network):
        for l in range(1,len(network)):
            network[l].sum = self.propagate(network[l-1].activations,network[l])

class layertest:
    def __init__(self,prevsize,size):
        self.bias = np.array([4,5])
        self.weights = np.array([[1,2,3],[1,2,3]])
        self.sum = np.empty(size, dtype=float)
        self.activations = np.empty(size, dtype=float)

prop = ForewardProp
layer1 = frame.layer()
currentlayer = test(np.array([[1,2,3],[1,2,3]]),np.array([4,5]))
prevlayer = np.array([1,2,3])
print(prop.propagate(prop,prevlayer,currentlayer))