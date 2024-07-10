import numpy as np


class layer:
    def __init__(self,prevsize,size):
        self.bias = np.random.rand(size)
        #self.weights = np.random.rand(size, prevsize)

    

net = np.empty(3,dtype=layer)
net[0] = layer(0,3)


print(net[0].bias)