import numpy as np
from tensor import Tensor

class Layer(object):

    def __init__(self):
        self.parameters = list()
    
    def get_parameters(self):
        return self.parameters

class Linear(Layer):

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2.0/(n_inputs))
        self.weight = Tensor(W, True)
        self.bias = Tensor(np.zeros(n_outputs), True)

        self.parameters.append(self.weight)
        self.parameters.append(self.bias)
    
    def forward(self, input):
        return input.mm(self.weight) + self.bias.expand(0, len(input.data))

    
class Sequential(Layer):
    def __init__(self, layers = list()):
        super().__init__()
        self.layers = layers
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input
    
    def get_parameters(self):
        params = list()
        for l in self.layers:
            params += l.get_parameters()
        return params
    
class Tanh(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.tanh()

class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.sigmoid()

class Relu(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        return input.relu()
    
class MSELoss(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return ((pred - target)*(pred - target)).sum(0)
