import numpy as np

class Tensor:

    def __init__(self, data, autograd = False, _children = (), creation_op = ''):
        self.data = np.array(data)
        self.grad = None
        self.childrens = set(_children)
        self.creation_op = creation_op
        self.autograd = autograd
        self._backward = lambda: None
        self.shape = self.data.shape
    
    def __repr__(self):
        return str(self.data.__repr__())
    
    def __str__(self):
        return str(self.data.__str__()) 

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        if(self.autograd):
            out = Tensor(self.data + other.data, True, (self, other), creation_op= "add")

            def _backward():
                if(self.grad is None):
                    self.grad = 1.0 * out.grad
                else:
                    self.grad += 1.0 * out.grad
                
                if(other.grad is None):
                    other.grad = 1.0 * out.grad
                else:
                    other.grad += 1.0 * out.grad

            out._backward = _backward
            return out
        else:
            return Tensor(self.data + other.data)
        
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        if(self.autograd):
            out = Tensor(self.data * other.data, True, (self, other), creation_op = "mul")

            def _backward():
                if(self.grad is None):
                    self.grad = other * out.grad
                else:
                    self.grad += other * out.grad

                if(other.grad is None):
                    other.grad = self * out.grad
                else:
                    other.grad += self * out.grad

            out._backward = _backward
            return out
        else:
            return Tensor(self.data * other.data)

    def __rmul__(self, other):
            return self * other
    
    def __truediv__(self, other): 
        return self * other**-1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __neg__(self):
        return self * -1
    
    def __radd__(self, other): 
        return self + other
    
    def exp(self):
        if self.autograd:
            x = self.data
            out = Tensor(np.exp(x), True ,(self,), 'exp')

            def _backward():
                if(self.grad is None):
                    self.grad = out * out.grad
                else:
                    self.grad += out * out.grad

            out._backward = _backward
            return out
        else :
            return Tensor(np.exp(x))
        
    def __pow__(self, other):
        if(self.autograd):
            out = Tensor(self.data ** other, True, (self, other), f'**{other}')

            def _backward():
                if (self.grad is None):
                    self.grad = other * (self ** (other - 1)) * out.grad
                else:
                    self.grad += other * (self ** (other - 1)) * out.grad

            out._backward = _backward
            return out 
        else:
            return Tensor(self.data ** other)
    
    
    def sigmoid(self):
        if(self.autograd):
            x = self.data
            t = 1 / (1 + np.exp(-x))
            out = Tensor(t, True, (self,), 'sigmoid')

            def _backward():
                if(self.grad is None):
                    self.grad = out.grad * (t * (1 - t))
                else:
                    self.grad += out.grad * (t * (1 - t))

            out._backward = _backward
            return out
        else:
            return Tensor(1 / (1 + np.exp(-self.data)))
        
    def tanh(self):
        if(self.autograd):
            x = self.data
            t = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
            out = Tensor(t, True, (self, ), 'tanh')
            
            def _backward():
                if(self.grad is None):
                    self.grad = out.grad * (1 - t**2)
                else:
                    self.grad += out.grad * (1 - t**2)
                    
            out._backward = _backward
            return out
        else:
            return Tensor(np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
    
    def relu(self):
        if(self.autograd):
            x = self.data
            t = np.maximum(x,0)
            out = Tensor(t, True, (self,), 'relu')

            def _backward():
                t1 = np.zeros_like(t)
                for i in range(len(t)):
                    if t[i] > 0:
                        t1[i] = 1
            
                if(self.grad is None):
                    self.grad = out.grad * t1
                else:
                    self.grad += out.grad * t1
            out._backward = _backward
            return out
        else:
            return Tensor(np.maximum(x,0))
    
    def expand(self, dim, copies):

        trans_cmd = list(range(0,len(self.shape)))
        trans_cmd.insert(dim,len(self.shape))
        new_data = self.data.repeat(copies).reshape(list(self.shape) + [copies]).transpose(trans_cmd)

        if(self.autograd):
            out = Tensor(new_data, True, (self,), 'expand')

            def _backward():
                if(self.grad is None):
                    self.grad = out.grad.sum(dim)
                else:
                    self.grad += out.grad.sum(dim)
            out._backward = _backward
            return out
        else:
            return Tensor(new_data)
                    
                
    def sum(self, dim):
        if(self.autograd):
            out = Tensor(self.data.sum(dim), True, (self,), 'sum')

            def _backward():
                if(self.grad is None):
                    self.grad = out.grad.expand(dim, self.shape[dim])
                else:
                    self.grad += out.grad.expand(dim, self.shape[dim])

            out._backward = _backward
            return out
        else:
            return Tensor(self.data.sum(dim))


    def mm(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        if (self.autograd):
            out = Tensor(self.data.dot(other.data), True, (self, other), 'mm')

            def _backward():
                if(self.grad is None):
                    self.grad = out.grad.mm(other.transpose())
                else:
                    self.grad += out.grad.mm(other.transpose())

                if(other.grad is None):
                    other.grad = self.transpose().mm(out.grad)
                else:
                    other.grad += self.transpose().mm(out.grad)

            out._backward = _backward
            return out
        else:
            return Tensor(self.data.dot(other.data))
        
    def transpose(self):
        if (self.autograd):
            out = Tensor(self.data.transpose(), True, (self,), 'transpose')

            def _backward():
                if(self.grad is None):
                    self.grad = out.grad.transpose()
                else:
                    self.grad += out.grad.transpose()

            out._backward = _backward
            return out
        else:
            return Tensor(self.data.transpose())

    
    def backward(self):

        nodes = []
        visited = set()
        def build_nodes(v):
            if v is not visited:
                visited.add(v)
                for child in v.childrens:
                    build_nodes(child)
                nodes.append(v)

        build_nodes(self)

        self.grad = Tensor(np.ones_like(self.data))
        for node in reversed(nodes):
            node._backward()
