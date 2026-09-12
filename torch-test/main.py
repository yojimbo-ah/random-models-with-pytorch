import torch
import math
import random

class Value:
    def __init__(self , data , _children=() , _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0
        self._backward = lambda : None

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data + other.data , (self , other) , '+')
        # defining the method for backward prop using chain rule
        def _backward():
            self.grad += out.grad * 1.0
            other.grad += out.grad * 1.0

        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (- other)
    
    def __mul__(self, other):
        other = other if isinstance(other , Value) else Value(other)
        out = Value(self.data * other.data , (self,  other), '*')
        # simaliar to add using chain rule
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other,(int,float)) , "only supports integers and floats"
        # assuming that we pass other as constant not Value object
        out = Value(self.data ** other ,(self ,) , 'pow')
        def _backward():
            # local deravative of the powe function 
            self.grad += other * self.data ** (other - 1) * out.grad
        out._backward = _backward
        return out
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x) , (self , ) , 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        # usese the power function as base for it function beceasese div is **-1
        return self * (other ** -1)
    
    def tanh(self):
        n = self.data
        t = ((math.exp(2*n) - 1) / (math.exp(2*n) + 1))
        out = Value (t , (self , ) , 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)     
        self.grad = 1
        build_topo(self)
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self , nin):
        self.w = [Value(random.uniform(-1 , 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1 , 1))

    def __call__(self,x):
        # zip pairs first with first , second with second ...etc
        # we are pairing the weight with the inputs
        
        act = sum(wi*xi for wi , xi in zip(self.w , x)) + self.b
        out = act.tanh()
        return out       

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self,nin,nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self,x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

class MLP:
    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i] , sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        return [p for layer in self.layers for p in layer.parameters()]
    
n = MLP(3 , [ 4 , 4 , 1])

xs = [
    [2.0 , 3.0 , -1.0] ,
    [3.0 , -1.0 , 0.5] ,
    [0.5 , 1.0 , 1.0 ] ,
    [1.0 , 1.0 , -1.0] 
] # training data set

ys = [1.0 , -1.0 , -1.0 , 1.0] # desired output

for step in range(20):
    # ---- forward pass ----
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt)**2 for ygt, yout in zip(ys, ypred))

    # ---- backward pass ----
    for p in n.parameters():
        p.grad = 0.0        # IMPORTANT: zero gradients before backward()
    loss.backward()

    # ---- update (gradient descent step) ----
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print(step, loss.data)

ypred = [n(x) for x in xs]
print(ypred)
print(n.parameters())
