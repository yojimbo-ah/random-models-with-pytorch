import math

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



# smallest neureull network in the world

# inputs 
x1 = Value(2.0)
x2 = Value(0.0)

#weights these are kind of parameters of the network
w1 = Value(-3.0)
w2 = Value(1.0)
# bias the other kind of paramter of neural network
b = Value(6.8813735870195432)
x1w1 = x1 * w1
x2w2 = x2 * w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
# o = n.tanh()
o = ((2*n).exp() - 1) / ((2*n).exp() + 1)
#using calculs chain rule to calculate them 
o.backward()
print(n.grad)