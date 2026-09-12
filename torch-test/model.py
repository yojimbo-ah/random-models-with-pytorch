import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

class Linear:
  
  def __init__(self, fan_in, fan_out, bias=True):
    self.weight = torch.randn((fan_in, fan_out), generator=g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  
  def __call__(self, x):
    self.out = x @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])


class BatchNorm1d:
  
  def __init__(self, dim, eps=1e-5, momentum=0.1):
    self.eps = eps
    self.momentum = momentum
    self.training = True
    # parameters (trained with backprop)
    self.gamma = torch.ones(dim)
    self.beta = torch.zeros(dim)
    # buffers (trained with a running 'momentum update')
    self.running_mean = torch.zeros(dim)
    self.running_var = torch.ones(dim)
  
  def __call__(self, x):
    # calculate the forward pass
    if self.training:
      if x.ndim == 2:
        dim = 0
      elif x.ndim == 3:
        dim = (0,1)
      xmean = x.mean(dim, keepdim=True) # batch mean
      xvar = x.var(dim, keepdim=True) # batch variance
    else:
      xmean = self.running_mean
      xvar = self.running_var
    xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
    self.out = self.gamma * xhat + self.beta
    # update the buffers
    if self.training:
      with torch.no_grad():
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
    return self.out
  
  def parameters(self):
    return [self.gamma, self.beta]

class Tanh:
  def __call__(self, x):
    self.out = torch.tanh(x)
    return self.out
  def parameters(self):
    return []

class Embedding:
  def __init__(self, num_embedding , embedding_dim):
    self.weight = torch.randn([num_embedding , embedding_dim])
  #translates the charratcers to vector form
  def __call__(self, IX):
    self.out = self.weight[IX]
    return self.out
  
  def parameters(self):
    return [self.weight]

class Flatten:
  def __init__(self , n):
    self.n = n

  def __call__(self , x):
    B , T , C = x.shape
    x = x.view(B , T//self.n , C * self.n)
    if x.shape[1] == 1:
      x = x.squeeze(1)
    self.out = x
    return self.out
  
  def parameters(self):
    return []

class Sequential:
  def __init__(self , layers):
    self.layers = layers

  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
      self.out = x
    return self.out

  def parameters(self):
    return [ p for layer in  self.layers for p in layer.parameters()]


torch.manual_seed(42)

# build the dataset
block_size = 8 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  
  for w in words:
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      context = context[1:] + [ix] # crop and append

  X = torch.tensor(X)
  Y = torch.tensor(Y)
  print(X.shape, Y.shape)
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr,  Ytr  = build_dataset(words[:n1])     # 80%
Xdev, Ydev = build_dataset(words[n1:n2])   # 10%
Xte,  Yte  = build_dataset(words[n2:])     # 10%

def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')


n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 68 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
n = 8

model = Sequential([
  Embedding(vocab_size , n_embd) , 
  Flatten(2) , Linear(n_embd * 2 , n_hidden , bias=False) ,
  Flatten(2) , Linear(n_hidden * 2 , n_hidden , bias=False) ,
  Flatten(2) , Linear(n_hidden * 2 , n_hidden , bias=False) ,
  BatchNorm1d(n_hidden) , Tanh() , Linear(n_hidden , vocab_size) ,
])

with torch.no_grad():
    model.layers[-1].weight *= 0.1

paramters = model.parameters()
for p in paramters:
    p.requires_grad = True

max_steps = 200000
batch_size = 32 
lossi = []



for _ in range(max_steps):
    # we will use minibatch we use aprox grad with more steps overall 
    ix = torch.randint( 0 , Xtr.shape[0] , (32 ,))
    Xb , Yb = Xtr[ix] , Ytr[ix]
    x = Xb
    # forward pass
    x = model(x)
    # calculting the loss
    loss = F.cross_entropy(x , Ytr[ix])

    # backward pass
    for p in paramters:
      p.grad = None
    loss.backward()


    #lr = lrs[_]
    lr = 0.1 if _ < 150000 else 0.01
    for p in paramters:
        p.data += -lr * p.grad
    if _ % 1000 == 0: 
        print(loss.item())
    lossi.append(loss.log10().item())

plt.plot(torch.tensor(lossi).view(200 , -1).mean(1))
plt.show()        

for layer in model.layers:
  layer.training = False

@torch.no_grad()
def split_loss(split):
  x , y = {
    'train' : (Xtr , Ytr) ,
    'val' : (Xdev , Ydev) ,
    'test' : (Xdev , Ydev)
  }[split]

  x = model(x)
  loss = F.cross_entropy(x , y)
  print(split , loss.item())


split_loss('train')
split_loss('val')