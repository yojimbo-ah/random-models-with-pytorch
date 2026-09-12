import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
g = torch.Generator().manual_seed(2137483647)


g = torch.Generator().manual_seed(2137483647)

words = open('names.txt' , 'r').read().splitlines()



chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 3
X , Y = [] , []





# buildint the training and dev and test datasets
def build_dataset(words):  
  X, Y = [], []
  for w in words:

    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
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

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])



# embeding matrice for having the characters we have 27 we will save them in 27 , 2 matrice
# we will use indexing directly here but it wouldbe the same if we use one_hot for tensor then multiply
# the matrice (it faster using indexing) 

# best for now can be improved with more testing
hidden_layer = 300 
char_vec = 8
char_vec_concat = char_vec * 3

C = torch.randn([27 , char_vec] , requires_grad=True , generator=g)



w1 = torch.randn([char_vec_concat , hidden_layer] , requires_grad=True , generator=g) # we have 6 by 100 beaceause we have 6 from the concatenated  matrice and 100 is just for the hidden layer 
b1 = torch.randn(hidden_layer , requires_grad=True , generator=g)

w2 = torch.randn([hidden_layer , 27] , requires_grad=True , generator=g)
b2 = torch.randn(27 , requires_grad=True , generator=g)

#choosing the right learnin rate
#track setup
#lre = torch.linspace(-3 , 0 , 1000)
#lrs = 10**lre

#lri = []
#lossi = []

# forward pass
for _ in range(300000):
    # we will use minibatch we use aprox grad with more steps overall 
    ix = torch.randint( 0 , Xtr.shape[0] , (32 ,))


    emb = C[Xtr[ix]]
    h = torch.tanh(emb.view([-1 , char_vec_concat]) @ w1 + b1)
    # we have 100 hidden neurons in the hidden layer and we want to have 27 output ( one for each character activation)

    parameters = [C , w1 , b1 , w2 , b2]
    logits = h @ w2 + b2

    # does the same as exp of logits then normliztion then the likelihood in better way
    loss = F.cross_entropy(logits , Ytr[ix])
    # backward pass
    for p in parameters:
        p.grad = None

    loss.backward()
    #lr = lrs[_]
    lr = 0.1 if _ < 150000 else 0.01
    for p in parameters:
        p.data += -lr * p.grad
        
        #lri.append(lre[_])
        #lossi.append(loss.item())


# ploting after track
#plt.plot(lri , lossi)
#plt.show()

emb = C[Xtr]
h = torch.tanh(emb.view([-1 , char_vec_concat]) @ w1 + b1)


logits = h @ w2 + b2


loss = F.cross_entropy(logits , Ytr)
print(loss.item())




emb = C[Xdev]
h = torch.tanh(emb.view([-1 , char_vec_concat]) @ w1 + b1)


logits = h @ w2 + b2


loss = F.cross_entropy(logits , Ydev)
print(loss.item())


for _ in range(20):
    out = []
    context = [0] * block_size # initialize with all ...
    while True:
      emb = C[torch.tensor([context])] # (1,block_size,d)
      h = torch.tanh(emb.view(1, -1) @ w1 + b1)
      logits = h @ w2 + b2
      probs = F.softmax(logits, dim=1)
      ix = torch.multinomial(probs, num_samples=1, generator=g).item()
      context = context[1:] + [ix]
      out.append(ix)
      if ix == 0:
        break
    
    print(''.join(itos[i] for i in out))

