import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

g = torch.Generator().manual_seed(2137483647)

#current best training : 2.016110897064209 and test is 2.0990817546844482

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
context = 3
hidden_layer = 300 
char_vec = 8
char_vec_concat = char_vec * context
vocab_size = 27




std = (5 / 3) / (vocab_size ** 0.5)



C = torch.randn([vocab_size , char_vec] , requires_grad=True , generator=g)



w1 = (torch.randn([char_vec_concat , hidden_layer] , generator=g) * std).requires_grad_(True) # we have 6 by 100 beaceause we have 6 from the concatenated  matrice and 100 is just for the hidden layer 
#b1 = (torch.randn(hidden_layer , generator=g) * 0.01).requires_grad_(True)

w2 = (torch.randn([hidden_layer , vocab_size] ,  generator=g) * 0.1).requires_grad_(True)
b2 = (torch.randn(vocab_size , generator=g) * 0).requires_grad_(True)

bngain = torch.ones((1 , hidden_layer) , requires_grad=True)
bnbias = torch.zeros((1 ,hidden_layer) , requires_grad=True)

bnmean_running = torch.zeros((1 , hidden_layer))
bnstd_running = torch.ones((1 , hidden_layer)) 

parameters = [C , w1  , w2 , b2 , bngain , bnbias]

#choosing the right learnin rate
#track setup
#lre = torch.linspace(-3 , 0 , 1000)
#lrs = 10**lre

#lri = []
#lossi = []
batch_size = 32
# forward pass
for _ in range(300000):
    # we will use minibatch we use aprox grad with more steps overall 
    ix = torch.randint( 0 , Xtr.shape[0] , (batch_size ,))


    emb = C[Xtr[ix]] # (batch_size , context , char_vec)
    
    embcat = emb.view([-1 , char_vec_concat]) # concat the vectors
    hpreact = embcat @ w1 # hidden layer pre activation 
    # batch normalization layer
    bnmeani = hpreact.mean( 0 , keepdim=True)
    bnstdi = hpreact.std(0 , keepdim=True)
    hpreact = bngain * ( hpreact - bnmeani ) / bnstdi + bnbias # standarization using gausaion desterbetion 
    with torch.no_grad():
      bnmean_running = 0.999 * bnmean_running + bnmeani * 0.001
      bnstd_running = 0.999 * bnstd_running + bnstdi * 0.001

      

    # we will use scale and shift 
    h = torch.tanh(hpreact) # hidden layer tanh 
    # we have 100 hidden neurons in the hidden layer and we want to have 27 output ( one for each character activation)

    logits = h @ w2 + b2 

    # does the same as exp of logits then normliztion then the likelihood in better way
    loss = F.cross_entropy(logits , Ytr[ix])
    # backward pass
    if _ % 20000 == 0:
       print(loss.item())

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



@torch.no_grad()
def split_loss(split):
  x , y = {
    'train' : (Xtr , Ytr) ,
    'val' : (Xdev , Ydev) ,
    'test' : (Xdev , Ydev)
  }[split]
  emb = C[x] 
  embcat = emb.view(emb.shape[0] , -1)
  hpreact = embcat @ w1 
  hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
  h = torch.tanh(hpreact)
  logits = h @ w2 + b2 
  loss = F.cross_entropy(logits , y)
  print(split , loss.item())


split_loss('train')
split_loss('val')

"""""
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
"""

