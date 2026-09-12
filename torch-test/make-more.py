import torch
import torch.nn.functional as F

g = torch.Generator().manual_seed(2137483647)

words = open('names.txt' , 'r').read().splitlines()



chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0





itos = {i: s for s, i in stoi.items()}
N = torch.zeros((27,27) , dtype=torch.int32)

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1 , ch2 in zip(chs , chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1 , ix2] += 1




P = (N+1).float()
P /= P.sum(1 , keepdim=True)


g = torch.Generator().manual_seed(2147483647)
out = []

for i in range(10):
    ix = 0
    nin = 5
    while True:
        p = P[ix]
        ix = torch.multinomial(p ,num_samples=1 ,replacement=True , generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break

print(''.join(out))

prob_lileklyhood = 0
n = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1 , ch2 in zip(chs , chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        prob_lileklyhood += logprob
        n += 1 
nll = -prob_lileklyhood / n
print(prob_lileklyhood)
print(nll)

xs = []
ys = []
count = 0
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1 , ch2 in zip(chs , chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
        count += 1
xs = torch.tensor(xs)
ys = torch.tensor(ys)

xenc = F.one_hot(xs , num_classes=27).float()
# ramdomizing the weights
w = torch.randn((27 , 27) , generator=g , requires_grad=True)

# gradient decent
for i in range(200):
    # Forward pass 
    logits = xenc @ w # first operation of the neurel network
    counts = logits.exp() # log counts equivalents to the N matrix
    probs = counts / counts.sum(1 , keepdim=True) # nprmalization here 

    # loss function using the negative likelyhood
    loss = -probs[torch.arange(count) , ys].log().mean() 
    # Backward pass

    w.grad = None # reseting the gradtiants
    loss.backward()
    w.data += -50 * w.grad
    
    print(loss.item())


for i in range(5):
    out = []
    ix = 0
    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float()
        logits = xenc @ w 
        counts = logits.exp()
        p = counts / counts.sum()

        ix = torch.multinomial(p , num_samples=1 , replacement=True , generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(out)



# each row inside the prob matrice can be interpretted as raw of probabilies beceause is it normalized ( currently weights are random)
# and inputs are only 5 double characters 

