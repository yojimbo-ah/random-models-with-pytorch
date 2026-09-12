import torch
import torch.nn.functional as F

torch.manual_seed(1337)

B , T , C = 4 , 8 , 32
head_size = 16
x = torch.randn(B,T,C)
key = torch.nn.Linear(C , head_size , bias=False)
query = torch.nn.Linear(C , head_size , bias=False)
value = torch.nn.Linear(C , head_size , bias=False)
k = key(x)
q = query(x)
wei = q @ k.transpose(-2 , -1) # (B,T,T)

tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0 , float('-inf'))
wei = wei * head_size ** -0.5
wei = F.softmax(wei, dim=-1)
v = value(x)
out = wei @ v

print(out)