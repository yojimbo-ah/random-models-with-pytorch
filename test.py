import numpy as np


# matrices with einsum 


def f(x : np.array) -> float:
    return np.einsum('i ->' , x) ** 2

y = f(np.array([1 ,2]))
def df (x:np.array) -> float:
    return 2 * (np.sum(x)) * np.ones_like(x)

dy = df(np.array([1,2])) 

print(dy)