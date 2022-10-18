from time import time
from  numba import njit

@njit
def a():

    x = 0

    while True:

        yield x

        y = ~(x << 1)

        x = (x - y) & y 

import time 


# t = time.time()
# [b(n) for n in range(2**n)]*3
# s1 = time.time() - t

n = 28

gen = a()

t = time.time()
[next(gen) for i in range(2**n)]*3
s2 = time.time() - t

print(s2)


