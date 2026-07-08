    
import itertools 
import numpy as np
from data_paths import npy_path

def init_statevector(N):

    Z_indices = [3 * 4**i for i in range(N)]
    indices = []

    for i in range(1,N):
        print(i)
        indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])

    np.save(npy_path('init14.npy'), np.array(indices))

# init_statevector(28)

a = np.load(npy_path('init14.npy'))
print(len(a))