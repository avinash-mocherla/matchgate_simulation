    
import itertools 
from pathlib import Path
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

def init_statevector(N):

    Z_indices = [3 * 4**i for i in range(N)]
    indices = []

    for i in range(1,N):
        print(i)
        indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])

    np.save(DATA_DIR / 'init14.npy', np.array(indices))

# init_statevector(28)

a = np.load(DATA_DIR / 'init14.npy')
print(len(a))