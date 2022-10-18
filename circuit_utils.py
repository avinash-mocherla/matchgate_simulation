import itertools
import numpy as np
from scipy.special import comb



I = np.array(np.identity(2),dtype =complex)
X = np.array([[0,1],[1,0]],dtype =complex)
Y = np.array([[0,-1j],[1j,0]],dtype =complex)
Z = np.array([[1,0],[0,-1]],dtype =complex)
H = 1/np.sqrt(2) * (X + Z)


def kron_product(list):
    res = list[0]
    for i in range(len(list)-1):
        res = np.kron(res, list[i+1])
    return res

def str_to_pauli(basis):
    pauli_list = []
    for pauli_string in basis:
        temp = []
     
        for letter in pauli_string:
   
            if letter == 'I':
                temp.append(I)
            elif letter == 'X':
                temp.append(X)
            elif letter == 'Y':
                temp.append(Y)
            elif letter =='Z':
                temp.append(Z)
        pauli_list.append(kron_product(temp))

    return pauli_list

linear_basis = str_to_pauli(['XI', 'YI', 'ZX', 'ZY'])
quadratic_basis = str_to_pauli(['IZ','XX','XY','YX','YY','ZI'])
cubic_basis = str_to_pauli(['IX','IY','XZ','YZ'])

linear_basis_cz = str_to_pauli(['IX', 'IY', 'ZX', 'ZY'])
quadratic_basis_cz = str_to_pauli(['XI','XZ','YI','YZ'])
cubic_basis_cz = str_to_pauli(['XX','XY','YX','YY'])
# extra_basis_cz = str_to_pauli(['II','XX', 'YY','ZZ'])



linear = np.array([4,8,13,14], dtype = np.int64)
quadratic = np.array([3,5,6,9,10,12], dtype =np.int64)
cubic = np.array([1,2,7,11], dtype = np.int64)


linear_cz = np.array([1,2,13,14], dtype = np.int64)
quadratic_cz = np.array([4,7,8,11], dtype =np.int64)
cubic_cz = np.array([5,6,9,10], dtype = np.int64)

def G(A,B): 
    return np.array([[A[0,0],0,0,A[0,1]],[0,B[0,0], B[0,1], 0],[0, B[1,0], B[1,1], 0],[A[1,0], 0 ,0, A[1,1]]], dtype = complex)

SWAP = G(I,X)

def len_L(N, N_swaps):
    sum = 0
    for i in range(N_swaps+1):
        sum += comb(2*N, 2*i + 2)
    return sum 

def create_comp_basis(N):

    
    init = np.zeros((2**N,2**N))

    basis = []

    for i in range(2**N):
        for j in range(2**N):
            temp = init.copy()
            temp[i][j] = 1
            basis.append(temp)

    return basis


def decompose(U, basis = 'pauli'):
    N = int(np.log2(len(U)))

    if basis == 'pauli':
        basis = list(itertools.product(['I', 'X', 'Y', 'Z'], repeat = N))

        mats = str_to_pauli(basis)

        for i, matrix in enumerate(mats):
           print(basis[i], np.trace(matrix @ U)/4)

    
    elif basis == 'comp':
        
        mats = create_comp_basis(N)
        i = 0
        for matrix in mats: 
            a = np.trace(matrix@U)/4
            if a >0:
                i+=1
                print(a)
        print(i)
def tensor_identity(N,U,indices):

    U_tot = np.identity(2**N, dtype = complex)
 
    if type(indices) == int: 
        U = [np.identity(2) for i in range(0, indices)] +[U] + [np.identity(2) for i in range(indices+1, N)]
    else: 
        U = [np.identity(2) for i in range(0, indices[0])] + [U] + [np.identity(2) for i in range(indices[1]+1, N)]
        
    U = kron_product(U)

    U_tot = U @ U_tot  

    return U_tot


    
U = lambda theta: np.array([[1,0,0,0], 
              [0,1,0,0], 
              [0,0,1,0],
              [0,0,0,np.exp(1j*theta)]])

 



basis = list(itertools.product(['I', 'X', 'Y', 'Z'], repeat = 2))

basis = str_to_pauli(basis)

def _R(U):
    U_dag = U.conj().T  
    G = np.zeros(shape =(len(basis), len(basis)), dtype = float)
    for i in range(len(basis)):
        for j in range(len(basis)):
            G[i][j] = np.real(np.trace(basis[j]@U@basis[i]@U_dag))/4
    return G

U = lambda theta: np.array([[1,0,0,0], 
              [0,1,0,0], 
              [0,0,1,0],
              [0,0,0,np.exp(1j*theta)]])


I = np.array(np.identity(2),dtype =complex)
X = np.array([[0,1],[1,0]],dtype =complex)
Y = np.array([[0,-1j],[1j,0]],dtype =complex)
Z = np.array([[1,0],[0,-1]],dtype =complex)
H = 1/np.sqrt(2) * (X + Z)

A = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], dtype = complex)
K = lambda theta: G(I, np.array([[np.cos(theta), -1j * np.sin(theta)], 
                                 [-1j* np.sin(theta), np.cos(theta)]] ))

CZ = lambda theta: np.array([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,np.exp(1j * theta)]])

H = lambda theta: G(I, np.array([[np.cos(theta), -1 *np.sin(theta)], 
                                  [ np.sin(theta), np.cos(theta)]] ))      

iSWAP =  G(I, np.array([[0, 1j ], 
                        [1j,0]] ))

R = lambda theta: np.array([[np.cos(theta), 1j*np.sin(theta)],
    [1j*np.sin(theta), np.cos(theta)]])

UNL = lambda a, b, c: G( np.exp(1j*c)*R(a-b), np.exp(-1j*c) *R(a+b))

rot = lambda x1,x2: np.array([[np.exp((x1+x2)*1j),  0, 0, 0],
                            [0, np.exp((x1-x2)*1j),  0, 0],
                            [0, 0, np.exp((x2-x1)*1j),  0],
                            [0, 0, 0, np.exp((-x1-x2)*1j)]])

Cphase = lambda phi: np.array([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,np.exp(1j*phi)]])


SWAP = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]])

def swap(c, i, j):
    c = list(c)
    c[i], c[j] = c[j], c[i]
    return ''.join(c)

def hamming(n):
    c = 0
    while n:
        c += 1
        n &= n - 1

    return c


def len_L(N, N_swaps):
    sum = 0
    for i in range(N_swaps+1):
        sum += comb(2*N, 2*i + 2)
    return sum 
    


def g(N,n,m):
    import matplotlib.pyplot as plt
    import scipy.special as sp

  
    y = [(N/(i+1) * sum([len_L(n,j) for j in range(i+1)])) for i in range(m)]

    def f(n,m):
        k = (2*m)+2
        r = k**2/(2*n-k+1)**2
        return 1/((1-r)**2)

    def g(n,m):
        k = (2*m)+2
        r = (k**2)/(2*n - k + 1)**2
        return 1/(1-r)

    print([g(n,i) for i in range(m)])


    bound = [(N/(i+1)) * f(n,i) * comb(2*n, 2*i +2) for i in range(m)]
    bound2 = [N * g(n,i) * comb(2*n, 2*i +2) for i in range(m)]
    bound3 =  [((N*(i+2))/(i+1)) * g(n,i) * comb(2*n, 2*i +2) for i in range(m)]
    print(y)
    print(bound)
    print(bound2)
    print(bound3)
    
    plt.plot(np.log(bound3), c='pink')
    plt.plot(np.log(bound2), c='green')
    plt.plot(np.log(bound), c='blue')
    plt.plot(np.log(y), c= 'red')

    plt.show()






    
