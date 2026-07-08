
import itertools
from zlib import ZLIB_RUNTIME_VERSION

import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from circuit_utils import *
from copy import copy
from scipy.special import comb
from circuit import FH_circuit, clifford_circuit, convert_non_nn_ladder, random_type_circuit, layered_circuit
import matplotlib.pyplot as plt
import time
from numba import njit, jit
from numba.typed import Dict
from numba.core import types
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def _npy_path(filename):
    """Return the absolute path for a cached NumPy file."""
    return os.path.join(DATA_DIR, filename)
  
@njit(cache = True)
def dot(b, A):
    return np.dot(b, A.T)

@njit(cache = True)
def dot1(A,b):
    return np.dot(A,b)


@njit(cache = True)
def bin_to_int(ls):
    sum = 0
    for index, i  in enumerate(ls): 
    
        sum += i * 2**(3-index)

    return sum

@njit(cache = True)
def bin_to_int_cz(ls):
    sum = 0
    for index, i  in enumerate(ls): 
    
        sum += i * 2**(1-index)

    return sum

@njit(cache=True)
def get_orbit(N, index,q):
    quotient = index
    # print('quotient', quotient)
    binary_string = np.zeros(2*N, dtype = np.int64)
    # print('binary_string', binary_string)
    i = 1

    while quotient > 0:
        quotient, remainder = divmod(quotient,2)
        binary_string[-i] = remainder
        i+=1 

    subspace_group = np.int64(bin_to_int(binary_string[2*q:2*q+4]))
    # print('subspace group', subspace_group)

    multiplier = np.int64(4** (N-q-2))
    # print('multiplier', multiplier)

    subspace = np.int64(subspace_group * multiplier)
    # print('subspace', subspace)

    stem = np.int64(index - subspace)
    # print('stem', stem)

    if subspace_group in linear:
        return 0,stem + (multiplier * linear)
    elif subspace_group in quadratic:
        return 1,stem + (multiplier * quadratic)
    elif subspace_group in cubic:
        return 2,stem + (multiplier * cubic)
    else: return 3, np.array([index])

@njit
def a():
    x = 0

    while True:

        yield x

        y = ~(x << 1)

        x = (x - y) & y 
   
    

@njit(cache=True)
def get_cz_orbit(N, index,q1,q2):
    quotient = index
    binary_string = np.zeros(2*N, dtype = np.int64)

    i = 1
  

    while quotient > 0:
        quotient, remainder = divmod(quotient,2)
        binary_string[-i] = remainder
        i+=1 
 
    subspace_group1 = bin_to_int_cz(binary_string[2*q1:2*q1+2])
    subspace_group2 = bin_to_int_cz(binary_string[2*q2:2*q2+2])
    subspace_group = np.int64(subspace_group1*4 + subspace_group2)
  
    multiplier1 = np.int64(4** (N-q1-1))
    multiplier2 = np.int64(4** (N-q2-1))


    subspace = np.int64((subspace_group1 * multiplier1) + (subspace_group2 * multiplier2))


    stem = np.int64(index - subspace)


    if subspace_group in linear_cz:
        return 0,stem + (multiplier1 * np.array([0,0,3,3],dtype = np.int64) + multiplier2 *np.array([1,2,1,2],dtype = np.int64))
    elif subspace_group in quadratic_cz:
        return 1,stem +(multiplier1 * np.array([1,1,2,2],dtype = np.int64) + multiplier2 *np.array([0,3,0,3],dtype = np.int64))
    elif subspace_group in cubic_cz:
        return 2,stem + (multiplier1 * np.array([1,1,2,2],dtype = np.int64) + multiplier2 *np.array([1,2,1,2],dtype = np.int64))
    else: return 3, np.array([index])

@njit(cache = True)
def main_loop(N,q,dict,R0,R1,R2,flag,threshold):
    visited_indices = set()
        
    for index in dict.copy():  
        if index not in visited_indices:
            orbit = get_orbit(N, index, q)
            visited_indices.update(orbit[1])

            if orbit[0] != 3:
                                
                if orbit[0] == 0:
                    current_R = R0
                elif orbit[0] == 1:
                    current_R = R1
                elif orbit[0] == 2:
                    current_R = R2

                read_and_write(dict, current_R,orbit,flag,threshold)

            else: pass


@njit(cache = True)
def main_loop_cz(N,q1,q2,dict,R0,R1,R2,flag,threshold):
    visited_indices = set()
        
    for index in dict.copy():  
        if index not in visited_indices:

            orbit = get_cz_orbit(N, index, q1,q2)
          
            visited_indices.update(orbit[1])

            if orbit[0] != 3:
                                
                if orbit[0] == 0:
                    current_R = R0
                elif orbit[0] == 1:
                    current_R = R1
                elif orbit[0] == 2:
                    current_R = R2

                read_and_write(dict, current_R,orbit,flag,threshold)

            else: pass

@njit
def read_and_write(dict, current_R, orbit,flag,threshold):

    sub_vector = np.zeros(len(current_R))

    for i,indexi in enumerate(orbit[1]):
            sub_vector[i] = dict.get(indexi, 0)
    
    if flag == 1:
        sub_vector = dot(sub_vector, current_R)   
    else: sub_vector = dot1(current_R, sub_vector)   

    for i,indexi in enumerate(orbit[1]):
        if np.abs(sub_vector[i]) > threshold: 
            dict[indexi] = sub_vector[i]
    

@jit(forceobj = True)
def _R(U, basis):

    U_dag = U.conj().T
    R = np.zeros((len(basis), len(basis)), dtype = float)
                                                    
    for i in range(len(basis)):
        for j in range(len(basis)):
            R[i][j] = np.real(np.trace(U_dag@basis[i]@U@basis[j]))/4
    
            
    return R

@njit
def expectation(measurement_vector, rho_vector):
    # Get dot product between measurement vector and rho vector 
    dot = 0

    for key, value in measurement_vector.items():
        
        dot += rho_vector.get(key,0) * value

    return dot


class Simulator4:
    
    def __init__(self, N):
        self.N = N
        # self.lengths = [1]
        self.msm_lengths = []
        self.rho_lengths = []
        
       

    def init_statevector(self):
        
        if self.num_swaps > self.N - 2:
            path = _npy_path(f'init{self.N}.npy')
            is_file = os.path.isfile(path)
            if is_file:
                # print('Initial statevector already found!')
                indices = np.load(path)
                # print("Loaded.")
            else: 
                print('Initial statevector not found...')
                gen = a()
                indices = [next(gen) for i in range(2**self.N)]
                np.save(path, indices)
                print("Saved.")
        else:
            Z_indices = [3 * 4**i for i in range(self.N)]
            indices = []

            if self.num_swaps < self.N - 2:

                for i in range(1,self.num_swaps+2):
                    indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])       

            else:
                for i in range(1,self.N):
                    indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])       


        statevector = Dict.empty(key_type=types.int64,
                                value_type=types.float64)
        j = 0
        for i in indices: 
            j+=1
            statevector[i] = 1.0
        return statevector



    def simulate(self, circuit, measurement_vector = None, rho_vector = None, verbose = False,threshold = 0.0):

        #Need to change this to number of unique SWAP's
        self.num_swaps = sum([1 for i in circuit if (i[0] == 'SWAP' or i[0] == 'CZ')]) 

        if rho_vector == None: 
            rho_vector = self.init_statevector()
            self.rho_lengths.append(len(rho_vector))
    

        if measurement_vector == None:
            z_row_index = np.int64(2**(2*self.N-1) + 2**(2*self.N-2))
   
            measurement_vector =  Dict.empty(
                                key_type=types.int64,
                                value_type=types.float64)

            measurement_vector[z_row_index] = 1.0



            self.msm_lengths.append(len(measurement_vector))
       
        xi = 0 
        xj = 0
        
        self.circuit = circuit

        self.num_gates = len(circuit)

        while (xi + xj) <= (self.num_gates)-1:
            
            if verbose == True: 
                print('Progress:',np.round((xi + xj) / self.num_gates,2)*100,'%')
   

            if len(measurement_vector) <= len(rho_vector):
                x = -(xj+1)
                self.apply_gate(measurement_vector, x, flag = 1,threshold=threshold)
                xj+=1 
                self.msm_lengths.append(len(measurement_vector))
            else: 
                x = xi
                self.apply_gate(rho_vector, x, flag = 0,threshold=threshold)
                xi +=1
                self.rho_lengths.append(len(rho_vector))

        exp = expectation(measurement_vector, rho_vector)
        self.msm_vector=measurement_vector
        self.rho_vector=rho_vector
        return exp

    def apply_gate(self,dict, x, flag,threshold):

            gate_type, U, qubits = self.circuit[x]

            if gate_type == 'SWAP':
                dict = self.apply_SWAP_gate(dict, qubits)

            if gate_type == 'CZ':
                if flag == 1: 
                    U = U.conj().T
                
                R0 = _R(U, linear_basis_cz)
                R1 = _R(U, quadratic_basis_cz)
                R2 = _R(U, cubic_basis_cz)

                main_loop_cz(self.N, qubits[0],qubits[1], dict,R0,R1,R2, flag,threshold)

            elif gate_type == 'MG':
                if flag == 1: 
                    U = U.conj().T
                
                R0 = _R(U, linear_basis)
                R1 = _R(U, quadratic_basis)
                R2 = _R(U, cubic_basis)

                main_loop(self.N, qubits[0], dict,R0,R1,R2, flag,threshold)


    def apply_SWAP_gate(self, statevector, qubits):
        visited_indices = set()
        for index in statevector.copy():
            if index not in visited_indices: 
                binary = format(index, "0"+str(2*self.N) + "b")
       
                string = binary[0:2*qubits[0]] + binary[2*qubits[1]:2*qubits[1]+2] + binary[2*qubits[0] + 2: 2* qubits[1]] + binary[2*qubits[0]:2*qubits[0] + 2] + binary[2*qubits[1]+2:2*self.N]

                new_index = int(string, 2)
                visited_indices.add(new_index)

                statevector[new_index], statevector[index] =  statevector[index], statevector.get(new_index,0)
        return statevector

# from circuit import mod_trotter_circuit
# n,m = 5,5
# threshold = 1e-6
# circuit = mod_trotter_circuit(n,m,1,0.2,0.9,1)
# simulator = Simulator4(2*n)
# res = simulator.simulate(circuit, verbose = True, threshold = threshold )
# print(simulator.msm_vector)
# plt.plot(simulator.msm_lengths)
# simulator2 = Simulator4(2*n)
# res1 = simulator2.simulate(circuit, verbose = True, threshold = 0.0 )
# # print(simulator.msm_vector)
# plt.plot(simulator2.msm_lengths)

# print(sum(simulator2.msm_lengths)/sum(simulator.msm_lengths))



# print('difference', res-res1, 'tol',threshold, 'ratio',  (res-res1)/threshold)
# print(sum(simulator.msm_lengths)*threshold*simulator.msm_lengths[-1]*0.03)
# # print(simulator.msm_lengths[-1]*threshold)


# plt.show()