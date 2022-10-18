
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
def main_loop(N,q,dict):
    visited_indices = set()
        
    for index in dict.copy():  
        if index not in visited_indices:
            key,orbit = get_orbit(N, index, q)
            visited_indices.update(orbit)
            
            write(dict, orbit)


@njit(cache = True)
def main_loop_cz(N,q1,q2,dict):
    visited_indices = set()
        
    for index in dict.copy():  
        if index not in visited_indices:
            key,orbit = get_cz_orbit(N, index, q1,q2)
            visited_indices.update(orbit)

            write(dict, orbit)

@njit
def write(dict, orbit):
    for i,indexi in enumerate(orbit):
        dict[indexi] = 0
    
class Simulator5:
    
    def __init__(self, N):
        self.N = N
        # self.lengths = [1]
        self.msm_lengths = []
        self.rho_lengths = []

    def init_statevector(self):

        Z_indices = [3 * 4**i for i in range(self.N)]
        indices = []

        for i in range(1,self.num_swaps+2):
            indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])       

        statevector = Dict.empty(key_type=types.int64,
                                value_type=types.float64)

        for i in indices: 
            statevector[i] = 1.0
        return statevector



    def simulate(self, circuit, measurement_vector = None, rho_vector = None, verbose = False):

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
                self.apply_gate(measurement_vector, x)
                xj+=1 
                self.msm_lengths.append(len(measurement_vector))
            else: 
                x = xi
                self.apply_gate(rho_vector, x)
                xi +=1
                self.rho_lengths.append(len(rho_vector))



    def apply_gate(self,dict, x):
            gate_type,U, qubits = self.circuit[x]

            if gate_type == 'CZ':
                main_loop_cz(self.N, qubits[0],qubits[1], dict)

            elif gate_type == 'MG':
               
                main_loop(self.N, qubits[0], dict)

