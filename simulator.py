
import itertools

import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from circuit_utils import *
from copy import copy
from scipy.special import comb
from circuit import clifford_circuit, convert_non_nn_ladder, random_type_circuit, layered_circuit
import matplotlib.pyplot as plt
import time
from numba import njit
import cProfile
  

def dot(b, A):
    return np.dot(b, np.transpose(A))

class Simulator:
    
    def __init__(self, N, pruned = True):
        self.N = N
        self.linear = np.array([4,8,13,14], dtype = 'object')
        self.quadratic = np.array([3,5, 6, 9, 10,12], dtype = 'object')
        self.cubic = np.array([1,2,7,11], dtype = 'object')
        self.pruned = pruned
        basis = list(itertools.product(['I', 'X', 'Y', 'Z'], repeat = 2))
        self.basis = str_to_pauli(basis)
      
        self.lengths = [1]
        self.measurements = []
        self.rho_lengths = []

    def init_statevector(self):

        Z_indices = [3 * 4**i for i in range(self.N)]
        indices = []

        for i in range(1,self.num_swaps+2):
            indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])       

        statevector = dict(zip(indices, list(np.ones(len(indices)))))
        return statevector

   
    # def simulate(self, circuit, measurement_vector):


   
    def simulate(self, circuit, measurement_vector = None, rho_vector = None, threshold=0.0, verbose = False):

        #Need to change this to number of unique SWAP's
        self.num_swaps = sum([1 for i in circuit if (i[0] == 'SWAP' or i[0] == 'CZ')]) 

        if rho_vector == None: 
            rho_vector = self.init_statevector()
            z_row_index = 2**((2*self.N)-1)+2**((2*self.N)-2)
            rho_vector = {z_row_index:1.0}

        if measurement_vector == None:
            z_row_index = 2**((2*self.N)-1)+2**((2*self.N)-2)
            measurement_vector = {z_row_index:1.0}  

        xi = 0 
        xj = 0
        
        self.time = 0

        self.num_gates = len(circuit)
        threshold_counter = 0
        times =0 
        while (xi + xj) <= (self.num_gates) - 1:
            
            if verbose == True: 
                print('Progress:',np.round((xi + xj) / self.num_gates,2)*100,'%')
            
            gate_type, U, qubits = circuit[-(xj+1)]
            xj += 1

            # if gate_type == 'CZ' or  gate_type == 'SWAP':
            if gate_type == 'SWAP':
                measurement_vector = self.apply_SWAP_gate(measurement_vector, qubits)

            if gate_type == 'H':
                measurement_vector = self.apply_H_gate(measurement_vector, qubits)

            if gate_type == 'CZ':
                
                R = self._R(U.conj().T,'CZ')
                R.append([[1.0]])
                visited_indices = set()
                
                for index in measurement_vector.copy():  
                    if index not in visited_indices:
                        
                        orbit = self.get_CZ_orbit(index, qubits)
              
                        visited_indices.update(orbit[1])

                        current_R = R[orbit[0]]
                        sub_vector = np.zeros(len(current_R))
                
                        for i,indexi in enumerate(orbit[1]):
                            sub_vector[i] = measurement_vector.get(indexi, 0)
                        

                        # res = dot(sub_vector, np.transpose(current_R))
                        
                        res = dot(sub_vector, current_R)
                        
                        for i,indexi in enumerate(orbit[1]):
                            measurement_vector[indexi] = res[i]

            elif gate_type == 'MG':
                
                R = self._R(U.conj().T,'MG')
                R.append([[1.0]])
            
                visited_indices = set()
                
                start = time.time()
                for index in measurement_vector.copy():  
                    if index not in visited_indices:
                        
                        s = time.time()
                        orbit = self.get_orbit(index, qubits[0])
                        
                        visited_indices.update(orbit[1])

                        current_R = R[orbit[0]]
         
                        sub_vector = np.zeros(len(current_R))
                        
                        for i,indexi in enumerate(orbit[1]):
                            sub_vector[i] = measurement_vector.get(indexi, 0)
                        
                        res = dot(sub_vector, current_R)
                        
                        for i,indexi in enumerate(orbit[1]):
                            # print(np.abs(res[i]))
                            if np.abs(res[i]) >= threshold:
                                measurement_vector[indexi] = res[i]
                            
                            else: pass
                        e = time.time() - s 
                        self.time += e 
                        

                end = time.time()
                times += (end-start)
            if self.pruned == True: 
                measurement_vector= self.prune_statevector(measurement_vector, threshold)
                self.measurement_vector = measurement_vector

            self.measurement_vector = measurement_vector
            # self.measurements.append(list(self.measurement_vector.values()))
            self.lengths.append(len(self.measurement_vector))
   



        # print('times', times)
        # char_list = ['0','3']
        # indices = [i for i in measurement_vector.keys() if all([char in char_list for char in np.base_repr(i,4)])]     
        # self.num_indices = len(indices)
        # rho_vector = self.init_statevector(self.N, effective_Nswaps)
   
        exp = self.expectation(measurement_vector,rho_vector)
        # exp = sum([measurement_vector[i] for i in indices])
        return exp
        
    def prune_statevector(self,statevector, threshold):
        statevector = {x:y for x,y in statevector.items() if np.abs(y) > threshold}
        return statevector

    def _R(self, U ,gate_type):

        U_dag = U.conj().T
       
        R = []
        if gate_type == 'MG':
            collection = [linear_basis,quadratic_basis,cubic_basis]
        if gate_type == 'CZ' or gate_type == 'SWAP':
            collection = [linear_basis_cz, quadratic_basis_cz, cubic_basis_cz]
    
        for basis in collection:
            G = np.zeros(shape =(len(basis), len(basis)), dtype = float)
                                                            
            for i in range(len(basis)):
                for j in range(len(basis)):
                    G[i][j] = np.real(np.trace(U_dag@basis[i]@U@basis[j]))/4
                    
            R.append(G)
        return R

    def get_CZ_orbit(self, index, qubits):
        binary = format(index, "0"+str(2*self.N) + "b")
      
        subspace = binary[2*qubits[0]:2*qubits[0]+2] + binary[ 2*qubits[1]: 2*qubits[1]+2]
  
        stem = int(binary[0:2*qubits[0]] + '00' + binary[2*qubits[0]+2:2*qubits[1]] + '00' + binary[2*qubits[1] +2 : 2*self.N],2)
      
        if subspace in ['0001', '0010','1101', '1110']: #IX, IY, ZX, ZY 
            return 0,stem + ( (4**(self.N - qubits[0] - 1) * np.array([0,0,3,3],dtype ='object')) + (4**(self.N - qubits[1] - 1) * np.array([1,2,1,2],dtype ='object')) )
        elif subspace in ['0100', '0111','1000', '1011']:#XI, XZ, YI, YZ
            return 1,stem + ( (4**(self.N - qubits[0] - 1) * np.array([1,1,2,2],dtype ='object')) + (4**(self.N - qubits[1] - 1) * np.array([0,3,0,3],dtype ='object')) )
        elif subspace in ['0101','0110','1001','1010']:  #XX, XY, YX, YY
            return 2,stem + ( (4**(self.N - qubits[0] - 1) * np.array([1,1,2,2],dtype ='object')) + (4**(self.N - qubits[1] - 1) * np.array([1,2,1,2],dtype ='object')) )
        else: return 3, [index]

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

    def get_orbit(self, index, q):
        
        binary = format(index, "0" + str(2*self.N) + "b")
        subspace = binary[2*q: 2*q + 4]
        stem = int(binary[0:2*q] + '0000' + binary[2*q + 4:2*self.N],2)
 
        if subspace in ['0100', '1000','1101', '1110']:
            return 0,stem + (4**(self.N-q-2) * self.linear)
        elif subspace in ['1100', '1010','1001', '0101', '0110','0011']:
            return 1,stem + (4**(self.N-q-2) * self.quadratic)
        elif subspace in ['0001','0010','0111','1011']:
            return 2,stem + (4**(self.N-q-2) * self.cubic)
        else:
            return 3, [index]


    def apply_H_gate(self, statevector, q):
        visited_indices = set()

        for index in statevector.copy():
            if index not in visited_indices:
                binary = format(index, "0" + str(2*self.N) + "b")
                subspace = binary[2*q: 2*q + 2]
                stem = int(binary[0:2*q] + '00' + binary[2*q + 2:2*self.N],2)

                if subspace == '11':
                    new_index = stem + 4**(self.N-q-1) 
                    visited_indices.add(new_index)
                    statevector[new_index], statevector[index] =  statevector[index], statevector.get(new_index,0)
                elif subspace == '01':
                    new_index = stem + 4**(self.N-q-1) * 3
                    visited_indices.add(new_index)
                    statevector[new_index], statevector[index] =  statevector[index], statevector.get(new_index,0)
                elif subspace == '10':
                    visited_indices.add(index)
                    statevector[index] =  -statevector[index]
                else: visited_indices.add(index)

               
        return statevector

    def expectation(self,measurement_vector, rho_vector):
        # Get dot product between measurement vector and rho vector 
        dot = 0
    
        for key, value in measurement_vector.items():
         
            dot += rho_vector.get(key,0) * value

        return dot
    
    def simulate_hard(self,circuit):
        
        if self.N > 12: 
            raise ValueError("Cannot hard simulate N > 12")

        U_tot = np.identity(2**self.N, dtype = complex)

        for gate in tqdm(circuit):
            U = gate[1]            
     
            indices = gate[2]
            U = [np.identity(2) for i in range(0, indices[0])] + [U] + [np.identity(2) for i in range(indices[1]+1, self.N)]
            U = kron_product(U)
            U_tot = U @ U_tot     

        Z1 = [Z] + [I for j in range(1,self.N)]
        Z_tot = kron_product(Z1)
    
        U_dag = U_tot.conj().T
        exp = np.real((U_dag @ Z_tot @ U_tot)[0,0])

        return exp


def test_standard():

    simulator = Simulator(n)

    start_time = time.time()
    res = simulator.simulate(circuit,verbose = True)
    print(simulator.lengths)
    time1 = time.time() - start_time


# test_standard()


# N,n,N_swaps = 100,32,1
# circuit = random_type_circuit(N,n,N_swaps)

# simulator = Simulator(n, pruned=True)
# res = simulator.simulate(circuit,verbose = True)
# print(simulator.lengths,res)

