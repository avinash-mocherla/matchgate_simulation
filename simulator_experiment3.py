
import cProfile
import itertools

import numpy as np
from numpy.testing._private.utils import measure
import scipy.sparse as sp
from circuit_utils import *
from copy import copy
from tqdm import tqdm 
from scipy.special import comb
from circuit import convert_non_nn_ladder, random_type_circuit, layered_circuit
import matplotlib.pyplot as plt
import time

   
class Simulator:
    
    def __init__(self, N, pruned = False):
        self.N = N
        self.linear = np.array([4,8,13,14], dtype = 'object')
        self.quadratic = np.array([3,5, 6, 9, 10,12], dtype = 'object')
        self.cubic = np.array([1,2,7,11], dtype = 'object')
        self.pruned = pruned
        basis = list(itertools.product(['I', 'X', 'Y', 'Z'], repeat = 2))
        self.basis = str_to_pauli(basis)
      
        self.lengths = [1]
        self.measurement_lengths = []
        self.rho_lengths = []

    def init_statevector(self):

        Z_indices = [3 * 4**i for i in range(self.N)]
        indices = []

        for i in range(1,self.num_swaps+2):
            indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])       

        statevector = dict(zip(indices, list(np.ones(len(indices)))))
        return statevector

   
    # def simulate(self, circuit, measurement_vector):
    def simulate(self, circuit, measurement_vector = None, rho_vector = None, threshold=0.0):

        #Need to change this to number of unique SWAP's
        self.num_swaps = sum([1 for i in circuit if (i[0] == 'SWAP' or i[0] == 'CZ')]) 

        # if rho_vector == None: 
        #     # pass
        #     rho_vector = self.init_statevector()
          
        if measurement_vector == None:
            z_row_index = 2**((2*self.N)-1)+2**((2*self.N)-2)

            measurement_vector = {z_row_index:1.0}  

        xi = 0 
        xj = 0
        
        self.num_gates = len(circuit)
        threshold_counter = 0
        while (xi + xj) <= (self.num_gates) - 1:
            
            # print('Progress:',np.round((xi + xj) / self.num_gates,2)*100,'%')
        
            gate_type, U, qubits = circuit[-(xj+1)]
            xj += 1

            # if gate_type == 'CZ' or  gate_type == 'SWAP':
            if gate_type == 'SWAP':
                measurement_vector = self.apply_SWAP_gate(measurement_vector, qubits)
                threshold_counter += 1

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
                        
                        res = np.dot(sub_vector, np.transpose(current_R))
                        
                        for i,indexi in enumerate(orbit[1]):
                            measurement_vector[indexi] = res[i]

            elif gate_type == 'MG':

                R = self._R(U.conj().T,'MG')
                R.append([[1.0]])
            
                visited_indices = set()
                
                for index in measurement_vector.copy():  
                    if index not in visited_indices:
                        
                        orbit = self.get_orbit(index, qubits[0])
                        
                        visited_indices.update(orbit[1])

                        current_R = R[orbit[0]]
         
                        sub_vector = np.zeros(len(current_R))
                        
                        for i,indexi in enumerate(orbit[1]):
                            sub_vector[i] = measurement_vector.get(indexi, 0)
                        
                        res = np.dot(sub_vector, np.transpose(current_R))
                        
                        for i,indexi in enumerate(orbit[1]):
                            # print(np.abs(res[i]))
                            # if np.abs(res[i]) > threshold:
                            measurement_vector[indexi] = res[i]
                            # else: pass
                          
                self.measurement_vector = measurement_vector
                self.lengths.append(len(measurement_vector))
                self.measurement_lengths.append(measurement_vector)
   

            if self.pruned == True: 
                measurement_vector= self.prune_statevector(measurement_vector, threshold)

        char_list = ['0','3']
        indices = [i for i in measurement_vector.keys() if all([char in char_list for char in np.base_repr(i,4)])]     
        self.num_indices = len(indices)
        # rho_vector = self.init_statevector(self.N, effective_Nswaps)
        # exp = self.expectation(measurement_vector,rho_vector)
        exp = sum([measurement_vector[i] for i in indices])
        return exp
        
    def prune_statevector(self,statevector, threshold):
        statevector = {x:y for x,y in statevector.items() if np.abs(y) >= threshold}
        return statevector
    
    def _R(self, U ,gate_type):

        #Potentially Broken
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
        # print('binary',binary)
        # print('qubits',qubits )
        subspace = binary[2*qubits[0]:2*qubits[0]+2] + binary[ 2*qubits[1]: 2*qubits[1]+2]
        # print('subspace', subspace)
        stem = int(binary[0:2*qubits[0]] + '00' + binary[2*qubits[0]+2:2*qubits[1]] + '00' + binary[2*qubits[1] +2 : 2*self.N],2)
        # print('stem', stem)
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


def givens_circuit(N):
    circuit1 = []
    circuit2 = []
    for i in range(N-2):
        A1 = A(np.random.rand() * 2*np.pi)
        circuit1.append(('MG', G(I,A1), (i+1,i+2)))
    for i in range(N-2):
        A1 = A(np.random.rand() * 2*np.pi)
        circuit2.append(('MG', G(I,A1), (i,i+1)))
    givens_circuit1 = [val for pair in zip(circuit1, circuit2) for val in pair]
    
    # print([i[2] for i in givens_circuit1])

    circuit3 = []
    circuit4 = []
    for i in range(N-2):
        A1 = A(np.random.rand() * 2*np.pi)
        circuit3.append(('MG', G(I,A1), (i+1+N,i+2+N)))
    for i in range(N-2):
        A1 = A(np.random.rand() * 2*np.pi)
        circuit4.append(('MG', G(I,A1), (i+N,i+1+N)))
    givens_circuit2 = [val for pair in zip(circuit3, circuit4) for val in pair]

    givens_circuit = [val for pair in zip(givens_circuit1, givens_circuit2) for val in pair]

    givens_circuit.insert(0,('MG', G(X,X), (0,1)))
    givens_circuit.insert(1,('MG', G(X,X), (N,N+1)))
    return givens_circuit

def trotter_circuit(N_sites, N_steps, J, U, tau):

    trotter_circuit = []
    #Add initial Givens rotations to circuit
    trotter_circuit.extend(givens_circuit(N_sites))
    for i in range(N_steps):
        trotter_step_i = trotter_step(N_sites, J, U, tau)
        trotter_circuit.extend(trotter_step_i)
    return trotter_circuit

def trotter_step(N_sites, J, U, tau):
    trotter_step_list = []
    #Odd to Even hops, spin up
    for i in range(0,N_sites,2):
        gate = K(-tau*J)
        trotter_step_list.append(('MG',gate,(i,i+1)))
    #Odd to Even hops, spin down
    for i in range(0,N_sites,2):
        gate = K(-tau*J)
        trotter_step_list.append(('MG',gate,(i+N_sites,i+N_sites+1)))
    #Even to Odd hops, spin up
    for i in range(1,N_sites-1,2):
        gate = K(-tau*J)
        trotter_step_list.append(('MG',gate,(i,i+1)))
    #Even to Odd hops, spin down
    for i in range(1,N_sites-1,2):
        gate = K(-tau*J)
        trotter_step_list.append(('MG',gate,(i+N_sites,i+N_sites+1)))
    # onsite   
    for i in range(N_sites):
        phi = U*tau
        # z_rots = rot(phi/4, phi/4)
        trotter_step_list.append(('CZ', Cphase(phi), (i, i+N_sites)))
    return trotter_step_list

#Define Model parameters 
N_sites = 3
N_steps = 1
J = 0.3
U = 0.9
tau = float(0.3/J)

circuit = trotter_circuit(N_sites=N_sites, N_steps = N_steps, J = J, U = U, tau = tau)

results = []
results1 = []
results2 = []
results3 = []
ranks = []

depths = np.arange(1,2,1)

for d in depths:
    f = []
    l = []
    print('depth', d)

    for k in range(3):

        circuit = trotter_circuit(N_sites=N_sites, N_steps = N_steps, J = J, U = U, tau = tau)

        simulator = Simulator(2*N_sites, pruned = True)
        fast_res = simulator.simulate(circuit,threshold = 0.001)
        f.append(fast_res)
        # print(simulator.num_indices)
        simulator = Simulator(2*N_sites, pruned = False)
        slow_res = simulator.simulate(circuit)
        l.append(slow_res)
        # print(simulator.num_indices)
    # normal_lengths = simulator.measurement_lengths
   
    results.append(np.average(np.abs(np.array(f) - np.array(l))))
    # results1.append(np.average(np.array(l)))   
    # results2.append(np.max(np.array(f)))
    # results3.append(np.max(np.array(l)))

# print(results2, results3)

# diff = np.abs(np.array(results1) - np.array(results))
# diff1= np.abs(np.array(results3) - np.array(results2))
# print(diff1)
# print(diff)

plt.scatter(np.arange(len(results)),results)
# plt.plot(diff1)
plt.show()
# pruned_lengths = simulator.measurement_lengths

# print(pruned_lengths)

# three_index = np.array([pruned_lengths[i][3] for i in range(len(pruned_lengths))])
# print(three_index)



# three_index1 = np.array([normal_lengths[i][3] for i in range(len(normal_lengths))])
# print(three_index1- three_index)
# print(three_index1)

# # plt.plot(three_index)
# # plt.plot(three_index1)
# plt.plot(np.abs(three_index1 - three_index))
# plt.show()

# print('fast time', fast_time, 'slow time', slow_time)
# print('fast res', fast_res, 'slow res', slow_res)

