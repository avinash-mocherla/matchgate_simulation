
import itertools

import numpy as np

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
            
            print('Progress:',np.round((xi + xj) / self.num_gates,2)*100,'%')
        
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
                            if np.abs(res[i]) > threshold:
                                measurement_vector[indexi] = res[i]
                            else: pass
                          
            self.measurement_vector = measurement_vector
           
            self.measurement_lengths.append(measurement_vector)
   

            if self.pruned == True: 
                measurement_vector= self.prune_statevector(measurement_vector, threshold)
                self.measurement_vector = measurement_vector

            self.lengths.append(len(measurement_vector))
        
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




#Experiment 1: Pure Matchgate circuits, average pruned vector element - average exact vector element

# N_sites = 16
# N_steps = 10
J = 0.3
U = 0.9
tau = float(0.3/J)

tols = [0.1,0.031,0.01,0.0031,0.001,0.00031,0.0001,0.000031,0.00001]
repeats = 10

#Hard comparison result
fig2, ax = plt.subplots()


fig, axs = plt.subplots(2,2, figsize = (12,12))
fig.tight_layout(pad=8.0)
twin_axes = [axs[0][0].twinx(), axs[0][1].twinx(), axs[1][0].twinx(),axs[1][1].twinx()]

#(row,column, n_sites, n_steps)
params = [(0,0,4,2), (0,1,6,2), (1,0,8,2), (1,1,10,2)]

for param in params:
    #Times
    times = []
    times_max = []
    times_min = []
    times_std = []

    #Accuracies
    accuracies = []
    accuracies_max = []
    accuracies_min = []
    accuracies_std = []

    paulis = []
    paulis_max = []
    paulis_min = []
    paulis_std = []

    # circuit = trotter_circuit(N_sites=param[2], N_steps =param[3], J = J, U = U, tau = tau)



    for i, tol in enumerate(tols):
        #Create temporary variables for storing repeats
        temp1 = []
        temp2 = []
        temp3 = []
        for n in range(repeats):

            circuit = trotter_circuit(N_sites=param[2], N_steps =param[3], J = J, U = U, tau = tau)
            print('iter', i, 'repeat', n)

            simulator = Simulator(2*param[2], pruned =False)
            # start = time.time()
            hard_res = simulator.simulate(circuit)
            # hard_time = time.time() - start

            #Run experiement 
            simulator = Simulator(2*param[2], pruned = True)
            start = time.time()
            res = simulator.simulate(circuit,threshold = tol)
            time_taken = time.time() - start

            #Append raw_results
            temp1.append(time_taken)
            temp2.append(np.abs(res - hard_res))
            temp3.append(sum(simulator.lengths))

        times.append(np.average(temp1))
        times_max.append(max(temp1))
        times_min.append(min(temp1))
        times_std.append(np.std(temp1))
      
        accuracies.append(np.average(temp2))
        accuracies_max.append(max(temp2))
        accuracies_min.append(min(temp2))
        accuracies_std.append(np.std(temp2))

        paulis.append(np.average(temp3))
        paulis_max.append(max(temp3))
        paulis_min.append(min(temp3))
        paulis_std.append(np.std(temp3))

    


    st = 'Trotter circuit: %s sites, %s steps, %s repeats' %(param[2], param[3], repeats)
    axs[param[0]][param[1]].set_title(st)
    ax1 = twin_axes[2*param[0]+ 1*param[1]]
    # axs[param[0]][param[1]].scatter(np.log10(tols),np.log10(accuracies), c = 'blue', s = 5)
    # axs[param[0]][param[1]].errorbar(np.log10(tols),np.log10(accuracies), yerr =  (accuracies_max,accuracies_min), color = 'blue',  ecolor = 'blue', linestyle='',fmt='o',alpha=0.3)
    axs[param[0]][param[1]].errorbar(np.log10(tols),np.log10(accuracies), yerr =  0.430*np.divide(accuracies_std,accuracies), color = 'blue',  ecolor = 'blue', linestyle='',fmt='o',alpha=0.3)
    axs[param[0]][param[1]].scatter(np.log10(tols),np.log10(accuracies_max),c = 'blue',s = 5,marker = 'x')
    axs[param[0]][param[1]].scatter(np.log10(tols),np.log10(accuracies_min), c = 'blue',s = 5,marker = 'x')

    # ax1.scatter(np.log10(tols),np.log10(times), c= 'red', s = 5)
    # ax1.errorbar(np.log10(tols),np.log10(times), yerr = (times_max, times_min),color = 'red', ecolor = 'red', linestyle='',fmt='o',alpha=0.3)
    # ax1.errorbar(np.log10(tols),np.log10(times), yerr = 0.430*np.divide(times_std,times),color = 'red', ecolor = 'red', linestyle='',fmt='o',alpha=0.3)
    # ax1.scatter(np.log10(tols),np.log10(times_max), c= 'red',s = 5,marker = 'x')
    # ax1.scatter(np.log10(tols),np.log10(times_min), c= 'red',s = 5, marker = 'x')

    # ax1.errorbar(np.log10(tols),np.log10(paulis), yerr = (paulis_max, paulis_min),color = 'red', ecolor = 'red', linestyle='',fmt='o',alpha=0.3)
    ax1.errorbar(np.log10(tols),np.log10(paulis), yerr = 0.430*np.divide(paulis_std,paulis),color = 'red', ecolor = 'red', linestyle='',fmt='o',alpha=0.3)
    ax1.scatter(np.log10(tols),np.log10(paulis_max), c= 'red',s = 5,marker = 'x')
    ax1.scatter(np.log10(tols),np.log10(paulis_min), c= 'red',s = 5, marker = 'x')
  
    axs[param[0]][param[1]].set_xlabel('log(Tolerance)')
    axs[param[0]][param[1]].set_ylabel('log(Absolute error)', color='b')
    ax1.set_ylabel('log(Simulation Time)', color='r')

   
plt.savefig("accuracies.png", transparent=True)
plt.show()

