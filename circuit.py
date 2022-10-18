from circuit_utils import *
import numpy as np
import random
import itertools
from scipy.stats import unitary_group

R = lambda theta: np.array([[np.cos(theta), 1j*np.sin(theta)],
                            [1j*np.sin(theta), np.cos(theta)]])

UNL = lambda a, b, c: G( np.exp(1j*c)*R(a-b), np.exp(-1j*c) *R(a+b))

rot = lambda x1,x2: np.array([[np.exp((x1+x2)*1j),  0, 0, 0],
                            [0, np.exp((x1-x2)*1j),  0, 0],
                            [0, 0, np.exp((x2-x1)*1j),  0],
                            [0, 0, 0, np.exp((-x1-x2)*1j)]])

SWAP = np.array([[1,0,0,0],
                 [0,0,1,0],
                 [0,1,0,0],
                 [0,0,0,1]])

CZ = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,1,0],
               [0,0,0,-1]])

Cphase = lambda phi: np.array([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0],
                             [0,0,0,np.exp(1j*phi)]])

S = np.array([[1,0],
              [0,1j]])

def G(A,B): 
    return np.array([[A[0,0],0,0,A[0,1]],[0,B[0,0], B[0,1], 0],[0, B[1,0], B[1,1], 0],[A[1,0], 0 ,0, A[1,1]]], dtype = complex)

A = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], dtype = complex)

K = lambda theta: G(I, np.array([[np.cos(theta), -1j * np.sin(theta)], 
                 [-1j* np.sin(theta), np.cos(theta)]] ))

H = 1/np.sqrt(2) * np.array([[1,1],
                              [1,-1]])

iSWAP =  G(I, np.array([[0, 1j ], 
                        [1j,0]] ))


def haar_random_MG():
    x = unitary_group.rvs(2)
    x = x/np.linalg.det(x)
    y = unitary_group.rvs(2)
    y = y/np.linalg.det(y)

    MG = G(x, y)
    
    return MG

def random_MG():
    r1 = np.random.rand()
    r2 = np.random.rand()
    a = A(r1)
    b = A(r2)
    MG = G(a,b)

    return MG

def convert_non_nn_ladder(circuit):
    new_circuit = []
    SWAP = np.array([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]])
    
    for gate in circuit: 
  
        gate_type, U, qubits = gate[0], gate[1], gate[2]
        if qubits[1] - qubits[0] != 1:
            MG_circuit = []
            for i in range(qubits[0]+1, qubits[1])[::-1]:
                MG_circuit.append(('SWAP', SWAP,(i,i+1)))
     
            MG = ('MG', U, (qubits[0],qubits[0]+1))
            
            MG_circuit.append(MG)
            for i in range(qubits[0]+1, qubits[1]):
                MG_circuit.append(('SWAP',SWAP,(i,i+1)))
      
            new_circuit.extend(MG_circuit)           
            
        else: new_circuit.append(gate)
    return new_circuit

def convert_non_nn(circuit):
    new_circuit = []
    SWAP = np.array([[1,0,0,0],
                     [0,0,1,0],
                     [0,1,0,0],
                     [0,0,0,1]])
    
    for gate in circuit: 
  
        gate_type, U, qubits = gate[0], gate[1], gate[2]
        if qubits[1] - qubits[0] != 1:
            MG_circuit = []
            
            MG_circuit.append(('SWAP', SWAP,(qubits[0],qubits[1])))
     
            CZ = ('CZ', U, (qubits[0],qubits[0]+1))
            MG_circuit.append(CZ)
      
            MG_circuit.append(('SWAP', SWAP,(qubits[0],qubits[1])))
      
            new_circuit.extend(MG_circuit)           
            
        else: new_circuit.append(gate)
    return new_circuit

def create_pauli_error_string(N,p):
    pauli_string = []
  
    for i in range(N):
        choice = np.random.choice(['I','X', 'Y', 'Z'], p= [1-3*p,p,p,p])
        pauli_string.append(choice)

    # print(kron_product(str_to_pauli(pauli_string)))

    return(kron_product(str_to_pauli(pauli_string)))

def random_type_circuit(N_gates =1, N_qubits = 2, N_swaps = 0, N_CZ = 0, seed = 0 ):
    """
    create a circuit with random matchgates and swaps
    """
    circuit = []
    swap_circuit = []
    cz_circuit = []

    # np.random.seed(seed)

    #Define possible pairs
    nn_pairs = [(i,i+1) for i in range(N_qubits-1)]
    lenlist = [i for i in range(len(nn_pairs))]


    #Define Matchgate circuits
    for i in range(N_gates):   

        circuit.append(('MG', random_MG(), nn_pairs[np.random.choice(lenlist)])) 

    for n in range(N_swaps):
        # np.random.seed(seed)
        qubits = nn_pairs[np.random.choice(lenlist)]

        gate = np.array([[1,0,0,0],
                         [0,0,1,0],
                         [0,1,0,0],
                         [0,0,0,1]])

        gate_type = 'SWAP'

        swap_circuit.append((gate_type, gate, qubits))


    for n in range(N_CZ):
        # np.random.seed(seed)
        qubits = nn_pairs[np.random.choice(lenlist)]
        # np.random.seed(seed)
        r = np.random.random()

        phi = r*2*np.pi

        cz_circuit.append(('CZ', Cphase(phi), qubits ))

        # z_rots = rot(phi/4, phi/4)

        # cz_circuit.append([('CZ', UNL(0,0,-phi/4), qubits), ('MG', z_rots, qubits)])


    circuit.extend(swap_circuit)
    circuit.extend(cz_circuit)

    np.random.shuffle(circuit)

    new_circuit = []
    for i,entry in enumerate(circuit):
        if type(entry) == list:
            for c in entry:
                new_circuit.append(c)
        else:
            new_circuit.append(entry)

    return list(new_circuit)




def FH_circuit(N, layers = 1):

    trotter_circuit = []
    for i in range(layers):
        trotter_step = []
        for i in range(0,16,2):
            trotter_step.append(('MG',K(np.random.random() * 2 * np.pi),(i,i+1)))
        for i in range(0,8,2):
            phi = np.random.random() * 2 * np.pi
            trotter_step.append(('CZ', UNL(0,0,-phi/4), (i, i+8)))
            # z_rots = rot(phi/4, phi/4)
            # trotter_step.append(('MG', z_rots, qubits))
        for i in range(1,7,2):
            trotter_step.append(('MG',iSWAP, (i,i+1)))
        for i in range(9,15,2):
            trotter_step.append(('MG', iSWAP, (i,i+1)))
        for i in range(2,7,2):
            phi = np.random.random() * 2 * np.pi
            trotter_step.append(('CZ', UNL(0,0,-phi/4), (i, i+8)))
            # z_rots = rot(phi/4, phi/4)
            # trotter_step.append(('MG', z_rots, qubits))
        phi = np.random.random() * 2 * np.pi
        trotter_step.append(('CZ', UNL(0,0,-phi/4), (7,15)))
        # z_rots = rot(phi/4, phi/4)
        # trotter_step.append(('MG', z_rots, (7,15)))
        for i in range(1,7,2):
            trotter_step.append(('MG', K(np.random.random() * 2 * np.pi),(i,i+1)))
        for i in range(9,15,2):
            trotter_step.append(('MG',K(np.random.random() * 2 * np.pi),(i,i+1)))
        trotter_circuit.extend(trotter_step)
    return trotter_circuit 


def random_noisy_circuit_test(N_gates, N_qubits, N_swaps, p):
    """
    Create random noisy circuit with pauli error having a p chance of occuring after every two qubit gate:
    """

    circuit = []

    nn_pairs = [(i,i+1) for i in range(N_qubits-1)]

    for i in range(N_gates):

        gate_type = 'MG'

        qubits = random.choice(nn_pairs)
          
        r1 = np.random.rand() * 2*np.pi
        r2 = np.random.rand() * 2*np.pi

        A1 = A(r1)
        A2 = A(r2)

        gate = G(A1,A2)

        gate_type = 'MG'
               
        circuit.append((gate_type, gate, qubits))

        pauli_error = create_pauli_error_string(N_qubits, p)

        circuit.append(('noise', pauli_error, 'all'))

    return circuit


def random_noisy_circuit(N_gates, N_qubits, N_swaps, p):
    """
    Create random noisy circuit with pauli error having a p chance of occuring after every two qubit gate:
    Parses the circuit to convert pauli string to matchgates. 
    """

    circuit = []

    nn_pairs = [(i,i+1) for i in range(N_qubits-1)]

    for i in range(N_gates):

        gate_type = 'MG'

        qubits = random.choice(nn_pairs)
          
        r1 = np.random.rand() * 2*np.pi
        r2 = np.random.rand() * 2*np.pi

        A1 = A(r1)
        A2 = A(r2)

        gate = G(A1,A2)

        gate_type = 'MG'
               
        circuit.append((gate_type, gate, qubits))

        pauli_error_string = create_pauli_error_string(N_qubits, p)

    return circuit




def givens_circuit(N):
    circuit1 = []
    circuit2 = []
    for i in range(N-2):
        A1 = A(np.random.rand() * 2*np.pi)
        A2 = A(np.random.rand() * 2*np.pi)
        circuit1.append(('MG', G(A1,A2), (i,i+1)))
    for i in range(N-2):
        A1 = A(np.random.rand() * 2*np.pi)
        A2 = A(np.random.rand() * 2*np.pi)
        circuit2.append(('MG', G(A1,A2), (i+1,i+2)))
    givens_circuit = [val for pair in zip(circuit1, circuit2) for val in pair]
    return givens_circuit

# def layered_circuit(N_gates, N, N_swaps, lower_index = 0, upper_index = 1, N_layers = 2):
#     SWAP = np.array([[1,0,0,0],
#                     [0,0,1,0],
#                     [0,1,0,0],
#                     [0,0,0,1]])
#     circuit = []
#     for i in range(N_layers):
#         circuit.extend(random_type_circuit(N_gates, N))
#     for j in range(N_swaps):

#         circuit.append(('SWAP', SWAP, (lower_index,upper_index)))
#         for i in range(N_layers*(j+1)):
#             circuit.extend(random_type_circuit(N_gates, N))

#     for i in range(N_layers):
#             circuit.extend(givens_circuit(N))

#     return(circuit)

def layered_circuit(N_gates, N, N_swaps, lower_index = 0, upper_index = 1, N_layers = 2):
    SWAP = np.array([[1,0,0,0],
                    [0,0,1,0],
                    [0,1,0,0],
                    [0,0,0,1]])
    circuit = []

    for j in range(N_swaps):
        # for i in range(N_layers):
        circuit.extend(random_type_circuit(N_gates, N))
        circuit.append(('SWAP', SWAP, (lower_index,upper_index)))


    circuit.extend(random_type_circuit(N_gates, N))
    return(circuit)

    
def sparse_layered_circuit(N, N_swaps, lower_index = 0, upper_index = 1, N_layers = 2):
    SWAP = np.array([[1,0,0,0],
                    [0,0,1,0],
                    [0,1,0,0],
                    [0,0,0,1]])
    circuit = []
    # for i in range(N_layers):
    circuit.extend(single_layer_givens_circuit(N))
    # circuit.extend(single_layer_givens_circuit(N))
    for j in range(N_swaps):
        circuit.append(('SWAP', SWAP, (lower_index,upper_index)))
        circuit.extend(single_layer_givens_circuit(N))

    return(circuit)
    # for i in range(N_layers):
    #         circuit.extend(single_layer_givens_circuit(N))


    


def clifford_circuit(N_gates, N, N_CZ, N_H):
    #Create a matchgate circuit containing a specified number of clifford gates.
    nn_pairs = [(i,i+1) for i in range(N-1)]
    qubit_list = [i for i in range(N)]
    circuit = []

    for i in range(N_gates): 
        gate_type = 'MG'

        qubits = random.choice(nn_pairs)
          
        r1 ,r2= np.random.rand() * 2*np.pi,  np.random.rand() * 2*np.pi
        
        A1, A2 = A(r1), A(r2)

        gate = G(A1,A2)

        circuit.append((gate_type, gate, qubits))

    for i in range(N_CZ):
        gate_type = 'CZ'

        qubits = random.choice(nn_pairs)

        gate = CZ

        circuit.append((gate_type, gate, qubits))

    for i in range(N_H):

        gate_type = 'H'

        qubits = random.choice(qubit_list)

        gate = H

        circuit.append((gate_type, gate, qubits))

    np.random.shuffle(circuit)   
  
    return circuit



# def givens_circuit(N):
#     circuit1 = []
#     circuit2 = []
#     for i in range(N-2):
#         # A1 = A(np.random.rand() * 2*np.pi)
#         A1 = A(np.pi/8)
#         circuit1.append(('MG', G(I,A1), (i+1,i+2)))
#     for i in range(N-2):
#         # A1 = A(np.random.rand() * 2*np.pi)
#         A1 = A(np.pi/8)
#         circuit2.append(('MG', G(I,A1), (i,i+1)))
#     givens_circuit1 = [val for pair in zip(circuit1, circuit2) for val in pair]
    
#     circuit3 = []
#     circuit4 = []
#     for i in range(N-2):
#         # A1 = A(np.random.rand() * 2*np.pi)
#         A1 = A(np.pi/8)
#         circuit3.append(('MG', G(I,A1), (i+1+N,i+2+N)))
#     for i in range(N-2):
#         A1 = A(np.pi/8)
#         # A1 = A(np.random.rand() * 2*np.pi)
#         circuit4.append(('MG', G(I,A1), (i+N,i+1+N)))
#     givens_circuit2 = [val for pair in zip(circuit3, circuit4) for val in pair]

#     givens_circuit = [val for pair in zip(givens_circuit1, givens_circuit2) for val in pair]

#     givens_circuit.insert(0,('MG', G(X,X), (0,1)))
#     givens_circuit.insert(1,('MG', G(X,X), (N,N+1)))
#     return givens_circuit

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

def single_layer_givens_circuit(N):
    circuit = []
    for i in range(N-1):
        A1 = A(np.random.rand() * 2*np.pi)
        # A1 = A(np.pi/8)
        circuit.append(('MG', G(I,A1), (i,i+1)))
    return circuit

def mod_trotter_circuit(N_sites, N_steps,N_swaps, J,U, tau):
    mod_trotter_circuit = []
    #Add initial Givens rotations to circuit
    mod_trotter_circuit.extend(givens_circuit(N_sites))
    for i in range(N_steps):
        trotter_step_i = mod_trotter_step(N_sites, N_swaps, J, U, tau)
        mod_trotter_circuit.extend(trotter_step_i)
    return mod_trotter_circuit

def mod_trotter_step(N_sites, N_swaps, J, U, tau):
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
    for i in range(N_swaps):
        phi = U*tau
        # z_rots = rot(phi/4, phi/4)
        trotter_step_list.append(('CZ', Cphase(phi), (i, i+N_sites)))
    return trotter_step_list    


def brickwall(N, depth):

    circuit = []

    for i in range(depth):
    

        for i in range(0,4*N,2): 
            qubits = (i, i+1)
            U = haar_random_MG()
            circuit.append(('MG', U, qubits))
        
        for i in range(1,4*N-1,2):
            qubits = (i, i+1)
            U = haar_random_MG()
            circuit.append(('MG', U, qubits))
    

    return circuit 

