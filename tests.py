from cmath import e
from distutils.log import error
from re import L, T
import cProfile
from sqlite3 import TimestampFromTicks
from matplotlib import offsetbox
from numpy import average
from simulator import Simulator
from simulator3 import Simulator3
from simulator4 import Simulator4
from simulator5 import Simulator5
import matplotlib.pyplot as plt
from circuit import *
import time 



# def len_L(N, N_swaps):
#     sum = 0
#     for i in range(N_swaps+1):
#         sum += comb(2*N, 2*i + 2)
#     return sum 






#Figure 6 (Interaction vs Heisenberg)
def test12():
    
    qubits = 10

    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize=(15,15))

    #Plot 1: Low Swaps O(1), Low gates ~ O(1)

    N,n,m = 10,qubits,3

    circuit = sparse_layered_circuit(N,n,m, N_layers = 1)

    types = np.array([i[0] for i in circuit[::-1]])
    swap_indices = np.where(types=='SWAP')

    simulator = Simulator3(N = n)
    simulator.simulate(circuit, verbose =  False)
    
    simulator.rho_lengths[-1] = simulator.msm_lengths[-1]

    lengths1 = simulator.msm_lengths + simulator.rho_lengths[::-1]


    simulator = Simulator2(N = n)
    simulator.simulate(circuit, verbose = False)
    

    lengths2 = simulator.msm_lengths 
    max_len = (len_L(n, m))
    plt.rc('axes', labelsize=12)  

    # for index in swap_indices:
    #     axs[0][0].axvline(x = index, ymin = 0, ymax = max_len, color = 'red', linestyle ='-')

    # for i in range(m):
    #     axs[0][0].axhline(y = len_L(n,i), xmin = 0, xmax = len(lengths1), color = 'red', linestyle = 'dashed')
    # plt.hlines(y = len_L(n_qubits,), xmin = 0, xmax = len(lengths), color = 'darkred', linestyles = 'dashed')
    axs[0][0].hlines(y = len_L(n,3), xmin = 0, xmax = len(lengths1), color = 'darkred', linestyles = 'dashed')

    axs[0][0].set_title(r'(a)')
    axs[0][0].plot(lengths1, color = 'blue', label ='Interaction')
    axs[0][0].plot(lengths2, color = 'red', label ='Heisenberg')
    axs[0][0].set_ylabel(r'Pauli rank',fontsize = 12)
    # axs[0][0].set_xlabel(r'Gate Number',fontsize = 12)
    axs[0][0].set_xticks(range(0,len(lengths1),int(len(lengths1)/10)))
    axs[0][0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    axs[0][0].fill_between(range(0,len(lengths1)),lengths1, alpha =0.75, color='blue')
    axs[0][0].fill_between(range(0,len(lengths2)),lengths2, alpha =0.5, color='red')
    # axs[0][0].legend()
    # plt.show()

    #Plot 2: Low Swaps O(1), High gates ~ O(N)

    
    N,n,m = 10,qubits,3

    circuit = sparse_layered_circuit(N,n,m, N_layers = 2)

    types = np.array([i[0] for i in circuit[::-1]])
    swap_indices = np.where(types=='SWAP')

    simulator = Simulator3(N = n)
    simulator.simulate(circuit, verbose =  False)
    
    simulator.rho_lengths[-1] = simulator.msm_lengths[-1]

    lengths1 = simulator.msm_lengths + simulator.rho_lengths[::-1]

    simulator = Simulator2(N = n)
    simulator.simulate(circuit, verbose = False)
    

    lengths2 = simulator.msm_lengths 


    max_len = (len_L(n, m))


    # for index in swap_indices:
    #     axs[0][0].axvline(x = index, ymin = 0, ymax = max_len, color = 'red', linestyle ='-')

    # for i in range(m):
    #     axs[0][0].axhline(y = len_L(n,i), xmin = 0, xmax = len(lengths1), color = 'red', linestyle = 'dashed')
    axs[0][1].hlines(y = len_L(n,3), xmin = 0, xmax = len(lengths1), color = 'darkred', linestyles = 'dashed')

    axs[0][1].set_title(r'(b)')
    axs[0][1].plot(lengths1, color = 'blue', label ='Interaction')
    axs[0][1].plot(lengths2, color = 'red', label ='Heisenberg')
    # axs[0][1].set_ylabel(r'Pauli rank',fontsize = 12)
    # axs[0][1].set_xlabel(r'Gate Number',fontsize = 12)
    axs[0][1].set_xticks(range(0,len(lengths1),int(len(lengths1)/10)))
    axs[0][1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[0][1].fill_between(range(0,len(lengths1)),lengths1, alpha =0.75, color='blue')
    axs[0][1].fill_between(range(0,len(lengths2)),lengths2, alpha =0.5, color='red')
    # axs[0][1].legend()
    # plt.show()
    #Plot 3: High Swaps O(n), Low gates  O(1)

    N,n,m = 10,qubits,3

    circuit = sparse_layered_circuit(N,n,m, N_layers = 3)

    types = np.array([i[0] for i in circuit[::-1]])
    swap_indices = np.where(types=='SWAP')

    simulator = Simulator3(N = n)
    simulator.simulate(circuit, verbose =  False)
    

    simulator.rho_lengths[-1] = simulator.msm_lengths[-1]
    lengths1 = simulator.msm_lengths + simulator.rho_lengths[::-1]

    simulator = Simulator2(N = n)
    simulator.simulate(circuit, verbose = False)
    

    lengths2 = simulator.msm_lengths 


    max_len = (len_L(n, m))

    # for index in swap_indices:
    #     axs[0][0].axvline(x = index, ymin = 0, ymax = max_len, color = 'red', linestyle ='-')

    # for i in range(m):
    #     axs[0][0].axhline(y = len_L(n,i), xmin = 0, xmax = len(lengths1), color = 'red', linestyle = 'dashed')
    axs[1][0].hlines(y = len_L(n,3), xmin = 0, xmax = len(lengths1), color = 'darkred', linestyles = 'dashed')

    axs[1][0].set_title(r'(c)')
    axs[1][0].plot(lengths1, color = 'blue', label ='Interaction' )
    axs[1][0].plot(lengths2, color = 'red', label ='Heisenberg')
    axs[1][0].set_ylabel(r'Pauli rank',fontsize = 12)
    axs[1][0].set_xlabel(r'Gate Number',fontsize = 12)
    axs[1][0].set_xticks(range(0,len(lengths1),int(len(lengths1)/10)))
    axs[1][0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[1][0].fill_between(range(0,len(lengths1)),lengths1, alpha =0.75, color='blue')
    axs[1][0].fill_between(range(0,len(lengths2)),lengths2, alpha =0.5, color='red')
    # axs[1][0].legend()
    # plt.show()

    #Plot 4: High Swaps O(1), High gates O(N)

    
    N,n,m = 10,qubits,3

    circuit = sparse_layered_circuit(N,n,m, N_layers = 4)

    types = np.array([i[0] for i in circuit[::-1]])
    swap_indices = np.where(types=='SWAP')

    simulator = Simulator3(N = n)
    simulator.simulate(circuit, verbose =  False)
    

    simulator.rho_lengths[-1] = simulator.msm_lengths[-1]
    lengths1 = simulator.msm_lengths + simulator.rho_lengths[::-1]

    simulator = Simulator2(N = n)
    simulator.simulate(circuit, verbose = False)
    

    lengths2 = simulator.msm_lengths 


    max_len = (len_L(n, m))
   

    # for index in swap_indices:
    #     axs[0][0].axvline(x = index, ymin = 0, ymax = max_len, color = 'red', linestyle ='-')

    # for i in range(m):
    #     axs[0][0].axhline(y = len_L(n,i), xmin = 0, xmax = len(lengths1), color = 'red', linestyle = 'dashed')
    axs[1][1].hlines(y = len_L(n,3), xmin = 0, xmax = len(lengths1), color = 'darkred', linestyles = 'dashed')

    axs[1][1].set_title((r'(d)'))
    axs[1][1].plot(lengths1, color = 'blue', label ='Interaction')
    axs[1][1].plot(lengths2, color = 'red', label ='Heisenberg')
    # axs[1][1].set_ylabel(r'Pauli rank',fontsize = 12)
    axs[1][1].set_xlabel(r'Gate Number',fontsize = 12)
    axs[1][1].set_xticks(range(0,len(lengths1),int(len(lengths1)/10)))
    axs[1][1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axs[1][1].fill_between(range(0,len(lengths1)),lengths1, alpha =0.75, color='blue')
    axs[1][1].fill_between(range(0,len(lengths2)),lengths2, alpha =0.5, color='red')
    # axs[1][1].legend()
    
    # plt.show()
    # fig.suptitle('Simulation profile for different saturations')
    plt.show()

#Figure 4
def test13():
    from simulator3 import Simulator3

    N = 40
    n = 20
    n_swaps = 5

    circuit = layered_circuit(N, n, n_swaps)
    types = np.array([i[0] for i in circuit[::-1]])
    # print(types)
    swap_indices = np.where(types=='SWAP')[0]
    # print(swap_indices)

    simulator = Simulator3(n)
    
    simulator.simulate(circuit, verbose = True)


    # # for index in swap_indices:
    # #     plt.vlines(x = index, ymin = 0, ymax = max_len, linestyles ='dashed')

    # # for i in range(n_swaps):
    # #     plt.hlines(y = len_L(n,i), xmin = 0, xmax = len(simulator.msm_lengths), color = 'red', linestyles = 'dashed')
    # # plt.hlines(y = len_L(2*n_sites,10), xmin = 0, xmax = len(simulator.lengths), color = 'darkred', linestyles = 'dashed')
    
    np.save('staircase.npy', simulator.msm_lengths)
    
    # print(len(simulator.msm_lengths))
   

def plot_staircase():

    N = 60
    n = 31
    n_swaps = 10
    circuit = layered_circuit(N, n, n_swaps)
    types = np.array([i[0] for i in circuit[::-1]])
    swap_indices = np.where(types=='SWAP')[0]
    values = [len_L(n,i) for i in range(len(swap_indices)+1)]
    edges = [0] + list(swap_indices) + [len(circuit)]
    plt.stairs(values = values, edges = edges, fill = True, alpha = 0.3)
    lengths = np.load('staircase.npy')
    plt.plot(lengths)
    plt.ylabel(r'Pauli rank',fontsize=12)
    plt.xlabel(r'Gate Number',fontsize=12)
    plt.xticks(range(0,len(lengths),int(len(lengths)/10)))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.show()

# # plot_staircase()

# def bounds():

#     N = 7

#     orbit_sizes= []
#     bound_sizes = []

#     def chi(n,m):
#         return  sp.special.comb(2*n,2*m +2) * (((2*n - 2*m - 1)**2) / ((2*n - 2*m -1)**2 - (2*m +2)**2))

#     def bound(n,m):
#         mid = np.floor((n-2) / 2)
#         if m > mid:
#             return 2*chi(n,mid) - 2*chi(n,n - 3 - m)

#         else: return chi(n,m)
#     r  = 8
#     for i in range(r+1):
#         orbit_sizes.append(len_L(N,i))
#         bound_sizes.append(bound(N,i))

#     plt.rc('axes', labelsize=13)  
#     # plt.title(r"$\chi_{max}(n,m)$ as a function of $m$, $n = 7$")
#     plt.scatter(range(r+1),orbit_sizes, c= 'blue', marker='x', label = r'$\chi_{exact}$')
#     plt.scatter(range(r+1),bound_sizes, c= 'red', marker='x', label = r'$\chi_{bound}$')
#     plt.xlabel(r"$m$")
#     plt.ylabel(r" $\chi_{max}(n,m)$")
#     plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#     plt.legend()
#     plt.show()

# def get_indices(n):

#     Z_indices = [3 * 4**i for i in range(n)]
#     indices = []

#     for i in range(n+1):
#         indices.extend([sum(j) for j in itertools.combinations(Z_indices,i)])       


#     return indices

# def test_Z(n):

#     indices = get_indices(n)
#     # print(indices)
#     results = []
#     N,n,N_swaps = 10,n,0
#     circuit = random_type_circuit(N,n,N_swaps)

#     for index in indices:
#         measurement_vector = {index:1.0}
#         simulator = Simulator(n)
#         res = simulator.simulate(circuit, measurement_vector=measurement_vector, verbose = False)
#         results.append(np.round(res,7))
#     # print(results)
#     return results

# test_Z(4)

# def timevspauli():
#     # simulator = Simulator(2)
#     # circuit = sparse_layered_circuit(1,2,0,N_layers=1)
#     # simulator.simulate(circuit,verbose=True)
#     times = []
#     lengths = []
#     for i in [1]:
#         for j in [2,3]:
#             for k in [0,1]:
#                 # print(i,j,k)
#                 circuit = layered_circuit(j,k,N_layers=i)

#                 simulator = Simulator3(j)

#                 # s = time.time()
#                 simulator.simulate(circuit,verbose=True)
#                 # e = time.time() - s
#                 # length = sum(simulator.msm_lengths) + sum(simulator.rho_lengths)
#                 length = sum(simulator.lengths)
#                 lengths.append(length)
#                 times.append(simulator.time)


#     # print(times)
#     # print(lengths)

#     np.save('times.npy',times)
#     np.save('lengths.npy',lengths)

# def load():

#     times = np.load('times.npy') 
#     lengths = np.load('lengths.npy')

#     # print('times', times)

#     zeros= np.array([index for index,i in enumerate(times) if i == 0])

#     times = np.delete(times,zeros)
#     lengths = np.delete(lengths,zeros)

#     a= np.log10(lengths)
#     b=np.log10(times)
#     m,c = np.polyfit(lengths, times,1)

#     # print(m,c)

#     lengthsx = np.logspace(np.log10(min(lengths)),np.log10(max(lengths)), 500)

#     y_fit = np.exp(m*np.log(lengthsx) +  np.log(10)*c)
#     # y_fit = np.exp(m*timesx + c)
#     plt.rc('axes', labelsize=12)  
#     plt.scatter(lengthsx, y_fit, s=2, marker = '_',c='darkred')
#     plt.scatter(lengths, times, s = 7, marker = 'x', c='red')
#     plt.xlabel(r'Total Pauli Rank')
#     plt.ylabel(r'Wall Clock Time (s)')
#     plt.yscale('log')
#     plt.xscale('log')
    
    # font = {'family' : 'normal',
    #     'weight' : 'bold',
    #     'size'   : 100}

    # matplotlib.rc('font', **font)
    # plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plt.show()

# Figure 8
def test11(save_name, n=5):

    
    thresholds = [0,1e-10,1e-8,1e-6,1e-4,1e-2]
    depths = [0,1,2,3,4,5]
    data = []
    times = []
    for d in depths: 
        circuit = mod_trotter_circuit(n,d,1,0.3,0.9,1)
        temps0,  temps2 = [], []
        for i,threshold in enumerate(thresholds):
            simulator = Simulator4(2*n)
            simulator.simulate(circuit[:10], verbose = False,threshold=threshold)
            
            print(d,i)
            simulator = Simulator4(2*n)
            t = time.process_time()
            simulator.simulate(circuit, verbose = True,threshold=threshold)
            wct = time.process_time() - t + 1e-3

            temps0.append(wct/int(len(circuit)))
            temps2.append((sum(simulator.msm_lengths) + sum(simulator.rho_lengths))/int(len(circuit)))
        times.append(temps0)
        data.append(temps2)

    np.save(save_name+'.npy', data)
    np.save(save_name+'times'+'.npy', times)

import scipy as sp
# test11('MG15r1', n = 15) 

def layered_scaling(n,m):
    c = 1/(m+1)
    s=0
    for i in range(1,m+2):
        s += i * comb(2*n,((2*m)+4)-2*i)
    return c*s

def layered_bound(n,m):
    m_mid = np.floor(n/2) - 1
    if m < m_mid: 
        from scipy.special import comb
        c = 1/(m+1)
        coeff = comb(2*n,2*m+2)
        r = (((2*m)+2)/((2*n)-(2*m)-1))**2
        t = 1/((1-r)**2)
        return c*coeff*t
    if m >= m_mid:
        return layered_bound_largem(n,m)

def layered_bound_largem(n,m):
    m_mid = np.floor(n/2) - 1
    phi = lambda x:  sp.stats.norm.cdf(x)
    r = (((2*(m_mid-1))+2)/((2*n)-(2*(m_mid-1))-1))**2
    t1 = 1/(m+1) * 2**((2*n)-1) * (m-m_mid+1) * phi(np.sqrt(8/n)*((m-m_mid)/2)**2)
    t2 = 1/(m+1) * 1/((1-r)**2) * comb(2*n, 2*m_mid-2)
    return t1 + t2
    
def bound_layered3(n,m):
    m_mid = np.floor(n/2) - 1
    # alpha = np.sqrt(2/n)*(m-m_mid)
    alpha = np.sqrt(8/n)*((m-m_mid)+0.5)
    # phi = lambda x: (1 + sp.special.erf(x*2))/2
    phi = lambda x:  sp.stats.norm.cdf(x)

    return 2**(2*n-1+np.log2(phi(alpha)))

def general_bound(n,m):
    m_mid = int(np.floor(n/2) - 1)
  
    if m < m_mid: 
        coeff = comb(2*n,2*m+2)
        r = (((2*m)+2)/((2*n)-(2*m)-1))**2
        t = 1/(1-r)
        return coeff*t

    elif m >= m_mid: 
        # print(bound_layered3(n,m)/general_scaling(n,m))
        return(bound_layered3(n,m))

    # else: return float(2**(2*n-2))



def plot_Figure8():
    fig, axs = plt.subplots(nrows=3, ncols=2)
    depths5 = np.array([0,1,2,3,4,5,6,7,8,9,10],dtype=int)
    depths10 = np.array([0,1,2,3,4,5,6],dtype=int)
    depths15 = np.array([0,1,2,3,4],dtype=int)

    tols = [0,1e-10,1e-8,1e-6,1e-4,1e-2]
    tols = [0,-10,-8,-6,-4,-2]
    # ranks5= np.load('MG5r0.npy')
    # times5= np.load('MG5r0times.npy')
    # ratios = np.divide(np.array(times5),np.array(ranks5))
    # print(np.average(ratios), np.std(ratios))
    # ranks10= np.load('MG10r0.npy')  
    # times10= np.load('MG10r0times.npy')
    # ratios1 = np.divide(np.array(times10),np.array(ranks10))
    # print(np.average(ratios1), np.std(ratios1))

    # ranks5, times5 = np.load('MG5r1.npy'), np.load('MG5r1times.npy')
    ranks5 = sum([np.load('MG5r'+ str(r) + '.npy') for r in range(3)])/3 
    times5 = sum([np.load('MG5r'+ str(r) + 'times.npy') for r in range(3)])/3 + (0.001)
    times5[2][0] = 2e-3
    # ranks10, times10 = np.load('MG10.npy'), np.load('MG10times.npy')
    ranks10 = sum([np.load('MG10r'+ str(r) + '.npy') for r in range(2)])/2
    times10 = sum([np.load('MG10r'+ str(r) + 'times.npy') for r in range(2)])/2

    ranks15 = sum([np.load('MG15r'+ str(r) + '.npy') for r in range(1)])/1
    times15 = sum([np.load('MG15r'+ str(r) + 'times.npy') for r in range(1)])/1
    
    print(times15)

    for i, rank in enumerate(ranks5.T):
        axs[0][0].plot(depths5, rank, label = r'$\epsilon = %s$'%(str(tols[i])))

    # axs[0][0].set_ylabel('Total Pauli rank (per gate)',size=12)
    # axs[0][0].set_xlabel('Trotter steps', size=12)
    axs[0][0].set_xticks(depths5, size = 14)
    axs[0][0].set_yticks(depths5, size = 14)
    axs[0][0].grid(True)
    axs[0][0].set_yscale('log')
    axs[0][0].set_title(r'$n_{sites} = 5$', size = 14)
    axs[0][0].legend(loc = 2, ncol = 2, fontsize = 'small', title = 'Simulation Method')
    lens = [len(mod_trotter_circuit(10,d,1,0.3,0.9,1)) for d in range(10)]
    
    axs[0][0].plot([layered_scaling(10,i)for i in range(11)], c= 'red', linestyle = '--', label= r'$\chi_{t}^{(bound)}$')

    for i, rank in enumerate(ranks10.T):
        axs[1][0].plot(depths10, rank, label = 'MG' + str(tols[i]))
    axs[1][0].set_ylabel(r'Total Pauli rank (per gate), $\frac{\chi_{t}}{N}$',size=14)
    # axs[1][0].set_xlabel('Trotter steps', size=12)
    axs[1][0].set_xticks(depths10)
    axs[1][0].set_title(r'$n_{sites} = 10$', size = 14)
    axs[1][0].grid(True)
    axs[1][0].set_yscale('log')
    # axs[1][0].legend(loc = 0, ncol = 2, fontsize = 'small', title = 'Simulation Method')
    axs[1][0].plot([layered_scaling(20,i)for i in range(7)], c= 'red', linestyle = '--')
   
    for i, rank in enumerate(ranks15.T):
        axs[2][0].plot(depths15, rank, label = 'MG' + str(tols[i]))
    # axs[2][0].set_ylabel('Total Pauli rank (per gate)',size=12)
    axs[2][0].set_xlabel(r'Trotter steps', size=14)
    axs[2][0].set_xticks(depths15)
    axs[2][0].grid(True)
    axs[2][0].set_title(r'$n_{sites} = 15$', size = 14)
    axs[2][0].set_yscale('log')
    # axs[2][0].legend(loc = 0, ncol = 2, fontsize = 'small', title = 'Simulation Method')
    axs[2][0].plot([layered_scaling(30,i)for i in range(5)], c= 'red', linestyle = '--')
    
    for i, time in enumerate(times5.T):
        axs[0][1].plot(depths5, time, label = 'MG' + str(tols[i]))
    # axs[0][1].set_ylabel('Total time taken (per gate)',size=12)
    # axs[0][1].set_xlabel('Trotter steps', size=12)
    axs[0][1].set_xticks(depths5)
    axs[0][1].grid(True)
    axs[0][1].set_title(r'$n_{sites} = 5$', size = 14)
    axs[0][1].set_yscale('log')
    # axs[0][1].legend(loc = 0, ncol = 2, fontsize = 'small', title = 'Simulator Method')


    for i, time in enumerate(times10.T):
        axs[1][1].plot(depths10, time, label = 'MG' + str(tols[i]))

    axs[1][1].set_ylabel(r'Total time taken (per gate) (s)',size=14)
    # axs[1][1].set_xlabel('Trotter steps', size=12)
    axs[1][1].set_xticks(depths10)
    axs[1][1].grid(True)
    axs[1][1].set_yscale('log')
    axs[1][1].set_title(r'$n_{sites} = 10$', size = 14)
    # axs[1][1].legend(loc = 0, ncol = 2, fontsize = 'x-large', title = 'Simulation Method', title_fontsize = 'x-large')
    axs[1][1].tick_params(axis='both', which='major', labelsize=14)
    axs[1][1].tick_params(axis='both', which='minor', labelsize=14)

    for i, time in enumerate(times15.T):
        axs[2][1].plot(depths15, time, label = 'MG' + str(tols[i]))

    # axs[2][1].set_ylabel('Total time taken (per gate)',size=12)
    axs[2][1].set_xlabel(r'Trotter steps', size=14)
    axs[2][1].set_xticks(depths15)
    axs[2][1].grid(True)
    axs[2][1].set_title(r'$n_{sites} = 15$', size = 14)
    axs[2][1].set_yscale('log')
    # axs[2][1].legend(loc = 0, ncol = 2, fontsize = 'x-large', title = 'Simulation Method', title_fontsize = 'x-large')
    axs[2][1].tick_params(axis='both', which='major', labelsize=14)
    axs[2][1].tick_params(axis='both', which='minor', labelsize=14)
    # fig.suptitle('Pauli Rank/Time versus Trotter Step')

    # plt.text(1,1, 'ddd',rotation = 'vertical', size = 13)

    plt. subplots_adjust(hspace=0.5)   
    plt.show()




from numba.typed import Dict
from numba.core import types

#Figure 9 
def error_vs_rank():

    n_sites = [2,3,4,5,6,7,8,9,10]
    depths = [0,1,2,3,4,5,6]
    thresholds = [1e-2,1e-4,1e-6,1e-8]

    deltas = []
    ranks = []

    for n in n_sites:
        for d in depths:

            circuit = mod_trotter_circuit(n,d,1,0.3,0.9,1)
            simulator = Simulator4(2*n)
            measurement_vector =  Dict.empty(
                                key_type=types.int64,
                                value_type=types.float64)

        
            true_res = simulator.simulate(circuit, verbose =True)
            ranks.append((sum(simulator.msm_lengths) + sum(simulator.rho_lengths))/len(circuit))

            delta = []
            for t in thresholds: 
                # print(n,d,t)
                simulator = Simulator4(2*n)
                approx_res = simulator.simulate(circuit, threshold = t, verbose = True)
                delta.append(np.abs(approx_res - true_res))
                # delta.append(np.abs(approx_res))

            deltas.append(delta)

    lines = np.array(deltas).T
    np.save('lines2.npy', lines)
    np.save('ranks2.npy', ranks)


def plot_error_vs_rank():
    thresholds = [1e-2,1e-4,1e-6,1e-8]
    lines = np.concatenate((np.load('lines.npy'), np.load('lines1.npy')), axis = 1)
    ranks = np.concatenate((np.load('ranks.npy'), np.load('ranks1.npy')), axis =0)
    # print(np.shape(np.load('ranks1.npy')))
 
    for i,line in enumerate(lines): 
        mask = np.where(line > 10**-13)
        line = line[mask]
        temp = ranks[mask] 
                
        plt.scatter(temp, line, label = '%s'%thresholds[i], s = 50, marker = '.')
               
        m,b = np.polyfit(np.log10(temp),np.log10(line), deg= 1)

        x = (np.logspace(0, np.log10(max(temp)), 1000, base = 10))
        y = 10**b*(x**m)
    
        plt.plot(x,y)

    # plt.title('Absolute error versus Total Pauli Rank', size = 12)
    plt.ylabel(r"Absolute Error $\delta$",size=15)
    plt.xlabel(r"Total Pauli rank (per gate) $\chi_{tot}/N$", size=15)
    plt.xticks(ranks)
    plt.tick_params(axis = 'both', which='major', labelsize=15)
    plt.tick_params(axis = 'both', which='minor', labelsize=15)
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 'lower right', ncol = 2, fontsize = 'x-large', title = r'Threshold, $\epsilon$', title_fontsize ='x-large')
    plt.show()


def bound_layered3(n,m):
    m_mid = np.floor(n/2) - 1
    # alpha = np.sqrt(2/n)*(m-m_mid)
    alpha = np.sqrt(8/n)*((m-m_mid)+0.5)
    # phi = lambda x: (1 + sp.special.erf(x*2))/2
    phi = lambda x:  sp.stats.norm.cdf(x)

    return 2**((2*n)-1+np.log2(phi(alpha)))

def general_scaling(n,m):
    s=0
    for i in range(0,m+1):
        s += comb(2*n,2*i + 2)
    return s
    
from scipy.special import comb
import scipy as sp
def general_bound(n,m):
    m_mid = int(np.floor(n/2) - 1)
  
    if m < m_mid: 
        coeff = comb(2*n,2*m+2)
        r = (((2*m)+2)/((2*n)-(2*m)-1))**2
        t = 1/(1-r)
        return coeff*t

    elif m >= m_mid: 
        # print(bound_layered3(n,m)/general_scaling(n,m))
        return(bound_layered3(n,m))

    else: return float(2**(2*n-2))

def general_bound_largem(n,m):
    import scipy as sp
    m2 = n - 2 - m
    if m2 > 0:
        # return 2**(2*n-1) * (1 - 2**(2*m2*np.log2(n/m2)-2*n + 1))
        r = (((2*m2)+2)/((2*n)-(2*m2)-1))**2
        return float(2**(2*n-1) - sp.special.comb(2*n,2*m2))
    if m2 <= 0:
        return float(2**(2*n-1))

# print([np.log2(general_bound(30,i)) for i in range(0,30)])

def layered_scaling(n,m):
    c = 1/(m+1)
    s=0
    for i in range(1,m+2):
        s += i * comb(2*n,((2*m)+4)-2*i)
    return c*s

def layered_bound(n,m):
    m_mid = np.floor(n/2) - 1
    if m <= m_mid: 
        from scipy.special import comb
        c = 1/(m+1)
        coeff = comb(2*n,2*m+2)
        r = (((2*m)+2)/((2*n)-(2*m)-1))**2
        t = 1/((1-r)**2)
        return c*coeff*t
    if m > m_mid:
        return layered_bound_largem(n,m)

def layered_bound_largem(n,m):
    m_mid = np.floor(n/2) - 1
    phi = lambda x:  sp.stats.norm.cdf(x)
    r = (((2*(m_mid-1))+2)/((2*n)-(2*(m_mid-1))-1))**2
    l = []
    
    for i in range(0,int(m-m_mid)):
        l.append(phi(np.sqrt(8/n)*(i+0.5)))
    print((sum(l)/(m-m_mid))*(1/(m+1)))
    # t1 = (1/(m+1)) * 2**((2*n)-1) * sum(l)
    t2 = (1/(m+1)) * 1/((1-r)**2) * comb(2*n, 2*m_mid)
    t1 = 1/(m+1) * (m- m_mid + 1) * 2**((2*n) - 1) * phi(np.sqrt(8/n)*((m-m_mid+1)/4 +0.5))
    return t1 + t2


#Figure 5

# for n in range(3,30,4):
#     plt.plot([np.log2(general_scaling(n,m)) for m in range(0,15)], c = 'black')
#     plt.plot([np.log2(general_bound(n,m)) for m in range(0,15)], c = 'blue',ls='--')
# plt.scatter([(int(np.floor(n/2))-1) for n in range(3,30,4)],[np.log2(general_scaling(n,int(np.floor(n/2))-1)) for n in range(3,30,4)], s= 10, c = 'red')
# plt.plot([(int(np.floor(n/2))-1) for n in range(3,30,4)],[np.log2(general_scaling(n,int(np.floor(n/2))-1)) for n in range(3,30,4)], c='red',linestyle='--', label=r'$m_{c}$')
# plt.text(13.5, 5.5, r'n=3', fontsize=14)
# plt.text(13.5, 13.5, r'n=7', fontsize=14)
# plt.text(13.5, 21.5, r'n=11', fontsize=14)
# plt.text(13.5, 29.5, r'n=15', fontsize=14)
# plt.text(13.5, 37.5, r'n=19', fontsize=14)
# plt.text(13.5, 45.5, r'n=23', fontsize=#14)
# plt.text(13.5, 53.5, r'n=27', fontsize=14)

# handles, labels = plt.gca().get_legend_handles_labels()

# from matplotlib.lines import Line2D
# import matplotlib.patches as mpatches

# line1= Line2D([0], [0], label=r'$\chi^{(general)}_{t}$', color='black')
# line2 = Line2D([0], [0], label=r'$\chi^{(bound)}_{t}$', color='blue',ls='--')
# handles.extend([line1,line2])

# plt.legend(handles=handles,fontsize=12)
# plt.xticks(size=14)
# plt.yticks(size=14)
# plt.xlabel(r'Number of non-matchgates, $m$',fontsize= 14)
# plt.ylabel(r'Total Pauli Rank, $log(\chi_{t})$',fontsize= 14)
# plt.show()


# for n in range(3,100,4):
#     plt.plot([np.log2(layered_scaling(n,m)) for m in range(0,n-2)], c = 'red')
#     plt.plot([np.log2(layered_bound(n,m)) for m in range(0,n-2)], c = 'blue')
# plt.scatter([(int(np.floor(n/2))-1) for n in range(3,60,4)],[np.log2(layered_scaling(n,int(np.floor(n/2))-1))for n in range(3,60,4)])
# plt.show()






























# def plot_bounds():
#     fig, axs = plt.subplots(nrows = 3, ncols = 1)
#     for n in range(11,12):
#         axs[0].plot([np.log2(bound_true(n,i)) for i in range(10,12)], c = 'red')
#         axs[0].scatter([(int(np.floor(n/2))-1)],[np.log2(bound_true(n,int(np.floor(n/2))-1))])

#     for m in range(0,50):
#         axs[1].scatter([n for n in range(2,150)],[np.log2(bound_true(n,m)) for n in range(2,150)], c = 'red')
#     axs[1].scatter([n for n in range(2,150)],[n for n in range(2,150)])

#     for m in range(0,50):
#         axs[2].scatter([n for n in range(2,150)],[np.log2(bound_true2(n,m)) for n in range(2,150)], c = 'red')
#     axs[2].scatter([n for n in range(2,150)],[1.5*n for n in range(2,150)])
#     plt.show()

# def chi_vs_n(n_range):
#     data = []
#     for n in [2,4,6,8,10]: 
#         tots = []
#         for d in [1,2,3,4,5]:
#             circuit = mod_trotter_circuit(n,d,1,0.3,0.9,1)
#             simulator = Simulator4(2*n)
#             simulator.simulate(circuit, verbose = True)
#             tot = (sum(simulator.msm_lengths)+sum(simulator.rho_lengths))/len(circuit)
#             tots.append(tot)
#         data.append(tots)
#     np.save('graph.npy', data)
       
# def plyt():
#     data = np.load('graph.npy')
#     print(data)
#     for i in data:
#         plt.plot(np.log10([1,2,3,4,5]),np.log10(i))
#         plt.xlabel('m')
#     plt.show()

#     for i in data.T:
#         plt.scatter(np.log10([4,8,12,16,20]),np.log10(i))
#         plt.xlabel('n')
#     plt.show()

# def even(n):
#     from scipy.stats import norm
#     l = []
#     g= []

#     for i in range (0,2*n+1,2):
#         l.append(comb(2*n,i)/2**(2*n-1))
#     for i in range (0,2*n+1):
#         g.append(comb(2*n,i)/2**(2*n))
    

#     x = np.arange(0, 2*n, 1000)

    # print(norm.pdf(x, 0.25*n, np.sqrt(n)/4))
    # print(np.array(l))
    
    # print(np.divide(norm.pdf(x, 0.5*n, np.sqrt(n)/4), np.array(l)))

#plot normal distribution with mean 0 and standard deviation 1
    # plt.plot(x, norm.pdf(x, n, np.sqrt(n/2)), c ='green')  
    # plt.plot(x, norm.pdf(x, 0.5*n, np.sqrt(n/8)), c ='pink') 
    # plt.plot(l, c= 'blue')
    # plt.plot(g, c='red')
    # plt.show()

# even(500)

# def test4():
#     J = 0.3
#     U = 0.9
#     tau = float(0.3/J)

#     tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
#     repeats = 15

#     #Init dual axes
#     fig, axs = plt.subplots(2,2, figsize = (12,12))
#     fig.tight_layout(pad=8.0)
#     twin_axes = [axs[0][0].twinx(), axs[0][1].twinx(), axs[1][0].twinx(),axs[1][1].twinx()]

#     #(row,column, n_sites, n_steps)
#     params = [(0,0,4,5), (0,1,4,10), (1,0,4,15), (1,1,4,20)]

#     for param in params:
#         #Times
#         times , times_max , times_min , times_std = [],[],[],[]
#         accuracies , accuracies_max,accuracies_min,accuracies_std = [],[],[],[]
#         paulis , paulis_max ,paulis_min ,paulis_std = [],[],[],[]

#         for i, tol in enumerate(tols):
#             #Create temporary variables for storing repeats
#             temp1 ,temp2 ,temp3 = [],[],[]
#             for n in range(repeats):
#                 #Init Trotter Circuit
#                 circuit = mod_trotter_circuit(N_sites=param[2], N_steps =param[3],N_swaps=1, J = J, U = U, tau = tau)
#                 print('tol', tol, 'iter', i, 'repeat', n)

#                 simulator = Simulator4(2*param[2])
#                 hard_res = simulator.simulate(circuit)

#                 #Run experiement 
#                 simulator = Simulator4(2*param[2])
#                 start = time.time()
#                 res = simulator.simulate(circuit,threshold = tol)
#                 time_taken = time.time() - start

#                 #Append raw_results
#                 temp1.append(time_taken)
#                 temp2.append(np.abs(res - hard_res))
#                 temp3.append(sum(simulator.msm_lengths)+sum(simulator.rho_lengths))

#             times.append(np.average(temp1))
#             times_max.append(max(temp1))
#             times_min.append(min(temp1))
#             times_std.append(np.std(temp1))
        
#             accuracies.append(np.average(temp2))
#             accuracies_max.append(max(temp2))
#             accuracies_min.append(min(temp2))
#             accuracies_std.append(np.std(temp2))

#             paulis.append(np.average(temp3))
#             paulis_max.append(max(temp3))
#             paulis_min.append(min(temp3))
#             paulis_std.append(np.std(temp3))

#         st = 'Trotter circuit: %s sites, %s steps, %s repeats' %(param[2], param[3], repeats)
#         axs[param[0]][param[1]].set_title(st)
#         ax1 = twin_axes[2*param[0]+ 1*param[1]]
#         axs[param[0]][param[1]].errorbar(np.log10(tols),np.log10(accuracies), yerr =  0.430*np.divide(accuracies_std,accuracies), color = 'blue',  ecolor = 'blue', linestyle='',fmt='o',alpha=0.4)
#         axs[param[0]][param[1]].scatter(np.log10(tols),np.log10(accuracies_max),c = 'blue',s = 5,marker = 'x')
#         axs[param[0]][param[1]].scatter(np.log10(tols),np.log10(accuracies_min), c = 'blue',s = 5,marker = 'x')

#         ax1.errorbar(np.log10(tols),np.log10(paulis), yerr = 0.430*np.divide(paulis_std,paulis),color = 'red', ecolor = 'red', linestyle='',fmt='o',alpha=0.3)
#         ax1.scatter(np.log10(tols),np.log10(paulis_max), c= 'red',s = 5,marker = 'x')
#         ax1.scatter(np.log10(tols),np.log10(paulis_min), c= 'red',s = 5, marker = 'x')
    
#         axs[param[0]][param[1]].set_xlabel('log(Tolerance)')
#         axs[param[0]][param[1]].set_ylabel('log(Absolute error)', color='b')
#         ax1.set_ylabel('log(Simulation Time)', color='r')

#     plt.show()

# def sum_of_cdfs(n,j):
#     s = 0
#     for i in range(j):
#         s += sp.stats.norm.cdf(4*(np.sqrt(2/n)*i))
#     return s
# def sum_of_cdfs2(n,j):    
#     t = j * sp.stats.norm.cdf(sum([4*np.sqrt(2/n)*i for i in range(j)])/j)
#     return t
    
# print([sum_of_cdfs(10,j) for j in range(1,10)])    
# print([sum_of_cdfs2(10,j) for j in range(1,10)])
