# -------- Start of the importing part -----------
from numba import cuda, jit, int32, float32, int64
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32, xoroshiro128p_normal_float32
import cupy as cp
from math import pow, hypot, ceil
from timeit import default_timer as timer
import numpy as np
import random
import sys
import os
import json
from GARL import RLagent
import val as validateSolution
import shutil
np.set_printoptions(threshold=sys.maxsize)
import gpuGrid
import val
# -------- End of the importing part -----------

# ------------------------- Start reading the data file -------------------------------------------
class vrp():
    def __init__(self, capacity=0, opt=0):
        self.capacity = capacity
        self.opt = opt
        self.nodes = np.zeros((1,4), dtype=np.float32)
    def addNode(self, label, demand, posX, posY):
        newrow = np.array([label, demand, posX, posY], dtype=np.float32)
        self.nodes = np.vstack((self.nodes, newrow))

def readInput():

	# Create VRP object:
    vrpManager = vrp()
	# First reading the VRP from the input #
    print('Reading data file...', end=' ')
    text_out = open('1000.out', 'a')
    print('Reading data file...', end=' ', file=text_out)
    text_out.close()
    fo = open('../test_set/%s/'%sys.argv[1][0]+sys.argv[1]+'.vrp',"r")
    lines = fo.readlines()
    for i, line in enumerate(lines):       
        while line.upper().startswith('COMMENT'):
            if len(sys.argv) <= 3:
                inputs = line.split()
                if inputs[-1][:-1].isnumeric():
                    vrpManager.opt = np.int32(inputs[-1][:-1])
                    break
                else:
                    try:
                        vrpManager.opt = float(inputs[-1][:-1])
                    except:
                        print('\nNo optimal value detected, taking optimal as 0.0')
                        text_out = open('1000.out', 'a')
                        print('\nNo optimal value detected, taking optimal as 0.0', file=text_out)
                        text_out.close()
                        vrpManager.opt = 0.0
                    break
            else:
                vrpManager.opt = np.int32(sys.argv[3])
                print('\nManual optimal value entered: %d'%vrpManager.opt)
                text_out = open('1000.out', 'a')
                print('\nManual optimal value entered: %d'%vrpManager.opt, file=text_out)
                text_out.close()
                break

        # Validating positive non-zero capacity
        if vrpManager.opt < 0:
            print(sys.stderr, "Invalid input: optimal value can't be negative!")
            text_out = open('1000.out', 'a')
            print(sys.stderr, "Invalid input: optimal value can't be negative!", file=text_out)
            text_out.close()
            exit(1)
            break

        while line.upper().startswith('CAPACITY'):
            inputs = line.split()
            vrpManager.capacity = np.float32(inputs[2])
			# Validating positive non-zero capacity
            if vrpManager.capacity <= 0:
                print(sys.stderr, 'Invalid input: capacity must be neither negative nor zero!')
                text_out = open('1000.out', 'a')
                print(sys.stderr, 'Invalid input: capacity must be neither negative nor zero!', file=text_out)
                text_out.close()
                exit(1)
            break       
        while line.upper().startswith('NODE_COORD_SECTION'):
            i += 1
            line = lines[i]
            while not (line.upper().startswith('DEMAND_SECTION') or line=='\n'):
                inputs = line.split()
                vrpManager.addNode(np.int16(inputs[0]), 0.0, np.float32(inputs[1]), np.float32((inputs[2])))
                # print(vrpManager.nodes)
                i += 1
                line = lines[i]
                while (line=='\n'):
                    i += 1
                    line = lines[i]
                    if line.upper().startswith('DEMAND_SECTION'): break 
                if line.upper().startswith('DEMAND_SECTION'):
                    i += 1
                    line = lines[i] 
                    while not (line.upper().startswith('DEPOT_SECTION')):                  
                        inputs = line.split()
						# Validating demand not greater than capacity
                        if float(inputs[1]) > vrpManager.capacity:
                            print(sys.stderr,
							'Invalid input: the demand of the node %s is greater than the vehicle capacity!' % vrpManager.nodes[0])
                            text_out = open('1000.out', 'a')
                            print(sys.stderr,
							'Invalid input: the demand of the node %s is greater than the vehicle capacity!' % vrpManager.nodes[0], file=text_out)
                            text_out.close()
                            exit(1)
                        if float(inputs[1]) < 0:
                            print(sys.stderr,
                            'Invalid input: the demand of the node %s cannot be negative!' % vrpManager.nodes[0])
                            text_out = open('1000.out', 'a')
                            print(sys.stderr,
                            'Invalid input: the demand of the node %s cannot be negative!' % vrpManager.nodes[0], file=text_out)
                            text_out.close()
                            exit(1)                            
                        vrpManager.nodes[int(inputs[0])][1] =  float(inputs[1])
                        i += 1
                        line = lines[i]
                        while (line=='\n'):
                            i += 1
                            line = lines[i]
                            if line.upper().startswith('DEPOT_SECTION'): break
                        if line.upper().startswith('DEPOT_SECTION'):
                            vrpManager.nodes = np.delete(vrpManager.nodes, 0, 0) 
                            print('Done.')
                            text_out = open('1000.out', 'a')                         
                            print('Done.', file=text_out)
                            text_out.close()
                            return(vrpManager.capacity, vrpManager.nodes, vrpManager.opt)
# ------------------------- End of reading the input data file ------------------------------------

# ------------------------- Start calculating the cost table --------------------------------------
@cuda.jit
def calc_cost_gpu(data_d, popsize, vrp_capacity, cost_table_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for row in range(threadId_row, data_d.shape[0], stride_x):
        for col in range(threadId_col, data_d.shape[0], stride_y):
            cost_table_d[row, col] = \
            round(hypot(data_d[row, 2] - data_d[col, 2], data_d[row, 3] - data_d[col, 3]))
# ------------------------- End calculating the cost table ----------------------------------------

# ------------------------- Start fitness calculation ---------------------------------------------
@cuda.jit
def fitness_gpu(cost_table_d, pop, fitness_val_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        fitness_val_d[row, 0] = 0
        pop[row, -1] = 1
        
        if threadId_col == 15:
            for i in range(pop.shape[1]-2):
                fitness_val_d[row, 0] += \
                cost_table_d[int(pop[row, i])-1, int(pop[row, i+1])-1]
            pop[row, -1] = fitness_val_d[row,0]
    
    cuda.syncthreads()

@cuda.jit
def fitness_gpu_new(cost_table_d, pop, fitness_val_d):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
  
    if threadId_row < pop.shape[0]:
        fitness_val_d[threadId_row, 0] = 0
        pop[threadId_row, -1] = 1
                
        for col in range(threadId_col, pop.shape[1]-2, stride_y):
            if col > 0:
                cuda.atomic.add(fitness_val_d, (threadId_row, 0), cost_table_d[int(pop[threadId_row, col])-1, int(pop[threadId_row, col+1])-1])

        pop[threadId_row, -1] = fitness_val_d[threadId_row,0]
    
    cuda.syncthreads()
# ------------------------- End fitness calculation ---------------------------------------------

# ------------------------- Start adjusting individuals ---------------------------------------------
@cuda.jit
def find_duplicates(pop, r_flag):
    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:
            # Detect duplicate nodes:
            for i in range(2, pop.shape[1]-1):
                for j in range(i, pop.shape[1]-1):
                    if pop[row, i] != r_flag and pop[row, j] == pop[row, i] and i != j:
                        pop[row, j] = r_flag
@cuda.jit
def shift_r_flag(r_flag, vrp_capacity, data_d, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        if threadId_col == 15:           
            # Shift all r_flag values to the end of the list:        
            for i in range(2, pop.shape[1]-2):
                if pop[row,i] == r_flag:
                    k = i
                    while pop[row,k] == r_flag:
                        k += 1
                    if k < pop.shape[1]-1:
                        pop[row,i], pop[row,k] = pop[row,k], pop[row,i]
@cuda.jit
def find_missing_nodes(r_flag, data_d, missing_d, pop):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        if threadId_col == 15:
            missing_d[row, threadId_col] = 0        
            # Find missing nodes in the solutions:
            for i in range(1, data_d.shape[0]):
                for j in range(2, pop.shape[1]-1):
                    if data_d[i,0] == pop[row,j]:
                        missing_d[row, i] = 0
                        break
                    else:
                        missing_d[row, i] = data_d[i,0]

@cuda.jit
def add_missing_nodes(r_flag, data_d, missing_d, pop):   
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):       
        if threadId_col == 15:           
            # Add the missing nodes to the solution:
            for k in range(missing_d.shape[1]):
                for l in range(2, pop.shape[1]-1):
                    if missing_d[row, k] != 0 and pop[row, l] == r_flag:
                        pop[row, l] = missing_d[row, k]
                        break
@cuda.jit
def cap_adjust(r_flag, vrp_capacity, data_d, pop):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        if threadId_col == 15:
            reqcap = 0.0        # required capacity
            
            # Accumulate capacity:
            i = 1
            while pop[row, i] != r_flag:
                i += 1  
                if pop[row,i] == r_flag:
                    break
            
                if pop[row, i] != 1:
                    reqcap += data_d[int(pop[row, i]-1), 1] # index starts from 0 while individuals start from 1                
                    if reqcap > vrp_capacity:
                        reqcap = 0
                        # Insert '1' and shift right:
                        new_val = 1
                        rep_val = pop[row, i]
                        for j in range(i, pop.shape[1]-2):
                            pop[row, j] = new_val
                            new_val = rep_val
                            rep_val = pop[row, j+1]
                else:
                    reqcap = 0.0
    cuda.syncthreads()

@cuda.jit
def cleanup_r_flag(r_flag, pop):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):
        for col in range(threadId_col, pop.shape[1], stride_y):
            if pop[row, col] == r_flag:
                pop[row, col] = 1
    
    cuda.syncthreads()
# ------------------------- End adjusting individuals ---------------------------------------------

# ------------------------- Start initializing individuals ----------------------------------------
@cuda.jit
def initializePop_gpu(rng_states, data_d, missing_d, pop_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
    # Generate the individuals from the nodes in data_d:
        for col in range(threadId_col, data_d.shape[0]+1, stride_y):
            pop_d[row, col] = data_d[col-1, 0]
        
        pop_d[row, 0], pop_d[row, 1] = 1, 1
        
    # # Randomly shuffle each individual on a separate thread:      
    # if threadId_row < pop_d.shape[0] and threadId_col > 1:
    #     for col in range(threadId_col, data_d.shape[0]+1, stride_y):
    #         rnd_col = 0
    #         while rnd_col == 0:
    #             # rnd = (xoroshiro128p_uniform_float32(rng_states, threadId_row*threadId_col)*(data_d.shape[0]-2))
    #             # To convert from row-column indexing to linear scalars, we use: col + row*array_width (i.e., array.shape[1])
    #             rnd = xoroshiro128p_uniform_float32(rng_states, col+(threadId_row*pop_d.shape[1]))*(data_d.shape[0]-2)
    #             rnd_col = int(rnd)+2

    #     pop_d[threadId_row, col], pop_d[threadId_row, rnd_col] =\
    #     pop_d[threadId_row, rnd_col], pop_d[threadId_row, col]
# ------------------------- End initializing individuals ------------------------------------------

# ------------------------- Start two-opt calculations --------------------------------------------
@cuda.jit
def reset_to_ones(pop):
   
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        for col in range(threadId_col, pop.shape[1], stride_y):
            pop[row, col] = 1   
    cuda.syncthreads()
    
@cuda.jit
def two_opt(pop, cost_table, candid_d_3):
    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop.shape[0], stride_x):    
        for col in range(threadId_col, pop.shape[1], stride_y):
            # candid_d_3[row, col] = 1
            if col+2 < pop.shape[1] :
                # Divide solution into routes:
                if pop[row, col] == 1 and pop[row, col+1] != 1 and pop[row, col+2] != 1:
                    route_length = 1
                    while pop[row, col+route_length] != 1 and col+route_length < pop.shape[1]:
                        candid_d_3[row, col+route_length] = pop[row, col+route_length]
                        route_length += 1

                    # Now we have candid_d_3 has the routes to be optimized for every row solution
                    total_cost = 0
                    min_cost =0

                    for i in range(0, route_length):
                        min_cost += \
                            cost_table[int(candid_d_3[row,col+i])-1, int(candid_d_3[row,col+i+1])-1]
                
                    # ------- The two opt algorithm --------
            
                    # So far, the best route is the given one (in candid_d_3)
                    improved = True
                    while improved:
                        improved = False
                        for i in range(1, route_length-1):
                                # swap every two pairs
                                candid_d_3[row, col+i], candid_d_3[row, col+i+1] = \
                                candid_d_3[row, col+i+1], candid_d_3[row, col+i]
                                
                                for j in range(0, route_length):
                                    total_cost += cost_table[int(candid_d_3[row,col+j])-1,\
                                                int(candid_d_3[row,col+j+1])-1]
                                
                                if total_cost < min_cost:
                                    min_cost = total_cost
                                    improved = True
                                else:
                                    candid_d_3[row, col+i+1], candid_d_3[row, col+i]=\
                                    candid_d_3[row, col+i], candid_d_3[row, col+i+1]
                    
                    for k in range(0, route_length):
                        pop[row, col+k] = candid_d_3[row, col+k]
# ------------------------- End two-opt calculations --------------------------------------------

# ------------------------- Start evolution process ---------------------------------------------
# --------------------------------- Cross Over part ---------------------------------------------
@cuda.jit
def select_candidates(pop_d, random_arr_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, assign_child_1):
    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):    
        for col in range(threadId_col, pop_d.shape[1], stride_y):
            if assign_child_1:
            #   First individual in pop_d must be selected:
                candid_d_1[row, col] = pop_d[0, col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]
            else:
            #   Create a pool of 4 randomly selected individuals:
                candid_d_1[row, col] = pop_d[random_arr_d[row, 0], col]
                candid_d_2[row, col] = pop_d[random_arr_d[row, 1], col]
                candid_d_3[row, col] = pop_d[random_arr_d[row, 2], col]
                candid_d_4[row, col] = pop_d[random_arr_d[row, 3], col]
    
    cuda.syncthreads()
@cuda.jit  
def select_parents(pop_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        for col in range(threadId_col, pop_d.shape[1], stride_y):
        # Selecting 2 parents with binary tournament   
        # ----------------------------1st Parent--------------------------------------------------            
            if candid_d_1[row, -1] < candid_d_2[row, -1]:
                parent_d_1[row, col] = candid_d_1[row, col]
            else:
                parent_d_1[row, col] = candid_d_2[row, col]

            # ----------------------------2nd Parent--------------------------------------------------
            if candid_d_3[row, -1] < candid_d_4[row, -1]:
                parent_d_2[row, col] = candid_d_3[row, col]
            else:
                parent_d_2[row, col] = candid_d_4[row, col]
       
    cuda.syncthreads()

@cuda.jit
def number_cut_points(candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2, count, min_n, max_n):
    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, candid_d_1.shape[0], stride_x):
        for col in range(threadId_col, candid_d_1.shape[1], stride_y):
            candid_d_1[row, col] = 1
            candid_d_2[row, col] = 1
            candid_d_3[row, col] = 1
            candid_d_4[row, col] = 1

        # Calculate the actual length of parents
        if threadId_col == 15:
            for i in range(0, candid_d_1.shape[1]-2):
                if not (parent_d_1[row, i] == 1 and parent_d_1[row, i+1] == 1):
                    candid_d_1[row, 2] += 1
                    
                if not (parent_d_2[row, i] == 1 and parent_d_2[row, i+1] == 1):
                    candid_d_2[row, 2] += 1

            # Minimum length of the two parents
            candid_d_1[row, 3] = \
            min(candid_d_1[row, 2], candid_d_2[row, 2]) 

            # Number of cutting points = (n/5 - 2)
            candid_d_1[row, 4] = candid_d_1[row, 3]//20 - 2
            # n_points = max(min_n, (count%(max_n*4000))//4000) # the n_points increases one every 5000 iterations till 20 then resets to 2 and so on
            candid_d_1[row, 4] = 2 # n_points is replaced by 2 for 2-point crossover
    cuda.syncthreads()

@cuda.jit
def add_cut_points(candid_d_1, candid_d_2, rng_states):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, candid_d_1.shape[0], stride_x):    
        if threadId_col == 15:
            no_cuts = candid_d_1[row, 4]
            
            for i in range(1, no_cuts+1):
                rnd_val = 0
                
            # Generate unique random numbers as cut indices:
                for j in range(1, no_cuts+1):
                    while rnd_val == 0 or rnd_val == candid_d_2[row, j]:
                        # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                        #       *(candid_d_1[row, 3] - 2) + 2 # random*(max-min)+min
                        rnd = xoroshiro128p_uniform_float32(rng_states, row*candid_d_1.shape[1])\
                            *(candid_d_1[row, 3] - 2) + 2 # random*(max-min)+min
                        # rnd = xoroshiro128p_normal_float32(rng_states, row*candid_d_1.shape[1])\
                        #       *(candid_d_1[row, 3] - 2) + 2 # random*(max-min)+min
                        rnd_val = int(rnd)+2            
                
                candid_d_2[row, i+1] = rnd_val
                
            # Sorting the crossover points:
            if threadId_col == 15: # Really! it is already up there! see the main if statement.
                for i in range(2, no_cuts+2):
                    min_val = candid_d_2[row, i]
                    min_index = i

                    for j in range(i + 1, no_cuts+2):
                        # Select the smallest value
                        if candid_d_2[row, j] < candid_d_2[row, min_index]:
                            min_index = j

                    candid_d_2[row, min_index], candid_d_2[row, i] = \
                    candid_d_2[row, i], candid_d_2[row, min_index]

    cuda.syncthreads()

@cuda.jit
def cross_over_gpu(random_arr, candid_d_1, candid_d_2, child_d_1, child_d_2, parent_d_1, parent_d_2, crossover_prob):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    # crossover_prob = 60
    for row in range(threadId_row, candid_d_1.shape[0], stride_x):
        for col in range(threadId_col, candid_d_1.shape[1] - 1, stride_y):
            if col > 1 and col < child_d_1.shape[1]-1:
                child_d_1[row, col] = parent_d_1[row, col]
                child_d_2[row, col] = parent_d_2[row, col]

                if random_arr[row, 0] <= crossover_prob: # Perform crossover with a probability of 0.6
                    # Perform the crossover:
                    no_cuts = int(candid_d_1[row, 4])
                    if col < candid_d_2[row, 2]: # Swap from first element to first cut point
                        child_d_1[row, col], child_d_2[row, col] =\
                        child_d_2[row, col], child_d_1[row, col]

                    if no_cuts%2 == 0: # For even number of cuts, swap from the last cut point to the end
                        if col > int(candid_d_2[row, no_cuts+1]) and col < int(child_d_1.shape[1]-1):
                            child_d_1[row, col], child_d_2[row, col] =\
                            child_d_2[row, col], child_d_1[row, col]

                    for j in range(2, no_cuts+1):
                        cut_idx = int(candid_d_2[row, j])
                        if no_cuts%2 == 0:
                            if j%2==1 and col >= cut_idx and col < int(candid_d_2[row, j+1]):
                                child_d_1[row, col], child_d_2[row, col] =\
                                child_d_2[row, col], child_d_1[row, col]
                        
                        elif no_cuts%2 == 1:
                            if j%2==1 and col>=cut_idx and col < int(candid_d_2[row, j+1]):
                                child_d_1[row, col], child_d_2[row, col] =\
                                child_d_2[row, col], child_d_1[row, col]

    cuda.syncthreads()
# ------------------------------------Mutation part -----------------------------------------------
@cuda.jit
def mutate(rng_states, child_d_1, child_d_2, mutation_prob):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, child_d_1.shape[0], stride_x):    
    # Swap two positions in the children, with 0.3 probability
        if threadId_col == 15:
            # mutation_prob = 30
            
            # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
            #       *(mutation_prob - 1) + 1 # random*(max-min)+min
            rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])*99 # random*(max-min)+min
            # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_1.shape[1])\
            #       *(mutation_prob - 1) + 1 # random*(max-min)+min
            rnd_val = int(rnd)+2
            if rnd_val <= mutation_prob: # Mutation operator of (mutation_prob%)
                i1 = 1
                
                # Repeat random selection if depot was selected:
                while int(child_d_1[row, i1]) == 1 or i1 >= child_d_1.shape[1]-1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #       *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                        *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_1.shape[1])\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    i1 = int(rnd)+2        

                i2 = 1
                while child_d_1[row, i2] == 1 or i2 >= child_d_1.shape[1]-1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_1.shape[1])\
                        *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_1.shape[1])\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    i2 = int(rnd)+2 
                    
                child_d_1[row, i1], child_d_1[row, i2] = \
                child_d_1[row, i2], child_d_1[row, i1]

            # Repeat for the second child:    
                i1 = 1
                
                # Repeat random selection if depot was selected:
                while child_d_2[row, i1] == 1 or i1 >= child_d_2.shape[1]-1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_2.shape[1])\
                        *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_2.shape[1])\
                    #     *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    i1 = int(rnd)+2        

                i2 = 1
                while child_d_2[row, i2] == 1 or i2 >= child_d_2.shape[1]-1:
                    # rnd = xoroshiro128p_uniform_float32(rng_states, row)\
                    #     *(child_d_1.shape[1] - 4) + 2 # random*(max-min)+min
                    rnd = xoroshiro128p_uniform_float32(rng_states, row*child_d_2.shape[1])\
                        *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    # rnd = xoroshiro128p_normal_float32(rng_states, row*child_d_2.shape[1])\
                    #     *(child_d_2.shape[1] - 4) + 2 # random*(max-min)+min
                    i2 = int(rnd)+2 
                    
                child_d_2[row, i1], child_d_1[row, i2] = \
                child_d_2[row, i2], child_d_1[row, i1]
            
        cuda.syncthreads()
# -------------------------- Update population part -----------------------------------------------
@cuda.jit
def select_individual(index, pop_d, individual):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):
        if row == index and threadId_col < pop_d.shape[1]:
            pop_d[row, threadId_col] = individual[row, threadId_col]

@cuda.jit
def update_pop(count, parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d):    
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)

    for row in range(threadId_row, pop_d.shape[0], stride_x):    
        for col in range(threadId_col, pop_d.shape[1], stride_y):
            if child_d_1[row, -1] <= parent_d_1[row, -1] and \
            child_d_1[row, -1] <= parent_d_2[row, -1] and \
            child_d_1[row, -1] <= child_d_2[row, -1]:

                pop_d[row, col] = child_d_1[row, col]
                pop_d[row, 0] = count

            elif child_d_2[row, -1] <= parent_d_1[row, -1] and \
            child_d_2[row, -1] <= parent_d_2[row, -1] and \
            child_d_2[row, -1] <= child_d_1[row, -1]:

                pop_d[row, col] = child_d_2[row, col]
                pop_d[row, 0] = count

            elif parent_d_1[row, -1] <= parent_d_2[row, -1] and \
            parent_d_1[row, -1] <= child_d_1[row, -1] and \
            parent_d_1[row, -1] <= child_d_2[row, -1]:

                pop_d[row, col] = parent_d_1[row, col]
                pop_d[row, 0] = count

            elif parent_d_2[row, -1] <= parent_d_1[row, -1] and \
            parent_d_2[row, -1] <= child_d_1[row, -1] and \
            parent_d_2[row, -1] <= child_d_2[row, -1]:

                pop_d[row, col] = parent_d_2[row, col]
                pop_d[row, 0] = count
                
    cuda.syncthreads()

# ------------------------- Mutation Fucntion --------------------------------------------------------
@cuda.jit
def inverse_mutate(random_min_max, pop, random_arr, mutation_prob):
    threadId_row, threadId_col = cuda.grid(2)
    stride_x, stride_y = cuda.gridsize(2)
    
    for row in range(threadId_row, pop.shape[0], stride_x):
        if random_arr[row,0] <= mutation_prob:
            for col in range(threadId_col, pop.shape[1], stride_y):
                start  = random_min_max[row, 0]
                ending = random_min_max[row, 1]
                length = ending - start
                diff   = col - start
                if col >= start and col < start+ceil(length/2):
                    pop[row, col], pop[row, ending-diff] = pop[row, ending-diff], pop[row, col]

# ------------------------- Definition of CPU functions ----------------------------------------------   
#def inverse_mutate(pop, popsize, mutation_prob):
#    random_min_max = cp.random.randint(2, pop.shape[1]-2, (popsize, 2))
#    random_min_max.sort()
#
#    # mutation_prob = 10
#    random_arr = cp.random.randint(1, 100, (popsize, 1))
#    for i, individual in enumerate(pop):
#        if random_arr[i] <= mutation_prob:
#            individual = cp.concatenate((individual[0:random_min_max[i,0]],\
#            cp.flip(individual[random_min_max[i,0]:random_min_max[i,1]], axis=0), individual[random_min_max[i,1]:]))
#            pop[i,:] = individual[:]

def select_bests(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize):
    # Select the best 5% from paernt 1 & parent 2:
    
    pool = parent_d_1[parent_d_1[:,-1].argsort()][0:0.9*popsize,:]
    # pool = parent_d_1[parent_d_1[:,-1].argsort()][0:0.7*popsize,:]
    # pool = parent_d_1[parent_d_1[:,-1].argsort()][0:0.05*popsize,:]
    
    pool = cp.concatenate((pool, parent_d_2[parent_d_2[:,-1].argsort()][0:0.9*popsize,:]))
    # pool = cp.concatenate((pool, parent_d_2[parent_d_2[:,-1].argsort()][0:0.7*popsize,:]))
    # pool = cp.concatenate((pool, parent_d_2[parent_d_2[:,-1].argsort()][0:0.05*popsize,:]))
    
    pool = pool[pool[:,-1].argsort()]

    # Sort child 1 & child 2:
    child_d_1 = child_d_1[child_d_1[:,-1].argsort()]
    child_d_2 = child_d_2[child_d_2[:,-1].argsort()]

    pop_d[0:0.9*popsize, :] = pool[0:0.9*popsize, :]
    # pop_d[0:0.7*popsize, :] = pool[0:0.7*popsize, :]
    # pop_d[0:0.05*popsize, :] = pool[0:0.05*popsize, :]
    
    pop_d[0.9*popsize:0.95*popsize, :] = child_d_1[0:0.05*popsize, :]
    # pop_d[0.7*popsize:0.85*popsize, :] = child_d_1[0:0.15*popsize, :]
    # pop_d[0.05*popsize:0.51*popsize, :] = child_d_1[0:0.46*popsize, :]
    
    pop_d[0.95*popsize:popsize, :] = child_d_2[0:0.05*popsize, :]
    # pop_d[0.85*popsize:popsize, :] = child_d_2[0:0.15*popsize, :]
    # pop_d[0.5*popsize:popsize, :] = child_d_2[0:0.5*popsize, :]

def cp_unique_axis0(array):
    if len(array.shape) != 2:
        raise ValueError("Input array must be 2D.")
    sortarr     = array[cp.lexsort(array.T[::-1])]
    mask        = cp.empty(array.shape[0], dtype=np.bool_)
    mask[0]     = True
    mask[1:]    = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
    return sortarr[mask]

# ------------------------- Start Main ------------------------------------------------------------
try:

    #####################################################################################################
    #---------------------  Logging population genes ----------------------------------------------------
    #path = os.getcwd() + '../results/genes/' + sys.argv[1] + '/'
    #try:
    #    os.mkdir(path)
    #    print('new directory created for {}'.format(sys.argv[1]))
    #except:
    #    print('directory {} already exists'.format(sys.argv[1]))
    
    #---------------------- Starting new output file ----------------------------------------------------
    os.remove('1000.out')
    f = open('1000.out', 'w')
    f.close()

    #----------------------------------------------------------------------------------------------------
    #####################################################################################################

    vrp_capacity, data, opt = readInput()
    n = int(sys.argv[4])
#####crossover_prob = int(sys.argv[5])
#####mutation_prob = int(sys.argv[6])
    popsize = -(-(n*(data.shape[0] - 1))//1000)*1000
    
    # popsize = 500
    print('Taking population size {}*number of nodes'.format(n))
    text_out = open('1000.out', 'a')
    print('Taking population size {}*number of nodes'.format(n),file=text_out)
    text_out.close()
    min_n = 1 # Minimum number of crossover points
    max_n = 1 # Maximum number of crossover points

    try:
        generations = int(sys.argv[2])
    except:
        print('No generation limit provided, taking 2000 generations...')
        text_out = open('1000.out', 'a')
        print('No generation limit provided, taking 2000 generations...', file=text_out)
        text_out.close()
        generations = 2000

    r_flag = 9999 # A flag for removal/replacement

    data_d = cuda.to_device(data)
    cost_table_d = cuda.device_array(shape=(data.shape[0], data.shape[0]), dtype=np.float32)

    pop_d = cp.ones((popsize, int(2*data.shape[0])+2), dtype=np.float32)

    missing_d = cp.zeros(shape=(popsize, pop_d.shape[1]), dtype=np.float32)

    missing = np.ones(shape=(popsize,1), dtype=np.bool)
    missing_elements = cuda.to_device(missing)

    fitness_val = np.zeros(shape=(popsize,1), dtype=np.float32)
    fitness_val_d = cuda.to_device(fitness_val)

    # GPU grid configurations:
    grid      = gpuGrid.GRID()
    blocks_x, blocks_y = grid.blockAlloc(data.shape[0], float(n))
    # blocks_no = 20
    tpb_x, tpb_y      = grid.threads_x, grid.threads_y

    print(grid)
    blocks            = (blocks_x, blocks_y)
    threads_per_block = (tpb_x, tpb_y)   

    val = val.VRP(sys.argv[1])
    val.read()
    val.costTable()
    # --------------Calculate the cost table----------------------------------------------
    calc_cost_gpu[blocks, threads_per_block](data_d, popsize, vrp_capacity, cost_table_d)
    # --------------Initialize population----------------------------------------------
    rng_states = create_xoroshiro128p_states(threads_per_block[0]**2 * blocks[0]**2, seed=random.randint(2,2*10**5))
    initializePop_gpu[blocks, threads_per_block](rng_states, data_d, missing_d, pop_d)

    for individual in pop_d:
        cp.random.shuffle(individual[2:-1])

    find_duplicates[blocks, threads_per_block](pop_d, r_flag)

    find_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, pop_d)
    add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, pop_d)

    shift_r_flag[blocks, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)
    cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)
    cleanup_r_flag[blocks, threads_per_block](r_flag, pop_d)

    # UNCOMMENT THE FOLLOWING LINES TO START WITH A GIVEN BEST SOLUTION:
    # best_given = cp.array([0, 0, 543, 542, 861, 541, 540, 539, 538, 537, 536, 535, 534, 533, 532, 531, 530, 529, 528, 527, 526, 525, 524, 523, 522, 521, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 464, 465, 345, 346, 347, 466, 467, 507, 547, 546, 506, 505, 584, 544, 545, 585, 586, 587, 588, 548, 508, 468, 469, 470, 471, 472, 473, 474, 475, 554, 553, 552, 551, 550, 549, 589, 590, 591, 592, 593, 594, 595, 555, 556, 557, 476, 477, 478, 517, 518, 558, 559, 519, 479, 480, 3, 0, 869, 949, 948, 947, 986, 946, 945, 985, 944, 984, 983, 943, 942, 941, 940, 939, 938, 937, 936, 935, 934, 933, 932, 931, 930, 929, 928, 927, 926, 925, 924, 923, 922, 921, 920, 919, 918, 917, 916, 915, 914, 913, 912, 911, 910, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 969, 970, 971, 972, 973, 974, 975, 976, 977, 978, 979, 980, 981, 982, 583, 582, 581, 580, 579, 578, 577, 576, 575, 574, 573, 572, 571, 570, 569, 568, 567, 566, 565, 564, 563, 562, 561, 560, 520, 1, 0, 10, 11, 12, 13, 176, 177, 178, 179, 180, 181, 182, 183, 184, 671, 672, 673, 713, 793, 794, 795, 796, 797, 798, 799, 800, 801, 761, 760, 759, 758, 757, 756, 755, 754, 753, 752, 751, 750, 749, 748, 747, 746, 745, 744, 743, 742, 741, 740, 739, 738, 658, 659, 619, 618, 617, 616, 615, 614, 613, 612, 651, 691, 652, 653, 654, 655, 656, 657, 737, 736, 735, 734, 733, 732, 731, 730, 650, 649, 648, 647, 607, 608, 609, 610, 611, 331, 330, 329, 328, 327, 326, 286, 287, 288, 289, 290, 291, 292, 332, 333, 334, 335, 336, 337, 338, 343, 344, 19, 0, 191, 271, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 403, 402, 401, 400, 399, 398, 397, 396, 395, 394, 393, 392, 391, 431, 430, 429, 428, 427, 426, 425, 424, 423, 422, 421, 420, 419, 418, 417, 416, 415, 414, 413, 412, 411, 410, 409, 408, 407, 406, 405, 404, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 351, 350, 349, 348, 61, 60, 58, 14, 15, 16, 17, 18, 20, 21, 22, 24, 25, 26, 29, 30, 0, 2, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 880, 879, 878, 877, 876, 875, 874, 873, 872, 871, 870, 829, 909, 828, 868, 908, 907, 867, 827, 866, 906, 826, 825, 865, 905, 824, 864, 904, 903, 863, 823, 822, 862, 902, 821, 901, 900, 860, 820, 819, 899, 859, 858, 818, 898, 897, 817, 857, 856, 816, 896, 895, 855, 815, 814, 854, 894, 893, 813, 853, 852, 851, 850, 889, 890, 891, 892, 812, 811, 810, 809, 849, 808, 848, 888, 847, 807, 887, 886, 846, 806, 845, 805, 804, 844, 885, 884, 883, 803, 843, 802, 842, 881, 841, 0, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 325, 606, 646, 645, 644, 643, 683, 684, 685, 686, 766, 765, 764, 763, 762, 882, 999, 998, 997, 996, 995, 994, 993, 992, 991, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 987, 988, 1028, 1029, 1030, 990, 989, 788, 787, 786, 785, 784, 783, 702, 703, 704, 705, 706, 666, 665, 664, 663, 662, 622, 623, 624, 625, 626, 707, 708, 628, 667, 627, 35, 0, 38, 37, 36, 670, 710, 630, 629, 669, 668, 709, 789, 790, 791, 792, 712, 711, 631, 632, 633, 634, 635, 674, 714, 715, 675, 716, 676, 636, 637, 677, 717, 678, 679, 719, 718, 638, 639, 680, 640, 720, 721, 681, 641, 642, 682, 722, 723, 724, 725, 726, 727, 687, 688, 689, 690, 729, 728, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 701, 661, 621, 620, 660, 700, 699, 698, 697, 696, 695, 694, 693, 692, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 0, 39, 34, 33, 32, 31, 28, 27, 23, 57, 56, 55, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 160, 159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145, 144, 143, 142, 141, 140, 139, 138, 137, 136, 96, 59, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 0, 193, 192, 272, 273, 313, 314, 274, 275, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 445, 444, 443, 442, 441, 440, 439, 438, 437, 436, 435, 434, 433, 432, 312, 311, 310, 309, 308, 307, 509, 510, 511, 512, 513, 514, 515, 516, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 306, 305, 304, 303, 302, 301, 300, 299, 298, 297, 296, 295, 294, 293, 135, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 0, 4, 5, 6, 7, 8, 9, 215, 214, 213, 212, 211, 210, 209, 208, 207, 247, 248, 249, 250, 251, 252, 253, 254, 255, 216, 256, 257, 217, 218, 219, 220, 221, 222, 342, 341, 340, 339, 258, 259, 260, 261, 262, 263, 264, 265, 266, 226, 225, 224, 223, 185, 186, 187, 188, 189, 190, 230, 229, 228, 227, 267, 268, 269, 270, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196, 195, 194, 78, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 68754],\
    #                        dtype=np.int32) # here we define an array with the best given solution
    #                                        # (this one is for an 1040-node problem)
    # best_given = cp.add(best_given, cp.ones(best_given.shape))
    # pop_d[-1,:] = best_given[:]

    # --------------Calculate fitness----------------------------------------------
    fitness_gpu[blocks, threads_per_block](cost_table_d, pop_d, fitness_val_d)
    # print(pop_d[:,40], pop_d.shape)

    pop_d = pop_d[pop_d[:,-1].argsort()] # Sort the population to get the best later

    # asnumpy_first_pop = cp.asnumpy(pop_d)

    # --------------Evolve population for some generations----------------------------------------------
    # Create the pool of 6 arrays of the same length
    candid_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)
    candid_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)
    candid_d_3 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)
    candid_d_4 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)

    parent_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)
    parent_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)

    child_d_1 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)
    child_d_2 = cp.ones((popsize, pop_d.shape[1]), dtype=np.float32)

    cut_idx = np.ones(shape=(pop_d.shape[1]), dtype=np.int32)
    cut_idx_d = cuda.to_device(cut_idx)

    minimum_cost = float('Inf')
    old_time = timer()

    count = 0
    count_index = 0
    best_sol = 0
    assign_child_1 = False
    last_shuffle = 10000
    total_time = 0.0
    time_per_loop = 0.0
    ####################################################################################################################
    #---------------------- Solution Validator -------------------------------------------------------------------------
    # create a vrp solution validator, read the problem file in, and create a cost table on the cpu
    referee = validateSolution.VRP(sys.argv[1])
    referee.read()
    referee.costTable()

    print('Solution validator initilized')
    f = open('1000.out', 'a')
    print('Solution validator initialized', file=f)
    f.close()

    #------------------------RL agent ---------------------------------------------------------------------------------
    # parameters for the rl agent
    cRange = np.array(range(10, 110, 10))
    mRange = np.array(range(10, 110, 10))
    alpha = 0.7
    gamma = 0.1
    epsilon = 0.3
    initCrossover = 60
    initMutation = 30
    fitness = np.min(pop_d[:,-1])
    problem = sys.argv[1]
    results = {}
    
    agent = RLagent(alpha, gamma, epsilon, cRange, mRange, initCrossover, initMutation, fitness, problem)
    
    print('Reinforcement agent initialized')
    f = open('1000.out', 'a')
    print('Reinforcement agent initialized', file=f)
    f.close()

    #-------------------------------------------------------------------------------------------------------------------
    ####################################################################################################################


    while count <= generations:
        if minimum_cost <= opt:
            break
        
        ############################################################################################################################
        #-------------- Agent Decides ----------------------------------------------------------------------------------------------
        
        # conduct the first action with the given initial parameters
        if count == 0:
            crossover_prob, mutation_prob = agent.initAction()
        
        # otherwise, every 4 generations decide on an action (crossover and mutation rates)
        elif count % 4 == 0:
            crossover_prob, mutation_prob = agent.decide(count)

        #---------------------------------------------------------------------------------------------------------------------------
        #333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333333

        random_arr = np.arange(popsize, dtype=np.int32).reshape((popsize,1))
        random_arr = np.repeat(random_arr, 4, axis=1)
        
        random.shuffle(random_arr[:,0])
        random.shuffle(random_arr[:,1])
        random.shuffle(random_arr[:,2])
        random.shuffle(random_arr[:,3])    
        
        random_arr_d = cuda.to_device(random_arr)
        
        select_candidates[blocks, threads_per_block]\
                        (pop_d, random_arr_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, assign_child_1)

        select_parents[blocks, threads_per_block]\
                    (pop_d, candid_d_1, candid_d_2, candid_d_3, candid_d_4, parent_d_1, parent_d_2)  
        
        number_cut_points[blocks, threads_per_block](candid_d_1, candid_d_2, \
                            candid_d_3, candid_d_4, parent_d_1, parent_d_2, count, min_n, max_n)
        
        rng_states = create_xoroshiro128p_states(popsize*pop_d.shape[1], seed=random.randint(2,2*10**5))
        add_cut_points[blocks, threads_per_block](candid_d_1, candid_d_2, rng_states)
        
        random_arr = cp.random.randint(1, 100, (popsize, 1))
        cross_over_gpu[blocks, threads_per_block](random_arr, candid_d_1, candid_d_2, child_d_1, child_d_2, parent_d_1, parent_d_2, crossover_prob)
        
        # Performing mutation
        rng_states = create_xoroshiro128p_states(popsize*child_d_1.shape[1], seed=random.randint(2,2*10**5))
        # mutate[blocks, threads_per_block](rng_states, child_d_1, child_d_2, mutation_prob)
        random_min_max = cp.random.randint(2, pop_d.shape[1]-2, (popsize, 2))
        random_min_max.sort()
        random_arr = cp.random.randint(1, 100, (popsize, 1))
        inverse_mutate[blocks, threads_per_block](random_min_max, child_d_1, random_arr, mutation_prob)
        
        random_min_max = cp.random.randint(2, pop_d.shape[1]-2, (popsize, 2))
        random_min_max.sort()
        random_arr = cp.random.randint(1, 100, (popsize, 1))
        inverse_mutate[blocks, threads_per_block](random_min_max, child_d_2, random_arr, mutation_prob)
        #inverse_mutate(child_d_1, popsize, mutation_prob)
        #inverse_mutate(child_d_2, popsize, mutation_prob)
        
        # Adjusting child_1 array
        find_duplicates[blocks, threads_per_block](child_d_1, r_flag)

        find_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_1)
        add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_1)

        shift_r_flag[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_1)
        cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_1)
        cleanup_r_flag[blocks, threads_per_block](r_flag, child_d_1)
        
        # Adjusting child_2 array
        find_duplicates[blocks, threads_per_block](child_d_2, r_flag)

        find_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_2)
        add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, child_d_2)

        shift_r_flag[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_2)
        cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, child_d_2)
        cleanup_r_flag[blocks, threads_per_block](r_flag, child_d_2)    
        # --------------------------------------------------------------------------
        # Performing the two-opt optimization and Calculating fitness for child_1 array
        reset_to_ones[blocks, threads_per_block](candid_d_3)
        
        two_opt[blocks, threads_per_block](child_d_1, cost_table_d, candid_d_3)

        fitness_gpu[blocks, threads_per_block](cost_table_d, child_d_1, fitness_val_d)
        # --------------------------------------------------------------------------
        # Performing the two-opt optimization and Calculating fitness for child_2 array
        reset_to_ones[blocks, threads_per_block](candid_d_3)

        two_opt[blocks, threads_per_block](child_d_2, cost_table_d, candid_d_3)

        fitness_gpu[blocks, threads_per_block](cost_table_d, child_d_2, fitness_val_d)
        # --------------------------------------------------------------------------
        # Creating the new population from parents and children
        # update_pop[blocks, threads_per_block](count, parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d)
        select_bests(parent_d_1, parent_d_2, child_d_1, child_d_2, pop_d, popsize)
        # --------------------------------------------------------------------------
        # # # Replacing duplicates with random individuals from child_d_1
        pop_d = cp_unique_axis0(pop_d)
        repeats = 0
        while pop_d.shape[0] < popsize:
            if repeats >= popsize-1:
                break
            rndm = random.randint(0, popsize-1)
            pop_d = cp.concatenate((pop_d, cp.array([child_d_1[rndm,:]])), axis=0)
            # pop_d = cp_unique_axis0(pop_d)
            repeats += 1
        # # --------------------------------------------------------------------------
        # # Replacing duplicates with random individuals from child_d_2
        pop_d = cp_unique_axis0(pop_d)
        repeats = 0
        while pop_d.shape[0] < popsize:
            if repeats >= popsize-1:
                break
            rndm = random.randint(0, popsize-1)
            pop_d = cp.concatenate((pop_d, cp.array([child_d_2[rndm,:]])), axis=0)       
            # pop_d = cp_unique_axis0(pop_d)
            repeats += 1
        # --------------------------------------------------------------------------
        # Replacing duplicates with random individuals from parent_d_1
        pop_d = cp_unique_axis0(pop_d)
        repeats = 0
        while pop_d.shape[0] < popsize:
            if repeats >= popsize-1:
                break
            rndm = random.randint(0, popsize-1)
            pop_d = cp.concatenate((pop_d, cp.array([child_d_2[rndm,:]])), axis=0)       
            # pop_d = cp_unique_axis0(pop_d)
            repeats += 1
        # --------------------------------------------------------------------------
        # Replacing duplicates with random individuals from parent_d_2
        pop_d = cp_unique_axis0(pop_d)
        repeats = 0
        while pop_d.shape[0] < popsize:
            if repeats >= popsize-1:
                break
            rndm = random.randint(0, popsize-1)
            pop_d = cp.concatenate((pop_d, cp.array([child_d_2[rndm,:]])), axis=0)       
            # pop_d = cp_unique_axis0(pop_d)
            repeats += 1
        # # --------------------------------------------------------------------------
        # x = np.insert(x, 0, count, axis=1)
        # pop_d = cp.array(x)

        # --------------------------------------------------------------------------
        # Picking best solution
        old_cost = minimum_cost
        # best_sol = pop_d[0,:]
        best_sol = pop_d[pop_d[:,-1].argmin()]
        minimum_cost = best_sol[-1]
        
        worst_sol = pop_d[pop_d[:,-1].argmax()]
        worst_cost = worst_sol[-1]
       # print('GENERATION:', count - 1,'     best cost:', minimum_cost, '\n     best solution:',best_sol,'\n     worst cost:', worst_cost, '\n     worst solution', worst_sol) 
        delta = worst_cost-minimum_cost
        average = cp.average(pop_d[:,-1])

        if minimum_cost == old_cost: # To calculate for how long the quality did not improve
            count_index += 1
        else:
            count_index = 0

        # Shuffle the population after a certain number of generations without improvement 
        assign_child_1 = False
        # if count_index >= 5000 and count%last_shuffle == 0:
        #     last_shuffle *= 2.5
        #     count_index = 0
        #     r = 1
        #     #r = random.randint(1, 2)     
        #     print('\nCaught possible early convergence (%d)'%r)
        #     print('Shuffling population\n')
        #     pop_d[0,:] = pop_d[pop_d[:,-1].argmin()]

        #     for individual in pop_d[1:,:]:
        #         cp.random.shuffle(individual[2:-1])
            
        #     assign_child_1 = True # Force child 1 to participate in every cross over after shuffling
            
        #     # Adjust population after shuffling       
        #     find_duplicates[blocks, threads_per_block](pop_d, r_flag)

        #     find_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, pop_d)
            
        #     add_missing_nodes[blocks, threads_per_block](r_flag, data_d, missing_d, pop_d)

        #     shift_r_flag[blocks, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)
 
        #     cap_adjust[blocks, threads_per_block](r_flag, vrp_capacity, data_d, pop_d)

        #     cleanup_r_flag[blocks, threads_per_block](r_flag, pop_d)

        if count == 0:
            print('At first generation, Best: %d,'%minimum_cost, 'Worst: %d'%worst_cost, \
                'delta: %d'%delta, 'Avg: %.2f'%average)

            # text_out = open('1000.out', 'a')
            # print('At first generation, Best: %d,'%minimum_cost, 'Worst: %d'%worst_cost, \
            #     'delta: %d'%delta, 'Avg: %.2f'%average, file=text_out)
            # text_out.close()
        elif (count+1)%1 == 0:
            print('After %d generations, Best: %d,'%(count+1, minimum_cost), 'Worst: %d'%worst_cost, \
                'delta: %d'%delta, 'Avg: %.2f'%average)
            # text_out = open('1000.out', 'a')
            # print('After %d generations, Best: %d,'%(count+1, minimum_cost), 'Worst: %d'%worst_cost, \
            #     'delta: %d'%delta, 'Avg: %.2f'%average, file=text_out)
            # text_out.close()
        
        # if (count+1)%10000 == 0:
        #     print('\nProblem {}, best solution so far is:\n{}'.format(sys.argv[1], best_sol))
        #     text_out = open('1000.out', 'a')
        #     print('\nProblem {}, best solution so far is:\n{}'.format(sys.argv[1], best_sol), file=text_out)
        #     text_out.close()

        
        # ----------------------------create json file with gene results from the current generation ----------------------------
        #genes_out = [] # json results
        #pop_out = pop_d.copy()[:,1:].tolist() # a copy of the population which is converted to a list because numpy objects cannot be saved onto a json file and the first column is drop containg the generation
        #while len(pop_out) > 0:
        #    genes = pop_out.pop() # an indivudal from the population
        #    temp = {'fitness':genes.pop(), 'genes': genes} # the generation and fitness values are popped out of the gene and the rest of the genes are recorded in the dictionary 
       #     genes_out.append(temp) # the results for the genes are appended to the json file
       # results = {'problem': sys.argv[1], 'generation':count, 'output':genes_out}  # format the result output
        #with open(path + 'gen' + str(count) + '.json', 'w') as f: # create a json file in the genes/ $problem$ folder
        #    json.dump(results, f, indent=4) # dump the results on the newly created json file
        #-------------------------------------------------------------------------------------------------------------------------

        count +=1

        #########################################################################################################################################
        #-------------------- Validation---------------------------------------------------------------------------------------------------------
        # save the results of this generation to help validate
        log = open('../results/validation/validate_' + sys.argv[1] + '_' + str(len(os.listdir('../results/validation/')) - 1) + '.out', 'a')
        print('After %d generations, Best: %d, '%(count, minimum_cost), 'Worst: %d'%worst_cost, 'delta: %d'%delta, 'Avg: %.2f'%average, file=log)
        log.close()

        # validate the current generations best solution
        referee.validate(pop_d, count)

        #-------------------- Agent Observes Envitoment -----------------------------------------------------------------------------------------
        # the agent observes the enviroment and updates itself every 4 generations
        if count % 4 == 0:
            agent.observe(pop_d)
            agent.update()

        #--------------------- Log Results ------------------------------------------------------------------------------------------------------
        # the results of this generation are saves
        
        results.update({count - 1 : int(minimum_cost)})

        #----------------------------------------------------------------------------------------------------------------------------------------
        #########################################################################################################################################
   
    current_time = timer()
    total_time = float('{0:.4f}'.format((current_time - old_time)))
    time_per_loop = float('{0:.4f}'.format((current_time - old_time)/(count-1)))

    best_sol = cp.subtract(best_sol, cp.ones_like(best_sol))
    best_sol[0] = best_sol[0] + 1
    best_sol[-1] = best_sol[-1] + 1

    print('---------\nProblem:', sys.argv[1], ', Best known:', opt)
    print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n')
    print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
        %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n')
    print('Best solution:', best_sol, end = '\n---------\n')

    text_out = open('1000.out', 'a')
    print('---------\nProblem:', sys.argv[1], ', Best known:', opt, file=text_out)
    print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n', file=text_out)
    print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
        %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n', file=text_out)
    print('Best solution:', best_sol, end = '\n---------\n', file=text_out)
    text_out.close()

    ####################################################################################################################################
    #-------------- Final Results Logged -----------------------------------------------------------------------------------------------
    # log the solution
    results = {'fitnesses':results,  'best cost':best_sol[-1].tolist(), 'best solution':best_sol.tolist()}
    with open('../results/cost/' + sys.argv[1] + '_cost_' + str(len(os.listdir('../results/agent/')) - 1) + '.json', 'w') as f:
        json.dump(results, f, indent=4)

    # create copy of output file move the copy to the output file
    shutil.copyfile('1000.out', '../results/output/log_' + sys.argv[1] + '_' + str(len(os.listdir('../results/output/'))) + '.out')
    
    #-----------------------------------------------------------------------------------------------------------------------------------
    ####################################################################################################################################


    del data_d
   # del cost_table_d
    del pop_d
    del missing_d
    del fitness_val_d

    del candid_d_1
    del candid_d_2
    del candid_d_3
    del candid_d_4

    del parent_d_1
    del parent_d_2

    del child_d_1
    del child_d_2

    del cut_idx_d

except KeyboardInterrupt:
    current_time = timer()
    total_time = float('{0:.4f}'.format((current_time - old_time)))
    time_per_loop = float('{0:.4f}'.format((current_time - old_time)/(count-1)))
    best_sol = cp.subtract(best_sol, cp.ones_like(best_sol))
    best_sol[0] = best_sol[0] + 1
    best_sol[-1] = best_sol[-1] + 1    

    print('---------\nProblem:', sys.argv[1], ', Best known:', opt)
    print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n')
    print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
        %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n')
    print('Best solution:', best_sol, end = '\n---------\n')

    text_out = open('1000.out', 'a')
    print('\nKeyboard interrupted...', file=text_out)
    print('---------\nProblem:', sys.argv[1], ', Best known:', opt, file=text_out)
    print('Time elapsed:', total_time, 'secs', 'Time per loop:', time_per_loop, 'secs', end = '\n---------\n', file=text_out)
    print('Stopped at generation %d, Best cost: %d, from Generation: %d'\
        %(count-1, best_sol[-1], best_sol[0]), end = '\n---------\n', file=text_out)
    print('Best solution:', best_sol, end = '\n---------\n', file=text_out)
    text_out.close()

    del data_d
    del cost_table_d
    del pop_d
    del missing_d
    del fitness_val_d

    del candid_d_1
    del candid_d_2
    del candid_d_3
    del candid_d_4

    del parent_d_1
    del parent_d_2

    del child_d_1
    del child_d_2

    del cut_idx_d
