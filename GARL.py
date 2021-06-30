 # 0. python file,   1. problem file,    2. generations  3. optimal, 4. population size
import numpy as np
import cupy as cp
import itertools
import json
import os

class RLagent():
    def __init__(self, alpha, gamma, epsilon, cRange, mRange, crossover, mutation, fitness, problem):
        ##################################################################################################################
        #------------------------------------------------ Hyperparameters ------------------------------------------------
        # the learning rate
        self.alpha = alpha

        # the discount rate
        self.gamma = gamma

        # the exploration rate
        self.epsilon = epsilon
    
        ##################################################################################################################
        #--------------------------------------- action, state, and reward spaces ----------------------------------------
        # get all combinations of crossover and mutation probabilites and convert it to a list 
        self.actionSpace = list(itertools.product(cRange, mRange))

        # get the rewards table
        self.rewardSpace = np.array([200,   150,   100,   50,    25,
                                     150,   113,   75,    38,    19,
                                     100,   75,    50,    25,    13,
                                     50,    38,    25,    113,   7,
                                     0,     0,    -10,   -20,   -30,
                                    -1000, -1500, -2000, -2500, -3000])
       
        # a dictionary of all possible states, where the state is the key, and the value is the index for the q table and rewards space
        self.stateSpace = { '(HH, VHD)': 0, '(HH, HD)':1,  '(HH, MD)':2,  '(HH, LD)':3,  '(HH, VLD)':4,
                            '(H, VHD)':5,   '(H, HD)':6,   '(H, MD)':7,   '(H, LD)':8,   '(H, VLD)':9,
                            '(L, VHD)':10,  '(L, HD)':11,  '(L, MD)':12,  '(L, LD)':13,  '(L, VLD)':14,
                            '(LL, VHD)':15, '(LL, HD)':16, '(LL, MD)':17, '(LL, LD)':18, '(LL, VLD)':19,
                            '(S, VHD)':20,  '(S, HD)':21,  '(S, MD)':22,  '(S, LD)':23,  '(S, VLD)':24,
                            '(I, VHD)':25,  '(I, HD)':26,  '(I, MD)':27,  '(I, LD)':28,  '(I, VLD)':29}

        # initilize the Q-table
        self.Q = np.zeros([len(self.stateSpace), len(self.actionSpace)])

        ##################################################################################################################
        # ----------------------------------------------- initialization  ------------------------------------------------
        # a variable keeping track of how much rewards it has recieved
        self.collected  = 0

        # create an array to keep count how often each action was taken
        self.actionCount = np.zeros(len(self.actionSpace))

        # the previous fitness variable is initilized with a verh high cost
        self.prevFitness = fitness
        
        # the current fitness delta
        self.fitness = 0

        # the current diversity index
        self.diversity = 1
        
        # the current reward awarded
        self.reward = 0

        # initialize the first state (high cost, and very high diversity)
        self.currState = 0

        # the first actions are given
        self.action = self.actionSpace.index((crossover, mutation))

        # initialie the json file
        path = 'results/SARSA/agent/'
        self.json = path + problem + '_agent_' + str(len(os.listdir(path))) + '.json'
        with open(self.json, 'w') as f:
            json.dump({}, f)

    #INPUT: 0 or 1
    #OUTPUT: the value if 1 and the index if 0. If there are several max values, then a single one is choosen arbitrary
    def __max(self, out, arr):
        # hold any ties found
        ties = []

        # set an initial top value
        top = float('-inf')

        # for each element in the array
        for i in range(len(arr)):

            # if the current value is the new highest value
            if arr[i] > top:

                # then reset the tie list
                ties = []

                # set the new top value
                top = arr[i]

                # add the top value to the tie list
                ties.append([i, arr[i]])

            # else if the current value is tied to the highest value
            elif arr[i] == top:

                # then add it to the tie list
                ties.append([arr[i], i])
        
        # pick a random index
        choice = np.random.choice(np.arange(len(ties)))

        # return the desired value
        return ties[choice][out]

    # INPUT: the fitnesses of the current generation
    # OUTPUT: the change in fitness as a percentage (and set the the current fitness as the previous fitness value for the next iteration)
    def __fitness(self, fitnesses):
        # get the min fitness of the population
        bestFitness = np.amin(fitnesses)
        # obtaint the difference between the current and previous fitness values
        delta = self.prevFitness - bestFitness
        
        # the difference is divided by the previous fitness to obtain a percentage
        deltaFitness = delta / self.prevFitness
        
        # the current fitness is set as the previous fitness for the next iteration
        self.prevFitness = bestFitness

        # return the fitness imrpovement as a percenetage
        return deltaFitness

    # INPUT: the population from the enviroment's response
    # OUTPUT: percentage of unique chromosomes in the population
    def __diversity(self, array):
        sortarr     = array[np.lexsort(array.T[::-1])]
        mask        = cp.empty(array.shape[0], dtype=cp.bool_)
        mask[0]     = True
        mask[1:]    = cp.any(sortarr[1:] != sortarr[:-1], axis=1)
        return sortarr[mask]

    # INPUT: the fitness delta and diversity index of the current population
    # OUTPUT: the numberical valeus covnerted into categorical values as a tuple used to represent the state
    def __state(self, fitness, diversity):
        # an if statment to convert numerical values into into categorical bins
        if fitness < 0:
            fState = 'I'
        elif fitness == 0:
            fState = 'S'
        elif fitness < 0.01:
            fState = 'LL'
        elif fitness < 0.05:
            fState = 'L'
        elif fitness < 0.25:
            fState = 'H'
        else:
            fState = 'HH'

        # an if statment to convert numerical values into into categorical bins
        if diversity <= 0.2:
            dState = 'VLD'
        elif diversity <= 0.4:
            dState = 'LD'
        elif diversity <= 0.6:
            dState = 'MD'
        elif diversity <= 0.8:
            dState = 'HD'
        else:
            dState = 'VHD'

        # the state is obtained and formatted for latter use
        state = '(' + fState + ', ' + dState + ')'
        
        # the state key is used to find its index in the state space and set as the new state for the agent
        self.nextState = self.stateSpace[state]

    # INPUT: the object's state variable
    # OUTPUT: the reward given the state
    def __reward(self):
        # the reward is look up in the table
        self.reward = self.rewardSpace[self.nextState]

        # the rewards is added to the collection
        self.collected += self.reward

    # used for printing output
    def __findState(self):
        for i in self.stateSpace:
            if self.stateSpace[i] == self.currState:
                return i

    # print and save results from each generation
    def __results(self, count):
        # a dictionary holding all the results
        action = self.actionSpace[self.action]
        results = {'action':(int(action[0]), int(action[1])) , 'state':self.__findState(), 'diversity index':float(self.diversity), 'fitness delta':int(self.fitness), 'reward':int(self.reward), 'collected':int(self.collected)}
     
        
        # the json file is opened
        with open(self.json, 'r+') as f:

            # the file is loaded up
            data = json.load(f)

            # the data is updated
            data.update({count:results})

            # the data is dumped back into the file
            f.seek(0)
            json.dump(data, f, indent=4)
        
        print()        
        for i in results:
            print('   ' + i + ':', results[i])
        print()

    # the first action is given
    def initAction(self):
        # reset the action count to disregard the first action
        self.actionCount = np.zeros(len(self.actionSpace))

        # the action count is updated
        self.actionCount[self.action] += 1
        
        # update the results log
        self.__results(0)

        # give the enviroment its action (the crossover and mutation probability)
        return self.actionSpace[self.action][0], self.actionSpace[self.action][1]

    # the agent decides an action for the enviroment
    def decide(self, count):
        # randomly decide to explore (with probability epsilon)
        if np.random.random() <= self.epsilon:

            # a random action is chosen
            self.action = int(np.random.randint(low=0, high=len(self.actionSpace)))

        # or exploit (with probability 1 - epsilon)
        else:

            # the max action is chosen
            self.action = int(self.__max(0, (self.Q[self.currState])))
        
        # the action count is updated
        self.actionCount[self.action] += 1

        # print and save the results
        self.__results(count)

        # give the enviroment its action (the crossover and mutation probability)
        return self.actionSpace[self.action][0], self.actionSpace[self.action][1]
        
    # the agent observes the enviroment's response to the agent's action
    def observe(self, envResponse):
        # obtain the population and their fitnesses after an action
        population = envResponse[:, 1:-1]
        fitnesses = envResponse[:, -1]
        
        # determine the delta of the previous fitness and the current best fitness of the population and the diversity 
        self.fitness = self.__fitness(fitnesses)
        self.diversity = self.__diversity(population).shape[0]/population.shape[0]
        
        # get the new state and rewards
        self.__state(self.fitness, self.diversity)
        self.__reward()

    # the Q table is updated along with other variables for the q learning algorithm
    def updateQlearning(self):
        # update the q table using the bellman equation
        self.Q[self.currState, self.action] += self.alpha * (self.reward + self.gamma * self.__max(1, self.Q[self.nextState]) - self.Q[self.currState, self.action] )

        # update the current state
        self.currState = self.nextState

    # the Q table is updated along with other variables for the SARSA algorithm
    def updateSARSA(self):
        # update the q table using the bellman equation
        self.Q[self.prevState, self.prevAction] += self.alpha * (self.reward + self.gamma * self.Q[self.currState, self.action] - self.Q[self.prevState, self.prevAction])

        # update the state and action variables
        self.prevAction = self.action
        self.prevState = self.currState

############################################################################################################
#---------------------------------------------- for debugging ----------------------------------------------
def Qlearning(agent):
    for gen in range(10):
        ########### Choose action
        if gen == 0:
            crossover, mutation = agent.initAction()
        else:
            crossover, mutation = agent.decide()
        
        ########### Imitate the enviroment with random population
        population = np.random.randint(low=1, high=1000, size=(100,100))
        
        ########### Observe state and rewards
        agent.observe(population)

        ########### Update the policy
        agent.updateQlearning()

def SARSA(agent):

    ########### take some random action
    crossover, mutation = agent.decide()
    agent.prevAction = agent.action
    agent.prevState = agent.currState

    for generation in range(10):
        ########### Imitate the enviroment with random population
        population = np.random.randint(low=1, high=10, size=(100,100))

        ########### Observe the state and reward
        agent.observe(population)

        # set the current state so that it may choose action from the current policy using the current state
        agent.currState = agent.nextState

        ########### Take that action
        crossover, mutation = agent.decide()

        ########### Update the policy
        agent.updateSARA()

if __name__ == '__main__':
    cRange = np.array(range(1, 10))/10
    mRange = np.array(range(1, 10))/10
    alpha = 0.7
    gamma = 0.1
    epsilon = 0.3
    crossover = 0.6
    mutation = 0.3
    agent = RLagent(alpha, gamma, epsilon, cRange, mRange, crossover, mutation)
    #Qlearning(agent)
    #SARSA(agent)
