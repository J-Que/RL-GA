import numpy as np
import json
import os

class VRP():
    def __init__(self, problem):
        self.file = problem
        self.n = np.int64(self.file.split('-')[1][1:])
        self.routes = np.int64(self.file.split('-')[2][1:])
        
        validation = '../results/validation/'
        self.log = validation + 'validate_' + problem + '_' + str(len(os.listdir(validation))) + '.out' 
        f = open(self.log, 'w')
        f.close()
      
    def __demand(self, text):
        demand = []
        for line in text:
            d = [int(line.split()[0]), int(line.split()[1])]
            demand.append(d)
        self.demand = np.array(demand, dtype=np.float64)

    def __coordinates(self, text):
        coords = []
        for line in text:
            node = [int(line.split()[0]), int(line.split()[1]), int(line.split()[2])]
            coords.append(node)
        self.coordinates = np.array(coords, dtype=np.float64)
   
    def read(self):
        f = open('../test_set/' + self.file[0] + '/' + self.file + '.vrp', 'r')
        text = f.readlines()

        for i, line in enumerate(text):
            if line.startswith('CAPACITY'): 
                self.capacity = np.float64( line.split() [-1] )
            
            if line.startswith('COMMENT'):
                try: 
                    self.bestKnown = np.float64( line.split() [-1] [:-1] )
                except: 
                    self.bestKnown = np.float64(0)
            
            if line.startswith('NODE_COORD_SECTION'):
                coords = i
            elif line.startswith('DEMAND_SECTION'):
                demand = i
            elif line.startswith('DEPOT_SECTION'):
                depot = i
        
        self.__demand(text[demand + 1 : depot])
        self.__coordinates(text[coords + 1 : demand])

    def __calculate(self, x):
        XT = np.array([x, ] * len(x))
        X = XT.transpose()
        Xdelta = X - XT
        Xsquared = Xdelta * Xdelta
        return Xsquared
    
    def costTable(self):
        x = self.__calculate(self.coordinates[:, 1])
        y = self.__calculate(self.coordinates[:, 2])
        costSquared = x + y
        self.exactCostTable = np.sqrt(costSquared)
        self.roundedCostTable = np.around(np.sqrt(costSquared))
    
    def __log(self, message):
        print(message)
        log = open(self.log, 'a')
        print(message, file=log)
        log.close()

    def __missing(self, solution):
        print()
        valid = True
        for i in range(self.n): # ones were subtracted so this range is correct, otherwise one off
            if i not in solution:
                self.__log('          !!!!!     MISSING NODE ' + str(i) +  '     !!!!!!')
                valid = False       
        if valid: self.__log('          NONE MISSING')

    def __repeat(self, solution):
        found = []
        valid = True
        for i in solution:
            if i != 0:
                if i not in found:
                   found.append(i)
                else:
                    self.__log('          !!!!!     REPEATED NODE ' + str(i) + '     !!!!!')
                    valid = False        
        
        if valid: self.__log('          NONE REPEATED')

        return valid

    def __capacity(self, solution):
        load = 0
        valid = True
        for num in solution:
            if num == 0:
                load = 0
            else:
                load += self.demand[int(num), 1]
                if load > self.capacity:
                    self.__log('          !!!!!      CAPCITY VIOLATED     !!!!!')
                    valid = False
        if valid: self.__log('          CAPACITY VALID')
        
        return valid

    def __cost(self, solution, GAsolution):
        exactCosts = 0
        roundedCosts = 0
        for num in range(1, len(solution)):
            i = int(solution[num])
            j = int(solution[num - 1])
            exactCosts += self.exactCostTable[i][j]
            roundedCosts += self.roundedCostTable[i][j]
        
        if round(exactCosts) == GAsolution or roundedCosts == GAsolution:
            self.__log('          -------------  EQUAL  -------------')
            valid = True
        else:
            self.__log('          -------------  NOT EQUAL !!!  -------------')
            valid = False

        self.__log('          GA solution: ' + str(GAsolution))
        self.__log('          Exact solution: ' + str(exactCosts))
        self.__log('          Rounded solution: '+ str(roundedCosts) + '\n')

        return valid

    def validate(self, population, generation):
        phenotype = population[population[:, -1].argmin()]
        cost = phenotype[-1]
        solution = phenotype[1:-1] - 1
        
        self.__log('     -------- Generation : ' +  str(generation))
        NoMissing = self.__missing(solution)
        NoRepeated = self.__repeat(solution)
        BelowCapacity = self.__capacity(solution)
        Match = self.__cost(solution, cost)

    def valid(self, solution):
        self.__log('---------', self.file)
        NoMissing = self.__missing(solution)
        NoRepeated = self.__repeat(solution)
        BelowCapacity = self.__capacity(solution)
        Match = self.__cost(solution, cost)

def validate():
    path = '../results/cost/'
    for f in os.listdir(path):
        end = f.find('_')
        problem = f[:end]

        with open(path + f, 'r') as j:
            data = json.load(f)
            solution = data['best solution'][1:-1]

            vrp = VRP(problem)
            vrp.log = 'validate.out' 
            vrp.read()
            vrp.costTable()
            vrp.valid(solution)

if __name__ == '__main__':
    

    phenotype = np.array([1.0, 0.0, 22.0, 7.0, 12.0, 8.0, 28.0, 26.0, 0.0, 17.0, 6.0, 3.0, 5.0, 25.0, 18.0, 16.0, 21.0, 0.0, 23.0, 30.0, 13.0, 29.0, 4.0, 0.0, 9.0, 1.0, 19.0, 24.0, 11.0, 15.0, 14.0, 0.0, 20.0, 27.0, 10.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 638.0])
    phenotype += 1
    population = np.array([phenotype, ] * 10)

    vrp.validate(population)


