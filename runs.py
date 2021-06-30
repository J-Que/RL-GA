#arguements: runs.py problem_starting_index, problem_ending_index, run_starting_index, run_ending_index
import os
import sys
import pandas as pd

def main():
    df = pd.read_csv('runs.csv')
    params = [[df['Problem'][i], int(df['Best Known'][i])] for i in range(int(sys.argv[1]), int(sys.argv[2]))]
    
    count = 1
    total = len(range(int(sys.argv[1]), int(sys.argv[2]))) * len(range(int(sys.argv[3]), int(sys.argv[4])))

    for i in range(int(sys.argv[3]), int(sys.argv[4])):
        for j in params:
            os.system('python gpu.py ' + str(j[0]) + ' 20000 ' + str(j[1]) + ' ' + str(i))
            print('*******************************************************')
            print('RUN:', str(count), 'of', str(total))
            count += 1

if __name__ == '__main__':
    main()
