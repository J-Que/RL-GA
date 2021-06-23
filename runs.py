import os
import sys
import pandas as pd
import concurrent.futures

def rlga(parameters):
    os.system('python gpu.py ' + parameters[0] + ' 20000 ' + parameters[1] + ' ' + str(index))
    return parameters[0]

def main():
    df = pd.read_csv('runs.csv')
    params = [[df['Problem'][i], df['Best Known'][i]] for i in range(sys.argv[1], sys.argv[2])]

    for i in range(sys.argv[3], sys.argv[4]):
        index = i

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(rlga, param) for param in params]
            
            for run in concurrent.futures.as_completed(results):
                print('*******************************************************')
                print(run.result(), '     ', index, 'FINISHED')

if __name__ == '__main__':
    main()