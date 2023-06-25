
from statistics import variance
import sys
import argparse
import os
from os.path import dirname, realpath, sep, pardir
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
from pprint import pprint
curr_dir = os.path.dirname(__file__)
# sys.path.append(curr_dir)
sys.path.append(os.getcwd())
pprint(sys.path) # Make sure the path including folders is set currently
from dec_opt.utils import pickle_it
from dec_opt.experiment import run_exp
import numpy as np
from dec_opt.logistic_regression import LogisticRegression
from dec_opt.linear_regression import LinearRegression
from dec_opt.non_linear_regression import NonLinearRegression


def _parse_args():
    parser = argparse.ArgumentParser(description='driver.py')
    parser.add_argument('--d', type=str, default='syn1',   help='Pass data-set')
    parser.add_argument('--task', type=str, default='lin_reg', help='Choose task')
    parser.add_argument('--r', type=str, default=os.path.join(curr_dir, './data/'),          help='Pass data root')
    parser.add_argument('--o', type=str, default=None, help='Pass op location')
    parser.add_argument('--stochastic', type=bool, default=False)
    parser.add_argument('--algorithm', type=str, default='FedNDL1')  #
    parser.add_argument('--var_proposed', type=float, default=0.0)  #Noise variance
    parser.add_argument('--n_cores', type=int, default=16)
    parser.add_argument('--topology', type=str, default='torus') #fully_connected,torus,ring

    parser.add_argument('--E', type=int, default=1, help = 'Local Epochs') 
    parser.add_argument('--consensus_lr', type=float, default=0.1) # Consensus Learning Rate
    parser.add_argument('--quantization_function', type=str, default='full') #'full'
    parser.add_argument('--num_bits', type=int, default=8)
    parser.add_argument('--fraction_coordinates', type=float, default=0.05)
    parser.add_argument('--dropout_p', type=float, default=0.)

    parser.add_argument('--epochs', type=int, default=100) # No of Epochs
    parser.add_argument('--lr_type', type=str, default='epoch_decay') #epoch_decay
    parser.add_argument('--initial_lr', type=float, default=.02) # Learning Rate
    parser.add_argument('--epoch_decay_lr', type=float, default=0.9)
    parser.add_argument('--regularizer', type=float, default=0.001) # L2 regularization

    parser.add_argument('--estimate', type=str, default='final')
    parser.add_argument('--n_proc', type=int, default=3)
    parser.add_argument('--n_repeat', type=int, default=3) # No of repeats
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    arg = _parse_args()
    # Below is for running 
    print(arg)
    print('==============================')
    print('Make sure to generate the data in Data_Reader for the first time')
    print('==============================')

    if not arg.o:
        directory = "results/" + arg.d + "/" + arg.topology+"/" 
    else:
        directory = "results/" + arg.o + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    result_file = arg.d + '.a_' + arg.algorithm + '.n_' + str(arg.n_cores) + '.t_' + \
        arg.topology+'.var'+str(arg.var_proposed)  
    
    if arg.task == 'log_reg':
        model = LogisticRegression(params=arg)
    elif arg.task == 'lin_reg':
        model = LinearRegression(params=arg)
    elif arg.task == 'nlin_reg':
        model = NonLinearRegression(params=arg)   
    else:
        raise NotImplementedError
        
    args = []
    results = []
    for random_seed in np.arange(1, arg.n_repeat + 1):
        arg.seed = random_seed
        results.append(run_exp(model=model, args=arg))

    # Dumps the results in appropriate files
    pickle_it(results, result_file, directory)
    print('results saved in "{}"'.format(directory))

    
