import argparse
from template import set_template

parser = argparse.ArgumentParser()
#Template
parser.add_argument('--template', type=str, default='.', help="load one's template")

# Basic Configuration/ Load a Fixed Configuration
parser.add_argument('--work_type', type=str, default='Character', help='CNN/RNN')

# Set GPU environment, Random Seed for Reproducing
parser.add_argument('--n_threads', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--cpu_only', action='store_true', help='enables CUDA training')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Set Load Path, Save Path
parser.add_argument('--data_path', type=str, default='./Dataset', help='path of input data')
parser.add_argument('--save_path', type=str, default='../../Feature_map', help='path of saving image directory')

parser.add_argument('--test_only', action='store_true', help='test only mode')
parser.add_argument('--load_path', type=str, default='20180525_baseline_002', help='load path')
parser.add_argument('--log_dir', type=str, default='./experiment', help='path of pre_trained data')

parser.add_argument('--batch_size', type=int, default=256, help='input batch size for training')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of batches to run')

parser.add_argument('--learning_rate', type=float, default=0.01, help='Base learning rate for Adam')
parser.add_argument('--decay_step', type=int, default=2000, help='Lr decay Step')
parser.add_argument('--gamma', type=float, default=0.5, help='Lr decay gamma')

parser.add_argument('--model', type=str, default='baseline', help='Name of model')
parser.add_argument('--print_model', action='store_true', help='print model')

parser.add_argument('--pre_train', type=str, default='.', help='path of pre_trained data if load from other directory')
parser.add_argument('--dictionary', type=list, default=list(), help='Dictionary of elements')

args = parser.parse_args()
set_template(args)

specials = ['{', '}', '(', ')', '[', ']', '<', '>', r'\{', r'\}',
            '+', '-', r'\pm', r'\times', r'\div', '=', r'\neq', r'\leq', r'\geq',
            '.', '_', '^', '&', '|', '/', "'", ",", '!', r'\prime', r'\frac',
            r'\cos', r'\sin', r'\tan', r'\log', r'\lim', r'\sqrt', r'\sum', r'\int',
            r'\cdot', r'\ldots',
            r'\forall', r'\in', r'\infty', r'\arrow', r'\to', r'\exists']
greeks = [r'\alpha', r'\beta', r'\gamma', r'\Delta', r'\lambda', r'\theta', r'\pi', r'\mu', r'\sigma', r'\phi']
numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

tokens = ['<pad>', '<start>', '<end>', '<unk>']
args.dictionary = list(tokens + sorted(specials+greeks+numbers+letters, key=len, reverse=True))
