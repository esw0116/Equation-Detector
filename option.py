import argparse

parser = argparse.ArgumentParser()

# Basic Configuration/ Load a Fixed Configuration
parser.add_argument('--work_type', type=str, default='Character', help='Custom/Mixture')

# Set GPU environment, Random Seed for Reproducing
parser.add_argument('--n_threads', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--cpu_only', action='store_true', help='enables CUDA training')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Set Load Path, Save Path
parser.add_argument('--data_path', type=str, default='./Dataset', help='path of input data')
parser.add_argument('--save_path', type=str, default='../../Feature_map', help='path of saving image directory')

parser.add_argument('--test_only', action='store_true', help='test only mode')
parser.add_argument('--load', action='store_true', help='load the latest model params')
parser.add_argument('--load_path', type=str, default='./experiment/20180525_baseline_002', help='load path')

parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
parser.add_argument('--num_batches', type=int, default=50000, help='Number of batches to run')

parser.add_argument('--learning_rate', type=float, default=0.002, help='Base learning rate for Adam')
parser.add_argument('--decay_step', type=int, default=2000, help='Lr decay Step')
parser.add_argument('--gamma', type=float, default=0.5, help='Lr decay gamma')

parser.add_argument('--model', type=str, default='baseline', help='Name of model')
parser.add_argument('--print_model', action='store_true', help='print model')

parser.add_argument('--log_dir', type=str, default='./experiment', help='path of pre_trained data')
parser.add_argument('--pre_train', type=str, default='.', help='path of pre_trained data')

args = parser.parse_args()
