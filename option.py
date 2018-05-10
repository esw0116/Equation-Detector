import argparse

parser = argparse.ArgumentParser()

# Basic Configuration/ Load a Fixed Configuration
parser.add_argument('--work_type', type=str, default='Custom', help='Custom/Mixture')

# Set GPU environment, Random Seed for Reproducing
parser.add_argument('--n_threads', type=int, default=2, help='number of threads for data loading')
parser.add_argument('--no_cuda', action='store_true', help='enables CUDA training')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Set Load Path, Save Path
parser.add_argument('--data_path', type=str, default='../../Dataset/benchmark', help='path of input data')
parser.add_argument('--save_path', type=str, default='../../Feature_map', help='path of saving image directory')
parser.add_argument('--regress', action='store_true', help='regression vs classification')

parser.add_argument('--test_only', action='store_true', help='regression vs classification')
parser.add_argument('--data_name', type=str, default='DIV2K', help='Name of Dataset')
parser.add_argument('--testdata_name', type=str, default='DIV2K', help='Name of Test Dataset')
parser.add_argument('--data_range', type=str, default='.', help='range of data')

parser.add_argument('--sr_factor', type=int, default=2, help='Interpolation factor.')
parser.add_argument('--grad_num', type=int, default=3, help='Number of Gradual Upscaling')
parser.add_argument('--ds_num', type=int, default=1, help='Number of HR Father')

parser.add_argument('--repeat', type=int, default=2, help='number of iterations')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size for training')
parser.add_argument('--num_batches', type=int, default=1800, help='Number of batches to run')
parser.add_argument('--patch_size', type=int, default=0, help='patch size when training')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')

parser.add_argument('--learning_rate', type=float, default=0.001, help='Base learning rate for Adam')
parser.add_argument('--decay_step', type=int, default=600, help='Lr decay Step')
parser.add_argument('--gamma', type=float, default=0.5, help='Lr decay gamma')
parser.add_argument('--l_only', action='store_true', help='use only Y channel')

parser.add_argument('--model', type=str, default='ZSSR_Resblock', help='Name of model')
parser.add_argument('--num_channels', type=int, default=3, help='Number of channels of input image')
parser.add_argument('--num_blocks', type=int, default=3, help='Number of Residual blocks in ZSSR')
parser.add_argument('--num_layers_per_block', type=int, default=2, help='Number of hidden layers per block!! in ZSSR')
parser.add_argument('--num_layers', type=int, default=8, help='Number of hidden layers per block!! in ZSSR')
parser.add_argument('--num_feats', type=int, default=128, help='Number of channels of feature map')
parser.add_argument('--print_model', action='store_true', help='print model')

parser.add_argument('--log_dir', type=str, default='./experiment', help='path of pre_trained data')
parser.add_argument('--resume', action='store_true', help='If true, resume from the latest')
parser.add_argument('--pre_train', type=str, default='.', help='path of pre_trained data')

args = parser.parse_args()
