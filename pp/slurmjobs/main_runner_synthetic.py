import sys
import argparse
from run import run_linear_synth


def main():
    parser = argparse.ArgumentParser(description='Get experiment configs')
    # arguments for synthesizing dataset
    parser.add_argument('--numdatapoints', type = int, help = 'size of the synthetic dataset to construct')
    parser.add_argument('--dimension', type = int, help = 'dimensionality of the synthetic data')
    parser.add_argument('--labelsigma', type = float, default = 0.0, help = 'std of the gaussian noise added to synthetic linear label')
    parser.add_argument('-seed', type = int, default = 21, help = 'seed for randomness in data generation and split')
    # algorithm specific arguments
    parser.add_argument('--runs', type=int, default = 10000, help = 'number of runs, each run computes the private estimate on the data')
    parser.add_argument('--traintestsplit', type = float, default = 0.1, help = 'fraction desired for test set, default is 90 percent to trainset, 10 percent test')
    
    #these parameter are varied in out experiments
    parser.add_argument('--lambreg', type = float, help = 'regularization(lamdbda) used in ridge regression')
    parser.add_argument('--frac_train', type=float, default = 1.0, help = 'fraction of training set to use for computing the private estimate')
    parser.add_argument('--f_c', type = float, help = 'fraction of high privacy("conservative") users in the training data')
    parser.add_argument('--f_m', type = float, help = 'fraction of medium privacy("medium") users in the training data')
    parser.add_argument('--eps_c', type = float, help = 'high privacy users epsilon is uniformly sampled ~ [eps_c, eps_m]')
    parser.add_argument('--eps_m', type = float, help = 'medium privacy users epsilon is uniformly sampled ~ [eps_m, eps_l]')
    parser.add_argument('--eps_l', type = float, default = 1.0, help = 'epsilon value for low privacy users, default is 1.0')

    parser.add_argument('--save_dir', type = str, default = '../../slurm-synth/', help= 'directory in which to store synthetic experimnents')
    
    args = parser.parse_args()

    print("synthesizing dataset args", args.numdatapoints, args.dimension, args.labelsigma) 
    print("algorithm specific args", args.runs, args.traintestsplit)
    print("privacy parameters", args.lambreg, args.frac_train, args.f_c, args.f_m, args.eps_c, args.eps_m, args.eps_l)

if __name__ == '__main__':
    main()