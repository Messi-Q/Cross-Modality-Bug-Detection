import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart contract vulnerability detection based on Cross-Modal Learning')
    parser.add_argument('-D', '--dataset', type=str, default='reentrancy',
                        choices=['timestamp', 'integeroveflow', 'delegatecall', 'reentrancy'])
    parser.add_argument('--m',  type=str, default='BERT',
                        choices=['BERT', 'GCN', 'GPT', 'word2vec'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay')
    parser.add_argument('--epoch', type=int, default=50, help='number of epoch')
    parser.add_argument('-b', '--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--cuda', default=False, help='using cuda')
    parser.add_argument('--seed', type=int, default=9930, help='random seed')
    parser.add_argument('--shuffle', action='store_true', default=True, help='shuffle dataset')

    return parser.parse_args()
