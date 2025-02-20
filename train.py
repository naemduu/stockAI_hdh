import os
import sys
if os.getcwd() not in sys.path:
    sys.path.insert(1, os.getcwd())
    print(os.getcwd())
import argparse
from pathlib import Path
from learning import train


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=32, help='Batch size')

    parser.add_argument('-e', '--num-epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--data-path', type=str, default='saves/candle_1d/AAPL_daily.csv', help='csv file location')

    parser.add_argument('-r', '--report-interval', type=int, default=20, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_transformer', help='Name of network')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 현재 등록된 sys.path 출력
    args = parse()
    train(args)
