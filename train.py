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
    parser.add_argument('--data-path', type=Path, default='saves/candle_1d/AAPL_daily_20250130204757.csv', help='csv file locaiton')

    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_Unet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/image/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/image/', help='Directory of validation data')
    parser.add_argument('--in-chans', type=int, default=1, help='Size of input channels for network')
    parser.add_argument('--out-chans', type=int, default=1, help='Size of output channels for network')
    parser.add_argument('--input-key', type=str, default='image_input', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 현재 등록된 sys.path 출력
    args = parse()
    train(args)
