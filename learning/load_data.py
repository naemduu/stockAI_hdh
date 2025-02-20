import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import utils


class StockDataset(Dataset):
    def __init__(self, csv_file, eval, seq_len=100):
        csv_file = utils.file_path(csv_file)
        EVAL_LENGTH = 200 + seq_len + 5
        if eval:
            self.data = pd.read_csv(csv_file)[2:2+EVAL_LENGTH]
        else:
            self.data = pd.read_csv(csv_file)[2+EVAL_LENGTH:]
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 5

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 입력 데이터 (예: Open, High, Low, Volume)
        inputs = self.data.iloc[idx:idx+self.seq_len, [1, 2, 3, 4, 5]].values  # Open, High, Low, Volume
        inputs = inputs.astype('float32').flatten().reshape(25, 20)  # 데이터 타입을 float32로 변환

        # 타겟 데이터 (5일 후 Open 가격 변화율)
        future = self.data.iloc[idx+self.seq_len+4, [4]].values.astype('float32')[0]
        latest = self.data.iloc[idx+self.seq_len-1, [4]].values.astype('float32')[0]
        labels = future/latest

        return torch.tensor(inputs, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def create_data_loaders(data_path, args, shuffle=False, eval=False, isforward=False):
    dataset = StockDataset(csv_file=data_path, eval=eval)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)
    return data_loader
