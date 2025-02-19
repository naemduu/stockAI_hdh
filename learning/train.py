import torch
import torch.nn as nn
import torch.optim as optim
from learning.load_data import create_data_loaders
from models import DNNModel, StockTransformer


def train(args):
    # 데이터 로딩
    data_loader = create_data_loaders(args.data_path, args)

    # 모델 초기화
    model = StockTransformer(input_dim=5)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 학습 루프
    for epoch in range(args.num_epochs):
        for batch in data_loader:
            inputs, labels = batch
            optimizer.zero_grad()
            price, sigma = model(inputs)
            loss = (price - labels)**2 / sigma + sigma
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')