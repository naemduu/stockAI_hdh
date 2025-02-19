import torch.nn as nn
import torch


class StockTransformer(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3):
        super(StockTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_price = nn.Linear(d_model, 1)
        self.fc_sigma = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        price_pred = self.fc_price(x[:, -1, :])
        confidence_pred = torch.exp(self.fc_sigma(x[:, -1, :]))
        return price_pred.squeeze(-1), confidence_pred.squeeze(-1)