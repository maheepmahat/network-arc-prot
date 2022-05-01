import torch, math
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

delta = 8
epochs = 150
bsize= 32
freq_printing = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        h0 = (torch.zeros(1,input_seq.size(1),self.hidden_layer_size).to(device),
                            torch.zeros(1,input_seq.size(1),self.hidden_layer_size).to(device))
        lstm_out, h0 = self.lstm(input_seq, h0)
        predictions = self.linear(lstm_out[-1])
        return predictions

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoder = nn.Linear(ninp, 1)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = output[-1]
        output = self.decoder(output)
        return output
    
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

def get_batch(source, i):
    inp, target = [], []
    seq_len = min(bsize, len(source) - 1 - i)
    for t in range(i, i+seq_len):
        inp.append(source[t][0].unsqueeze(-1))
        target.append(source[t][1])
    inp = torch.stack(inp, 1)
    target = torch.stack(target).flatten()
    return inp.to(device), target.to(device)

if __name__ == '__main__':
    # loading training data
    df = pd.read_csv('train.csv', sep=',')
    load = df['Lane 1 Flow (Veh/5 Minutes)'].values.astype(float)
    hourly_load = load.reshape((-1, 12)).mean(1)

    # loading test data
    df2 = pd.read_csv('test.csv', sep=',')
    load2 = df2['Lane 1 Flow (Veh/5 Minutes)'].values.astype(float)
    hourly_load2 = load2.reshape((-1, 12)).mean(1)

    # normalizing training and test data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_data = scaler.fit_transform(hourly_load.reshape(-1, 1))
    train_data = torch.FloatTensor(train_data).flatten()
    test_data = scaler.transform(hourly_load2.reshape(-1, 1))
    test_data = torch.FloatTensor(test_data).view(-1, delta).to(device)
    
    train_inout_seq = create_inout_sequences(train_data, delta)
    
    # Defining LSTM model
    lstm_model = LSTM().to(device)
    lstm_loss = nn.MSELoss()
    lstm_opt = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    
    # training LSTM model
    for i in range(epochs):
        tot_loss = 0.0
        for s in range(0, 623, bsize):
            seq, labels = get_batch(train_inout_seq, s)
            y_pred = lstm_model(seq).squeeze()
            
            loss = lstm_loss(y_pred, labels)
            lstm_opt.zero_grad()
            loss.backward()
            lstm_opt.step()
            tot_loss += loss.item()

        if i%freq_printing == 0:
            print(f'epoch: {i:3} loss: {tot_loss:10.8f}')

    # Defining Transformer model
    transformer_model = TransformerModel(100, 10, 10, 1, 0.2).to(device)    
    transformer_loss = nn.MSELoss()
    transformer_opt = torch.optim.AdamW(transformer_model.parameters(), lr=5e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(transformer_opt, 1.0, gamma=0.95)
    src_mask = transformer_model.generate_square_subsequent_mask(delta).to(device)

    # Training Transformer model
    for i in range(epochs):
        tot_loss = 0.0
        for s in range(0, 623, bsize):
            seq, labels = get_batch(train_inout_seq, s)
            y_pred = transformer_model(seq, src_mask).squeeze()

            loss = transformer_loss(y_pred, labels)
            transformer_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 0.7)
            transformer_opt.step()
            tot_loss += loss.item()
    
        if i%freq_printing == 0:
            print(f'epoch: {i:3} loss: {tot_loss:10.8f}')
        scheduler.step()