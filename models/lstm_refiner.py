import torch
import torch.nn as nn


class dLSTM(nn.Module):
    def __init__(self, batch_size, input_size, hidden_dim, device, drop, seq_len=10):
        super(dLSTM, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.sequence_len = seq_len
        self.input_size = input_size
        self.lstm_cells = []
        self.lstm1 = nn.LSTMCell(input_size, hidden_dim)
        self.lstm2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=drop)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, input_size)

    def forward(self, x, hc):
        output_seq = torch.empty((self.sequence_len, 
                                  self.batch_size, 
                                  self.input_size)).to(self.device)
        # pass the hidden and the cell state from one lstm cell to the next one
        # we also feed the output of the first layer lstm cell at time step t to the second layer cell
        # init the both layer cells with the zero hidden and zero cell states
        hc_1, hc_2 = hc, hc
        
        # for every step in the sequence
        for t in range(self.sequence_len):
            hc_1 = self.lstm1(x[t], hc_1)
            h_1, c_1 = hc_1
            hc_2 = self.lstm2(h_1, hc_2)
            h_2, c_2 = hc_2
            # form the output of the fc
            output_seq[t] = self.linear(self.dropout(h_2))
        return output_seq[-1]
    
    def init_hidden(self, device):
        # initialize the hidden state and the cell state to zeros
        return (torch.zeros(self.batch_size, self.hidden_dim).to(device),
                torch.zeros(self.batch_size, self.hidden_dim).to(device))
