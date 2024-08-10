import torch
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
class Student(nn.Module):
    def __init__(self, input_size=50, hidden_size=256, num_layers=4, action_space=2, device=device, dropout=0.5):
        super(Student, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.rnn_1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)

        self.fc = nn.Linear(hidden_size, action_space)
        self.dropout_1 = nn.Dropout(dropout)
        self.h0 = None

    def forward(self, x):
        #print(x.size())
        h0 = torch.zeros(self.num_layers, len(x), self.hidden_size).to(self.device)
        #print(h0.size())
        #  GRU
        x, _ = self.rnn_1(x, h0)  # out (batch_size, seq_length, hidden_size)
        out = self.dropout_1(x)
        out = self.fc(out[:, -1, :])
        return out



if __name__ == "__main__":
    network = Student()
    print(network)
