import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        
class Student(nn.Module):
    def __init__(self, input_size=50, hidden_size=256, num_layers=4, action_space=2, num_gaussians=8, device=device, dropout=0.8):
        super(Student, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.action_space = action_space
        self.num_gaussians = num_gaussians
        self.device = device
        self.h0 = None

        self.rnn_1 = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)


        self.z_pi = nn.Sequential(
            nn.Linear(hidden_size, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.z_mu = nn.Linear(hidden_size, num_gaussians * action_space)
        self.z_sigma = nn.Linear(hidden_size, num_gaussians * action_space)



    def forward(self, x):
        h0 = torch.zeros(self.num_layers, len(x), self.hidden_size).to(self.device)
        #  GRU

        x, _ = self.rnn_1(x, h0)  # out (batch_size, seq_length, hidden_size)
        pi = self.z_pi(x[:, -1, :])

        mu = self.z_mu(x[:, -1, :])
        mu = mu.view(-1, self.num_gaussians, self.action_space)
        sigma = torch.exp(self.z_sigma(x[:, -1, :]))  #positive
        sigma = sigma.view(-1, self.num_gaussians, self.action_space)
        

        return mu, sigma, pi


def mdn_loss(pi, sigma, mu, target):
    mix = D.Categorical(pi)    
    comp = D.Independent(D.Normal(mu,sigma), 1)
    gmm = D.MixtureSameFamily(mix, comp)
    return -gmm.log_prob(target).mean()



def mdn_sample(mu, sigma, pi):
    mix=D.Categorical(pi)
    comp=D.Independent(D.Normal(mu,sigma), 1)
    gmm=D.MixtureSameFamily(mix,comp)
    return gmm.sample()


if __name__ == "__main__":
    network = Student(input_size=2, hidden_size=2, num_layers=4, action_space=2, num_gaussians=8, device=device, dropout=0.8).to(device)
    test = torch.tensor([[[0.1,0.1],[0.1,0.1]],[[0.2,0.2],[0.2,0.2]]]).to(device)
    
    print("input_size: ", test.size())
    mu,sigma,pi = network(test)
    print("mu: ", mu)
    print("mu_size: ",mu.size())
    print("sigma: ", sigma)
    print("sigma_size: ",sigma.size())
    print("pi: ", pi)
    print("pi: ",pi.size())

    y = torch.tensor([[0.1,0.1],[0.1,0.1]]).to(device)
    loss = mdn_loss(pi, sigma, mu, y)
    print(loss)
    result = mdn_sample(mu,sigma,pi)
    print(result)
    #print(network)
