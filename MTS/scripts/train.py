import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch

with open('MTS/data/tv_sim_split_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('MTS/data/tv_sim_split_valid.pkl', 'rb') as f:
    X_valid = pickle.load(f)

def one_hot_encode(seq):
    mapping = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]

X_ohe_train = []
for i in range(np.shape(X_train)[0]):
    seq = X_train.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_train.append(one_hot_encode(pad_seq))

X_ohe_valid = []
for i in range(np.shape(X_valid)[0]):
    seq = X_valid.sequence[i]+'$'
    pad_seq = seq.ljust(70,'0')
    X_ohe_valid.append(one_hot_encode(pad_seq))
    
#model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1540, 512)
        self.fc21 = nn.Linear(512, 32)
        self.fc22 = nn.Linear(512, 32)
        self.fc3 = nn.Linear(32, 512)
        self.fc4 = nn.Linear(512, 1540)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1540))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
batch_size = 128
epochs = 35
model = VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)  

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1540), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

BCE = []
w = []
KLD = []
f = open("MTS/data/loss_w1.txt", "a")
for epoch in range(epochs):
    train_loss = 0
    kl_weight_count = 0
    for i in tqdm(range(0, len(X_train), batch_size)): 
        kl_weight_count = kl_weight_count + 1
        batch_X = torch.FloatTensor(X_ohe_train)[i:i+batch_size].view(-1, 1540)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(batch_X) 
        loss = loss_function(recon_batch, batch_X, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    f.write(f"Epoch: {epoch}. Train Loss: {train_loss/len(X_train)}\n")
    torch.save(model.state_dict(), "MTS/model/vae_self_tv_sim_split_kl_weight_1_batch_size_" + str(batch_size) + "_epochs" + str(epoch) + ".chkpt")
    
model.eval()
with torch.no_grad():
    for epoch in range(epochs):
        model.load_state_dict(torch.load("MTS/model/vae_self_tv_sim_split_kl_weight_1_batch_size_"+ str(batch_size) + "_epochs" + str(epoch)+".chkpt"))
        optimizer = optim.Adam(model.parameters(), lr=1e-3)  
        valid_loss = 0
        for i in tqdm(range(0, len(X_valid), batch_size)): 
            batchv_X = torch.FloatTensor(X_ohe_valid)[i:i+batch_size].view(-1, 1540)
            recon_batch, mu, logvar = model(batchv_X)
            loss = loss_function(recon_batch, batchv_X, mu, logvar)
            valid_loss += loss.item()
    
        f.write(f"Epoch: {epoch}. Valid Loss: {valid_loss/len(X_valid)}\n")

f.close()