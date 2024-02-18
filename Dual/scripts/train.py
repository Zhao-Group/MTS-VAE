#Modules
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import torch
import pickle
import time

#Functions
def one_hot_encode(seq):
    mapping = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]

#Data Loading
path = 'Dual/data/'
file_name_t = 'train.pkl'
file_name_v = 'valid.pkl'

open_file = open(path + file_name_t, "rb")
train = pickle.load(open_file)
open_file.close()

open_file = open(path + file_name_v, "rb")
valid = pickle.load(open_file)
open_file.close()

#Encoding
ohe_train = []
np.random.shuffle(train)   
for i in range(np.shape(train)[0]):
    seq = train[i]+'$'
    pad_seq = seq.ljust(80,'0')
    ohe_train.append(one_hot_encode(pad_seq))

ohe_valid = []
np.random.shuffle(valid)   
for i in range(np.shape(valid)[0]):
    seq = valid[i]+'$'
    pad_seq = seq.ljust(80,'0')
    ohe_valid.append(one_hot_encode(pad_seq))
    
#model
class VAE_dual(nn.Module):
    def __init__(self):
        super(VAE_dual, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(1760, 256)
        self.fc21 = nn.Linear(256, 32)
        self.fc22 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 256)
        self.fc4 = nn.Linear(256, 1760)
        self.dropout = nn.Dropout(0.3)
        
    def encode(self, x):
        h1 = self.dropout(self.relu(self.fc1(x)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1760))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
model = VAE_dual()

##Number of parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

batch_size = 32
epochs = 100
optimizer = optim.Adam(model.parameters(), lr=0.001)  

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1760), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(model, X, optimizer):
    model.train()
    epoch_loss = 0
    np.random.shuffle(X)
    
    num_batches = np.shape(X)[0] // batch_size
    
    for i in tqdm(range(0, np.shape(X)[0], batch_size)): 
        batch_X = torch.FloatTensor(X)[i:i+batch_size].view(-1, 1760)
        optimizer.zero_grad() 
        recon_batch, mu, logvar = model(batch_X) 
        loss = loss_function(recon_batch, batch_X, mu, logvar)
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
            
    return epoch_loss / num_batches

def evaluate(model, X):
    model.eval()
    epoch_loss = 0
    
    num_batches = np.shape(X)[0] // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, np.shape(X)[0], batch_size)): 
            batch_X = torch.FloatTensor(X)[i:i+batch_size].view(-1, 1760)
            recon_batch, mu, logvar = model(batch_X)
            loss = loss_function(recon_batch, batch_X, mu, logvar)
            epoch_loss += loss.item()
                            
    return epoch_loss / num_batches

best_valid_loss = float('inf')
early_stop = 5
count = 0
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

for epoch in range(epochs):
    
    start_time = time.time()
    
    train_loss = train(model, ohe_train, optimizer)
    valid_loss = evaluate(model, ohe_valid)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        count = 0 
        torch.save(model.state_dict(), 'Dual/model/dual_vae.chkpt')
        
    else: 
        count = count + 1
        
    if count == early_stop:
        print('Early Stop Limit Reached')
        break
    
    print(f"Time: {epoch_mins}m {epoch_secs}s")
    print(f"Epoch: {epoch}. Train Loss: {train_loss}\n")
    print(f"Epoch: {epoch}. Valid Loss: {valid_loss}\n")