import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import re
from Bio import SeqIO
from matplotlib import pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torchvision
from torch.autograd import Variable
import torch
from torchvision import transforms

def write_fasta(name, sequence_df):
    out_file = open(name + '.fasta', "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df.name[i] + '\n')
        out_file.write(sequence_df.sequence[i] + '\n')
    out_file.close()

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
        return self.decode(z), mu, logvar, z

model = VAE()
#load a model
model.load_state_dict(torch.load("MTS/model/vae_self_tv_sim_split_kl_weight_1_batch_size_128_epochs32.chkpt"))
cdict = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))  
rev_dict = {j:i for i,j in cdict.items()}

#sampling
with torch.no_grad():
    sample = torch.randn(1000, 32)
    sample = model.decode(sample).cpu()
    sample = sample.view(1000, 70, 22)

sampled_seqs = []
for i, seq in enumerate(sample):
    out_seq = []
    for j, pos in enumerate(seq):
        best_idx = pos.argmax()
        out_seq.append(rev_dict[best_idx.item()])
        
    final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
    sampled_seqs.append(final_seq) 

seq_to_check = []
count = 0
for i in range(np.shape(sampled_seqs)[0]):
    if sampled_seqs[i].find('$') == - 1 and sampled_seqs[i].find('0') == - 1:
        count = count + 1
        seq_to_check.append(['sample'+str(count),sampled_seqs[i]])

print(np.shape(seq_to_check))
filtered_seq_to_check = pd.DataFrame(seq_to_check, columns = ['name', 'sequence'])
    
print('Total number of sequences:', len(filtered_seq_to_check))
filtered_seq_to_check = filtered_seq_to_check.drop_duplicates(subset='sequence').reset_index().drop('index', axis=1)
print('Total sequences remaining after duplicate removal', len(filtered_seq_to_check))
write_fasta('MTS/data/amts', filtered_seq_to_check)