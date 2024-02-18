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
import re
from Bio import SeqIO
import umap
from scipy.stats import gaussian_kde
import random
import seaborn as sns
import matplotlib.lines as mlines
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from scipy.interpolate import interp1d

#Functions
def one_hot_encode(seq):
    mapping = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]

def read_fasta_v2(name):
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()])
            
    return data

def write_fasta(name, sequence_df):
    out_file = open(name + '.fasta', "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df['Name'][i] + '\n')
        out_file.write(sequence_df['Sequence'][i] + '\n')
    out_file.close()

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
for i in range(np.shape(train)[0]):
    seq = train[i]+'$'
    pad_seq = seq.ljust(80,'0')
    ohe_train.append(one_hot_encode(pad_seq))

ohe_valid = []
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
        return self.decode(z), mu, logvar, z

model = VAE_dual()

#load a model
model.load_state_dict(torch.load('Dual/model/dual_vae.chkpt'))
cdict = dict(zip("FIWLVMYCATHGSQRKNEPD$0", range(22)))  
rev_dict = {j:i for i,j in cdict.items()}

#Latent Code (Train+Valid Data)
with torch.no_grad():
    sample = torch.FloatTensor(ohe_train)
    output_sample = model.forward(sample)
    lc_train = output_sample[3]
    
with torch.no_grad():
    sample = torch.FloatTensor(ohe_valid)
    output_sample = model.forward(sample)
    lc_valid = output_sample[3]

lc_model = torch.cat((lc_train,lc_valid),0)

#Which one is mTP and which one is cTP? [Note: 1 sequence is dual targeting]
#Data Import
mtp = pd.DataFrame(read_fasta_v2('Dual/data/mtp_matrix_for_ml'), columns = ['Name','Sequence'])
ctp = pd.DataFrame(read_fasta_v2('Dual/data/ctp_stroma_for_ml'), columns = ['Name','Sequence'])

#Overlap between two transit peptide datasets created (As positive control - Q94K73 belongs to A. thaliana)
#Since there is one characterized dual targeting sequence Q94K73 [1-53 AA], duplicate can be removed if needed. But for now, we keep it.
Xm = list(mtp['Name']) 
Xc = list(ctp['Name'])

Xm_seq = list(mtp['Sequence']) 
Xc_seq = list(ctp['Sequence']) 

label_array = []
for i in range(len(train)):
    if train[i] in Xc_seq:
        label_array.append('Chloroplast')
    elif train[i] in Xm_seq:
        label_array.append('Mitochondria')
    else:
        print('?')
    
for i in range(len(valid)):
    if train[i] in Xc_seq:
        label_array.append('Chloroplast')
    elif train[i] in Xm_seq:
        label_array.append('Mitochondria')
    else:
        print('?')

#Visualization
cmap_dict = {'Chloroplast': 'Greens', 'Mitochondria': 'Reds'}
color_dict = {'Chloroplast': 'green', 'Mitochondria': 'red'}

#UMAP reduction and visualization
reducer = umap.UMAP(n_components = 2)
model_data_umap = reducer.fit_transform(lc_model)
model_data_umap = pd.DataFrame(data = model_data_umap, columns=['x1','x2'])
model_data_umap['label'] = label_array

model_data_umap.to_csv('Dual/data/latent_space_umap.csv', index = False)

# Define the distance threshold to determine proximity to centroid
distance_threshold = 0.4

# Dictionary to store the 'set' centroids for each class
centroids = {}
n_clusters_dict = {'Chloroplast': 4, 'Mitochondria': 3}

# Create separate KDE plots for each class
for label in model_data_umap['label'].unique():
    class_data = model_data_umap[model_data_umap['label'] == label]
    
    # Plot the KDE for the class
    sns.kdeplot(data=class_data, x='x1', y='x2', cmap=cmap_dict[label], label=label)
    
    '''
    if label == 'Chloroplast':
        # Calculate the centroid for the class
        centroid = np.mean(class_data[['x1', 'x2']], axis=0)
        centroids[label] = centroid

    else:
    '''
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters_dict[label])  # Adjust the number of clusters as needed
    cluster_labels = kmeans.fit_predict(class_data[['x1', 'x2']])

    # Find the cluster center with the highest density of points
    unique_labels, label_counts = np.unique(cluster_labels, return_counts=True)
    cluster_center = kmeans.cluster_centers_[np.argmax(label_counts)]
    #print("Cluster Center:", cluster_center)

    centroid = cluster_center    
    centroids[label] = centroid

    distances = pairwise_distances_argmin_min(class_data[['x1', 'x2']], [centroid])
    near_centroid_indices = np.where(distances[1] < distance_threshold)[0]
    near_centroid_points = class_data.iloc[near_centroid_indices]

    if label == 'Chloroplast':
        ctp_lv = lc_model[near_centroid_points.index]
    else:
        mtp_lv = lc_model[near_centroid_points.index]

    # Plot the points near the centroid on the KDE plot
    plt.scatter(
        x=near_centroid_points['x1'],
        y=near_centroid_points['x2'],
        color=color_dict[label],
        marker='o',
        s = 3,
        alpha = 0.5,
        label=f'Near Centroid ({label})',
    )

# Add legend
handles = []
for label, color in color_dict.items():
    handles.append(mlines.Line2D([], [], color=color, label=label))
handles.append(mlines.Line2D([], [], color='black', marker='o', linestyle='', label='Selected TPs'))

plt.legend(handles=handles)
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('Dual/data/latent_space_kde_umap_f.png', dpi = 400)

#Explore all possible combinations for N. tabacum and interpolate for 5 points including end points [0,0.4,0.5,0.6,1]
interpolation_len = 5
flag = 0
interpolated_seq_data = []
interpolated_latent_code = []
for i in range(np.shape(mtp_lv)[0]):
    for j in range(np.shape(ctp_lv)[0]):
        check_flag = 0
        lc_m = mtp_lv[i]
        lc_c = ctp_lv[j]
        linfit = interp1d([1,5], np.vstack([lc_m, lc_c]), axis=0)
        
        #sampling
        with torch.no_grad():
            sample_lc = torch.tensor(linfit([1,2.6,3,3.4,5])).float()
            sample = model.decode(sample_lc).cpu()
            sample = sample.view(interpolation_len, 80, 22)
            
        sampled_seqs = []
        for k, seq in enumerate(sample):
            out_seq = []
            for l, pos in enumerate(seq):
                best_idx = pos.argmax()
                out_seq.append(rev_dict[best_idx.item()])

            final_seq = ''.join(out_seq).rstrip('0').rstrip('$')
            sampled_seqs.append(final_seq) 
        
        #Check sampled sequence
        for m in range(interpolation_len):
            if '0' in sampled_seqs[m] or '$' in sampled_seqs[m]:
                check_flag = check_flag + 0
                break
            else:
                check_flag = check_flag + 1
                
        if check_flag/interpolation_len == 1:
            seq_check_flag = 1
        else:
            seq_check_flag = 0
        
        if flag == 0 and seq_check_flag == 1:
            flag = 1
            interpolated_seq_data = [sampled_seqs]
            interpolated_latent_code = sample_lc
        elif flag == 1 and seq_check_flag == 1:
            interpolated_seq_data.append(sampled_seqs)
            interpolated_latent_code = torch.cat((interpolated_latent_code, sample_lc), 0)
            
#Make a file to feed TargetP 2.0
GFP_seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
interpolation_data = []
for i in range(np.shape(interpolated_seq_data)[0]):
    curr_seq_list = [s + GFP_seq[1:] for s in interpolated_seq_data[i]]
    
    for j in range(interpolation_len):
        interpolation_data.append([str(5*i+j+1),curr_seq_list[j]])

interpolation_data = pd.DataFrame(interpolation_data, columns = ['Name', 'Sequence'])

#write fasta
write_fasta('Dual/data/selective_linear_interpolation_0.4_f', interpolation_data)