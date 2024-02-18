import pandas as pd
import numpy as np
import re
from Bio import SeqIO
from Bio import pairwise2 as pw2
import seaborn as sns
import matplotlib.pyplot as plt

#Functions
def validate(seq, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    if (pattern.match(seq)):
        return True
    return False

def clean(sequence_df):
    invalid_seqs = []

    for i in range(len(sequence_df)):
        if (not validate(sequence_df.sequence[i])):
            invalid_seqs.append(i)
    
    print('Total number of sequences dropped:', len(invalid_seqs))
    sequence_df = sequence_df.drop(invalid_seqs).reset_index().drop('index', axis=1)
    print('Total number of sequences remaining:', len(sequence_df))
    
    return sequence_df

def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name + '.fa'),'fasta')
    data = []
    egfp = 'VSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip().replace(egfp,'')])
            
    return data

uniprot = pd.read_excel('MTS/data/uniprot_transit_peptide.xlsx', header = None) 
vae_tp = pd.DataFrame(read_fasta('MTS/data/egfp_artificial_sequences'), columns = ['Name','Sequence'])

uniprot_tp = []
for i in range(1,np.shape(uniprot)[0]):
    
    #filtering for Mitochondrion and defined transit peptide length
    tp_end = uniprot[3][i].split('..')[1].split(';')[0]
    organelle = uniprot[3][i].split('note="')[1].split('";')[0]
    
    if tp_end.find('?') == -1:
        if organelle.find('Mitochondrion') > -1 and int(tp_end) > 5:
            tp_name = uniprot[0][i]
            tp_seq = uniprot[2][i][:int(tp_end)]
         
    uniprot_tp.append([tp_name,tp_seq])
    
uniprot_tp = pd.DataFrame(uniprot_tp, columns = ['name', 'sequence'])

print('Total number of sequences remaining:', len(uniprot_tp))
uniprot_tp = uniprot_tp.drop_duplicates(subset='sequence').reset_index().drop('index', axis=1)
print('Total sequences remaining after duplicate removal', len(uniprot_tp))

#Levenshtein
from Levenshtein import distance as lv

min_lev_h = [] 
min_perc_lev_h = [] 
for i in range(np.shape(vae_tp)[0]):
    lev_dist = []
    perc_lev_dist = []
    for j in range(np.shape(uniprot_tp)[0]):
        lev_dist.append(lv(vae_tp['Sequence'][i],uniprot_tp['sequence'][j]))
    
    min_lev_h.append(np.min(lev_dist))
    
#print(np.mean(min_perc_lev_h))

import pickle
with open('MTS/data/tv_sim_split_train.pkl', 'rb') as f:
    X_train = pickle.load(f)

min_lev = [] 
for i in range(np.shape(vae_tp)[0]):
    lev_dist = []
    for j in range(np.shape(X_train)[0]):
        lev_dist.append(lv(vae_tp['Sequence'][i],X_train['sequence'][j]))
    
    min_lev.append(np.min(lev_dist))

vae_tp_len = list(vae_tp['Sequence'].str.len())

plt.figure(figsize=(9, 6))
plt.xticks(fontsize=18)  # Font size for x-axis labels
plt.yticks(fontsize=18)  # Font size for y-axis labels

sns.distplot(vae_tp_len, hist=True, kde=True, rug=False, label = 'Length')
sns.distplot(min_lev, hist=True, kde=True, rug=False, label = 'Distance to training data')
sns.distplot(min_lev_h, hist=True, kde=True, rug=False, label = 'Distance to MTSs in UniProt')

plt.legend(fontsize=12)
plt.savefig('MTS/data/Edit Distance.png', dpi=400, bbox_inches = "tight")