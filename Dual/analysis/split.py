#Modules
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from Bio import SeqIO
import re
from sklearn.model_selection import train_test_split
import seaborn as sns
import pickle

#Functions
def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()])
            
    return data

#Data Import
mtp = pd.DataFrame(read_fasta('Dual/data/mtp_matrix_for_ml'), columns = ['Name','Sequence'])
ctp = pd.DataFrame(read_fasta('Dual/data/ctp_stroma_for_ml'), columns = ['Name','Sequence'])

#Overlap between two transit peptide datasets created (As positive control - Q94K73 belongs to A. thaliana)
#Since there is one characterized dual targeting sequence Q94K73 [1-53 AA], duplicate can be removed. But for now, we keep it.
Xm = list(mtp['Name']) 
Xc = list(ctp['Name'])
print(list(set(Xm) & set(Xc)))

#split into train and valid individually
ym = list(mtp['Sequence']) 
yc = list(ctp['Sequence'])

bins_m = np.linspace(0, len(ym) , 10)
bins_c = np.linspace(0, len(yc) , 10)

len_y_m = [len(i) for i in ym]
len_y_c = [len(i) for i in yc]
    
len_y_m_binned = np.digitize(len_y_m, bins_m)
len_y_c_binned = np.digitize(len_y_c, bins_c)

Xm_train, Xm_valid, ym_train, ym_valid = train_test_split(Xm, ym, test_size=0.10, stratify=len_y_m_binned)
Xc_train, Xc_valid, yc_train, yc_valid = train_test_split(Xc, yc, test_size=0.10, stratify=len_y_c_binned)

#Combine matrix and stroma TRAIN and VALID
y_train = ym_train + yc_train
y_valid = ym_valid + yc_valid

#storing training and valid
file_name_t = "Dual/data/train.pkl"
file_name_v = "Dual/data/valid.pkl"

open_file = open(file_name_t, "wb")
pickle.dump(y_train, open_file)
open_file.close()

open_file = open(file_name_v, "wb")
pickle.dump(y_valid, open_file)
open_file.close()

plt.figure(figsize=(9, 6))

#Mito
sns.distplot([len(i) for i in ym_train], hist=False, rug=False, label='Train_MTS')
sns.distplot([len(i) for i in ym_valid], hist=False, rug=False, label='Valid_MTS')

#Chloro
sns.distplot([len(i) for i in yc_train], hist=False, rug=False, label='Train_CTS')
sns.distplot([len(i) for i in yc_valid], hist=False, rug=False, label='Valid_CTS')

plt.legend(fontsize=12)
plt.xlabel('Length (in AA)', fontsize=12)
plt.savefig('Dual/data/Dual_VAE_Data_Split.png', dpi=400, bbox_inches = "tight")
