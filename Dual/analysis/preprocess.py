#Modules
import numpy as np 
import pandas as pd
import re
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns

#Functions
def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id.split('|')[1], str(fasta.seq).strip()])
            
    return data

def validate(seq, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    if (pattern.match(seq)):
        return True
    return False

def clean(sequence_df):
    invalid_seqs = []

    for i in range(len(sequence_df)):
        if (not validate(sequence_df['Sequence'][i])):
            invalid_seqs.append(i)

    print('Total number of sequences dropped:', len(invalid_seqs))
    sequence_df = sequence_df.drop(invalid_seqs).reset_index().drop('index', axis=1)
    print('Total number of sequences remaining:', len(sequence_df))
    
    return sequence_df

def write_fasta(name, sequence_df):
    out_file = open(name + '.fasta', "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df['Name'][i] + '\n')
        out_file.write(sequence_df['Sequence'][i] + '\n')
    out_file.close()
    
def write_fasta_v2(name, sequence_df):
    out_file = open(name + '.fasta', "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df['Name'][i] + '\n')
        out_file.write(sequence_df['Transit Peptide Sequence'][i] + '\n')
    out_file.close()

#Data Import
mtp = pd.read_excel('Dual/data/viridiplantae_matrix_reviewed.xlsx')
ctp = pd.read_excel('Dual/data/chloroplast_stroma_reviewed.xlsx')

mtp_all = pd.DataFrame(read_fasta('Dual/data/viridiplantae_matrix_all'), columns = ['Name','Sequence'])
ctp_all = pd.DataFrame(read_fasta('Dual/data/chloroplast_stroma_all'), columns = ['Name','Sequence'])

#Reviewed Transit Peptide in UniProt
mtp_reviewed = []
for i in range(np.shape(mtp)[0]):
    if type(mtp['Transit peptide'][i]) == str:
        mtp_len = mtp['Transit peptide'][i].split(';')[0].split('..')[1]
        if not '?' in mtp_len: 
            mtp_reviewed.append([mtp['Entry'][i],mtp['Sequence'][i][0:int(mtp_len)]])
    
mtp_reviewed = pd.DataFrame(mtp_reviewed, columns = ['Name','Transit Peptide Sequence'])

ctp_reviewed = []
for i in range(np.shape(ctp)[0]):
    if type(ctp['Transit peptide'][i]) == str:
        ctp_len = ctp['Transit peptide'][i].split(';')[0].split('..')[1]
        if not '?' in ctp_len: 
            ctp_reviewed.append([ctp['Entry'][i],ctp['Sequence'][i][0:int(ctp_len)]])
    
ctp_reviewed = pd.DataFrame(ctp_reviewed, columns = ['Name','Transit Peptide Sequence'])

#removing known transit peptide from Uniprot before feeding to TargetP 2.0
mtp_all = mtp_all[~mtp_all.Name.isin(list(mtp_reviewed['Name']))].reset_index(drop=True)
ctp_all = ctp_all[~ctp_all.Name.isin(list(ctp_reviewed['Name']))].reset_index(drop=True)

#Removing Invalid Sequences
print(len(mtp_all))
mtp_all = mtp_all.drop_duplicates(subset=['Sequence'], keep='first').reset_index(drop=True) 
print(len(mtp_all))
mtp_all = clean(mtp_all).reset_index(drop=True)
mtp_for_targetp = mtp_all[mtp_all['Sequence'].astype(str).str.startswith('M')].reset_index(drop=True)
print(len(mtp_for_targetp))

print(' ')

print(len(ctp_all))
ctp_all = ctp_all.drop_duplicates(subset=['Sequence'], keep='first').reset_index(drop=True) 
print(len(ctp_all))
ctp_all = clean(ctp_all).reset_index(drop=True)
ctp_for_targetp = ctp_all[ctp_all['Sequence'].astype(str).str.startswith('M')].reset_index(drop=True)
print(len(ctp_for_targetp))

#write fasta
write_fasta('Dual/data/dual_mtp_for_targetp', mtp_for_targetp)
write_fasta('Dual/data/dual_ctp_for_targetp', ctp_for_targetp)

#Fed these faste files to TargetP 2.0 software to obtain .targetp2 files

#TargetP2.0 result analysis (MTP)
targetp_mtp = open("Dual/data/dual_mtp_for_targetp_summary.targetp2", "r",encoding="utf-8")
targetp_mtp = pd.read_csv(targetp_mtp,sep='\t')
targetp_mtp = targetp_mtp.reset_index()
new_header_m = targetp_mtp.iloc[0] 
targetp_mtp = targetp_mtp[1:].reset_index(drop=True) 
targetp_mtp.columns = new_header_m

selected_mtp_df = targetp_mtp[targetp_mtp.Prediction.str.contains("mTP")].to_numpy()
print(selected_mtp_df.shape[0]/targetp_mtp.shape[0])

plt.hist(targetp_mtp['mTP'].astype(float),20)
plt.xlabel('Probability of being an MTS as predicted by TargetP 2.0')
plt.ylabel('Number of Artificial sequences')

mtp_from_targetp = []
for i in range(np.shape(targetp_mtp)[0]):
    if targetp_mtp['Prediction'][i] == 'mTP':
        curr_seq = mtp_for_targetp.loc[mtp_for_targetp['Name'] == targetp_mtp['# ID'][i], 'Sequence'].values[0]
        curr_len = int(targetp_mtp['CS Position'][i].split('-')[0].split('CS pos: ')[1])
        if curr_len < 80:
            mtp_from_targetp.append([targetp_mtp['# ID'][i],curr_seq[:curr_len]])
        
mtp_from_targetp = pd.DataFrame(mtp_from_targetp, columns = ['Name','Transit Peptide Sequence'])
mtp_reviewed_f = mtp_reviewed[mtp_reviewed['Transit Peptide Sequence'].str.len() < 80]
mtp_matrix = pd.concat([mtp_reviewed_f, mtp_from_targetp], axis=0).reset_index(drop=True) 

#TargetP2.0 result analysis (CTP)
targetp_ctp = open("Dual/data/dual_ctp_for_targetp_summary.targetp2", "r",encoding="utf-8")
targetp_ctp = pd.read_csv(targetp_ctp,sep='\t')
targetp_ctp = targetp_ctp.reset_index()
new_header_c = targetp_ctp.iloc[0] 
targetp_ctp = targetp_ctp[1:].reset_index(drop=True) 
targetp_ctp.columns = new_header_c

selected_ctp_df = targetp_ctp[targetp_ctp.Prediction.str.contains("cTP")].to_numpy()
print(selected_ctp_df.shape[0]/targetp_ctp.shape[0])

plt.hist(targetp_ctp['mTP'].astype(float),20)
plt.xlabel('Probability of being an MTS as predicted by TargetP 2.0')
plt.ylabel('Number of Artificial sequences')

ctp_from_targetp = []
for i in range(np.shape(targetp_ctp)[0]):
    if targetp_ctp['Prediction'][i] == 'cTP':
        curr_seq = ctp_for_targetp.loc[ctp_for_targetp['Name'] == targetp_ctp['# ID'][i], 'Sequence'].values[0]
        curr_len = int(targetp_ctp['CS Position'][i].split('-')[0].split('CS pos: ')[1])
        if curr_len < 80:
            ctp_from_targetp.append([targetp_ctp['# ID'][i],curr_seq[:curr_len]])
        
ctp_from_targetp = pd.DataFrame(ctp_from_targetp, columns = ['Name','Transit Peptide Sequence'])
ctp_reviewed_f = ctp_reviewed[ctp_reviewed['Transit Peptide Sequence'].str.len() < 80]
ctp_stroma = pd.concat([ctp_reviewed_f, ctp_from_targetp], axis=0).reset_index(drop=True) 

#write fasta
write_fasta_v2('Dual/data/mtp_matrix_for_ml', mtp_matrix)
write_fasta_v2('Dual/data/ctp_stroma_for_ml', ctp_stroma)