import numpy as np
import pandas as pd
import umap
import pickle
from Bio import SeqIO
import re

def validate(seq, pattern=re.compile(r'^[FIWLVMYCATHGSQRKNEPD]+$')):
    if (pattern.match(seq)):
        return True
    return False

def clean(sequence_df):
    invalid_seqs = []

    for i in range(len(sequence_df)):
        if (not validate(sequence_df['zf_sequence'][i])):
            invalid_seqs.append(i)

    print('Total number of sequences dropped:', len(invalid_seqs))
    sequence_df = sequence_df.drop(invalid_seqs).reset_index().drop('index', axis=1)
    print('Total number of sequences remaining:', len(sequence_df))
    
    return sequence_df

def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()])
            
    return data

in_fasta = 'MTS/data/model_organism_sequences_mts'
seqs_df_total = pd.DataFrame(read_fasta(in_fasta), columns = ['name', 'sequence'])

org_label = []
index_label = [] 
for i in range(np.shape(seqs_df_total)[0]):
    if 'YEAST' in seqs_df_total['name'][i]:
        org_label.append(1)
        index_label.append(i)
    elif 'Rhoto' in seqs_df_total['name'][i]:
        org_label.append(2)
        index_label.append(i)
    elif 'HUMAN' in seqs_df_total['name'][i]:
        org_label.append(3)
        index_label.append(i)
    elif 'TOBAC' in seqs_df_total['name'][i]:
        org_label.append(4)
        index_label.append(i)
        
seqs_df = seqs_df_total.iloc[index_label].reset_index(drop=True)
seqs_df['label'] = org_label

final_seqs_df = seqs_df[seqs_df.sequence.str.startswith('M')]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
final_seqs_df['label'].value_counts().plot(ax=ax, kind='bar')

arrays = np.load('MTS/data/model_organism_sequences_mts.npz', allow_pickle=True) 

embd_for_cluster = []
cluster_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(final_seqs_df['name']):
        cluster_data_embd_arranged.append(list(final_seqs_df.loc[final_seqs_df['name'] == i].values[0]))
        embd_for_cluster.append(arrays[i].item()['avg'])
        
cluster_data_embd_arranged_df = pd.DataFrame(cluster_data_embd_arranged, columns = ['name','sequence','label'])

reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(embd_for_cluster)

df_cluster_rd_unirep_umap = pd.DataFrame(data=embedding, columns=['x1','x2'])

cluster_center = []
org_number = 4
for i in range(org_number):
    curr_index = cluster_data_embd_arranged_df.index[cluster_data_embd_arranged_df['label'] == i+1].tolist()
    for j in range(len(curr_index)):
        if j == 0:
            cc_vector = embd_for_cluster[curr_index[j]]
        else:
            cc_vector = cc_vector + embd_for_cluster[curr_index[j]]
    
    if i == 0:
        cluster_center = cc_vector/len(curr_index)
    else:
        cluster_center = np.vstack((cluster_center,cc_vector/len(curr_index)))
        
print(cluster_center.shape)

file_no = 10
in_fasta = 'MTS/data/artificial_mts_for_organisms_' + str(file_no)
ini_amts_df = pd.DataFrame(read_fasta(in_fasta), columns = ['name', 'sequence'])

amts_df = ini_amts_df[ini_amts_df.sequence.str.startswith('M')]

arrays = np.load('MTS/data/artificial_mts_for_organisms_' + str(file_no) + '.npz', allow_pickle=True) 
embd_for_amts = []
amts_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(amts_df['name']):
        amts_data_embd_arranged.append(list(amts_df.loc[amts_df['name'] == i].values[0]))
        embd_for_amts.append(arrays[i].item()['avg'])
        
amts_data_embd_arranged_df = pd.DataFrame(amts_data_embd_arranged, columns = ['name','sequence'])

from scipy.spatial import distance
amts_label = []
selected_amts_index = []
for i in range(np.shape(amts_df)[0]):
    curr_d = []
    for j in range(org_number):
        curr_d.append(distance.euclidean(embd_for_amts[i], cluster_center[j])) 
    
    curr_d_density = []
    for j in range(np.shape(embd_for_cluster)[0]):
        curr_d_density.append(distance.euclidean(embd_for_amts[i], embd_for_cluster[j])) 
        
    top_name_list = list(cluster_data_embd_arranged_df['name'][np.argpartition(curr_d_density, 20)[:20]])
    c1 = sum('YEAST' in s for s in top_name_list)
    c2 = sum('Rhoto' in s for s in top_name_list)
    c3 = sum('HUMAN' in s for s in top_name_list)
    c4 = sum('TOBAC' in s for s in top_name_list)
    count_organism = [c1, c2, c3, c4]
    
    if np.argmin(curr_d) == np.argmax(count_organism):
        selected_amts_index.append(i)
        amts_label.append(np.argmin(curr_d)+1)
        
amts_data_embd_arranged_cf = amts_data_embd_arranged_df.iloc[selected_amts_index].reset_index(drop=True)
amts_data_embd_arranged_cf['org_label'] = amts_label

embd_for_amts_clustering = list(np.array(embd_for_amts)[selected_amts_index])

fig, ax = plt.subplots()
amts_data_embd_arranged_cf['org_label'].value_counts().plot(ax=ax, kind='bar')

embd_for_amts_clustering = list(np.array(embd_for_amts)[selected_amts_index])

fig, ax = plt.subplots()
amts_data_embd_arranged_cf['org_label'].value_counts().plot(ax=ax, kind='bar')

cc_unirep_umap = reducer.transform(cluster_center)
cc_unirep_umap = pd.DataFrame(data=cc_unirep_umap, columns=['x1', 'x2'])
print(cc_unirep_umap.shape)

#amts_unirep_umap = reducer.transform(embd_for_amts_clustering)
#amts_unirep_umap = pd.DataFrame(data=amts_unirep_umap, columns=['x1', 'x2'])
#print(amts_unirep_umap.shape)

amts_to_test_h = pd.read_csv('amts_to_test_h.csv')  
amts_to_test_sc = pd.read_csv('amts_to_test_sc.csv')  
amts_to_test_rt = pd.read_csv('amts_to_test_rt.csv')  
amts_to_test_nt = pd.read_csv('amts_to_test_nt.csv')  

frames = [amts_to_test_h, amts_to_test_sc, amts_to_test_rt, amts_to_test_nt]
amts_to_test = pd.concat(frames).reset_index(drop=True)

arrays = np.load('MTS/data/artificial_mts_for_organisms_' + str(file_no) + '.npz', allow_pickle=True) 
embd_for_amts_to_test = []
amts_to_test_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(amts_to_test['name']):
        amts_to_test_data_embd_arranged.append(list(amts_to_test.loc[amts_to_test['name'] == i].values[0]))
        embd_for_amts_to_test.append(arrays[i].item()['avg'])
        
amts_to_test_data_embd_arranged = pd.DataFrame(amts_to_test_data_embd_arranged, columns = ['name','sequence','mTP Probability','Cleavage Probability','length','org_label','distance_cluster_center','Distance_to_closest_natural_mts'])
amts_to_test_unirep_umap = reducer.transform(embd_for_amts_to_test)
amts_to_test_unirep_umap = pd.DataFrame(data=amts_to_test_unirep_umap, columns=['x1', 'x2'])
print(amts_to_test_unirep_umap.shape)

org_label_rep = []
for i in range(np.shape(amts_to_test_data_embd_arranged)[0]):
    if 'YEAST' == amts_to_test_data_embd_arranged['org_label'][i]:
        org_label_rep.append(1)
    elif 'Rhoto' == amts_to_test_data_embd_arranged['org_label'][i]:
        org_label_rep.append(2)
    elif 'HUMAN' == amts_to_test_data_embd_arranged['org_label'][i]:
        org_label_rep.append(3)
    elif 'TOBAC' == amts_to_test_data_embd_arranged['org_label'][i]:
        org_label_rep.append(4)
        
amts_to_test_data_embd_arranged['org_label_rep'] = org_label_rep

from scipy.stats import gaussian_kde
import matplotlib.cm as cm 

fig, ax = plt.subplots()
#ax = fig.add_subplot(111, projection='3d')
for g in np.unique(cluster_data_embd_arranged_df['label']):
    ix = np.where(cluster_data_embd_arranged_df['label'] == g)
    x = df_cluster_rd_unirep_umap['x1'].loc[ix]
    y = df_cluster_rd_unirep_umap['x2'].loc[ix]
    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    if g == 1:
        ax.scatter(x, y, c = z, cmap=cm.RdPu, s = 10, alpha = 0.4)
    elif g == 2:
        ax.scatter(x, y, c = z, cmap=cm.Oranges, s = 10, alpha = 0.4)
    elif g == 3:
        ax.scatter(x, y, c = z, cmap=cm.Blues, s = 10, alpha = 0.4)
    else:
        ax.scatter(x, y, c = z, cmap=cm.Greens, s = 10, alpha = 0.4)
        
cdict = {1: 'magenta', 2: 'orange', 3: 'blue', 4: 'green'}
for g in np.unique(cluster_data_embd_arranged_df['label']):
    ax.scatter(cc_unirep_umap['x1'][g-1], cc_unirep_umap['x2'][g-1], c = cdict[g], label = g, marker = '*', s = 200)

for g in np.unique(amts_to_test_data_embd_arranged['org_label_rep']):
    ix = np.where(amts_to_test_data_embd_arranged['org_label_rep'] == g)
    ax.scatter(amts_to_test_unirep_umap['x1'].loc[ix], amts_to_test_unirep_umap['x2'].loc[ix], c = cdict[g], marker = 'P', label = g, s = 50, alpha = 1)

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('MTS/data/clustering_amts_to_test', dpi=400)

org_label = []
for i in range(np.shape(amts_data_embd_arranged_cf)[0]):
    if amts_data_embd_arranged_cf['org_label'][i] == 1:
        org_label.append('YEAST')
    elif amts_data_embd_arranged_cf['org_label'][i] == 2:
        org_label.append('Rhoto')
    elif amts_data_embd_arranged_cf['org_label'][i] == 3:
        org_label.append('HUMAN')
    elif amts_data_embd_arranged_cf['org_label'][i] == 4:
        org_label.append('TOBAC')
        
amts_data_embd_arranged_cf['org_label'] = org_label

amts_data_embd_arranged_cf.to_csv('MTS/data/amts_labeled_cluster_final.csv',index=False)