import numpy as np
import pandas as pd
import umap
import pickle
from Bio import SeqIO
import re

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

from scipy.spatial import distance
arrays = np.load('MTS/data/artificial_mts_for_organisms_10.npz', allow_pickle=True) 

#S. cerevisiae
amts_org = pd.read_csv('MTS/data/amts_final_sc.csv')
curr_amts = amts_org[['name','sequence']] 

embd_for_amts = []
amts_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(curr_amts['name']):
        amts_data_embd_arranged.append(list(curr_amts.loc[curr_amts['name'] == i].values[0]))
        embd_for_amts.append(arrays[i].item()['avg'])
        
amts_data_embd_arranged_df = pd.DataFrame(amts_data_embd_arranged, columns = ['name','sequence'])

amts_label = []
amts_distance = []
for i in range(np.shape(curr_amts)[0]):
    curr_d = []
    for j in range(org_number):
        curr_d.append(distance.euclidean(embd_for_amts[i], cluster_center[j])) 
        
    if np.argmin(curr_d)+1 == 1:
        amts_distance.append(curr_d[0])
        
amts_org['distance_cluster_center'] =  amts_distance
sorted_amts = amts_org.sort_values('distance_cluster_center').reset_index(drop=True)

#AMTS selection
pack_count = 0

amts_seq_to_test = []

for i in range(np.shape(sorted_amts)[0]):
    if pack_count < 8:  
        pack_count = pack_count + 1
        amts_seq_to_test.append(sorted_amts.iloc[i].values)
            
amts_to_test = pd.DataFrame(amts_seq_to_test, columns = sorted_amts.columns)

#Sequence Diversity
nmts_sc = seqs_df[seqs_df['name'].str.contains('YEAST')].reset_index(drop=True)
from Levenshtein import distance as lv

dist_closest_nmts = []
for j in range(np.shape(amts_to_test)[0]):
    AA_distance = []
    for i in range(np.shape(nmts_sc)[0]):
        AA_distance.append(lv(nmts_sc['sequence'][i],amts_to_test['sequence'][j]))

    dist_closest_nmts.append(np.min(AA_distance))
    
amts_to_test['Distance_to_closest_natural_mts'] = dist_closest_nmts
amts_to_test.to_csv('MTS/data/amts_to_test_sc.csv',index=False)

#R. toruloides
amts_org = pd.read_csv('MTS/data/amts_final_rt.csv')
curr_amts = amts_org[['name','sequence']] 

embd_for_amts = []
amts_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(curr_amts['name']):
        amts_data_embd_arranged.append(list(curr_amts.loc[curr_amts['name'] == i].values[0]))
        embd_for_amts.append(arrays[i].item()['avg'])
        
amts_data_embd_arranged_df = pd.DataFrame(amts_data_embd_arranged, columns = ['name','sequence'])

amts_label = []
amts_distance = []
for i in range(np.shape(curr_amts)[0]):
    curr_d = []
    for j in range(org_number):
        curr_d.append(distance.euclidean(embd_for_amts[i], cluster_center[j])) 
        
    if np.argmin(curr_d)+1 == 2:
        amts_distance.append(curr_d[1])
        
amts_org['distance_cluster_center'] =  amts_distance
sorted_amts = amts_org.sort_values('distance_cluster_center').reset_index(drop=True)

#AMTS selection
pack_count = 0

amts_seq_to_test = []

for i in range(np.shape(sorted_amts)[0]):
    if pack_count < 8:  
        pack_count = pack_count + 1
        amts_seq_to_test.append(sorted_amts.iloc[i].values)
            
amts_to_test = pd.DataFrame(amts_seq_to_test, columns = sorted_amts.columns)

#Sequence Diversity
nmts_rt = seqs_df[seqs_df['name'].str.contains('Rhoto')].reset_index(drop=True)
from Levenshtein import distance as lv

dist_closest_nmts = []
for j in range(np.shape(amts_to_test)[0]):
    AA_distance = []
    for i in range(np.shape(nmts_rt)[0]):
        AA_distance.append(lv(nmts_rt['sequence'][i],amts_to_test['sequence'][j]))

    dist_closest_nmts.append(np.min(AA_distance))
    
amts_to_test['Distance_to_closest_natural_mts'] = dist_closest_nmts
amts_to_test.to_csv('MTS/data/amts_to_test_rt.csv',index=False)

#Human
amts_org = pd.read_csv('MTS/data/amts_final_h.csv')
curr_amts = amts_org[['name','sequence']] 

embd_for_amts = []
amts_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(curr_amts['name']):
        amts_data_embd_arranged.append(list(curr_amts.loc[curr_amts['name'] == i].values[0]))
        embd_for_amts.append(arrays[i].item()['avg'])
        
amts_data_embd_arranged_df = pd.DataFrame(amts_data_embd_arranged, columns = ['name','sequence'])

amts_label = []
amts_distance = []
for i in range(np.shape(curr_amts)[0]):
    curr_d = []
    for j in range(org_number):
        curr_d.append(distance.euclidean(embd_for_amts[i], cluster_center[j])) 
        
    if np.argmin(curr_d)+1 == 3:
        amts_distance.append(curr_d[2])
        
amts_org['distance_cluster_center'] =  amts_distance
sorted_amts = amts_org.sort_values('distance_cluster_center').reset_index(drop=True)

#AMTS selection
pack_count = 0

amts_seq_to_test = []

for i in range(np.shape(sorted_amts)[0]):
    if pack_count < 8:  
        #if sorted_amts['sequence'][i][0:2] == 'MA': #To preserve Kozak Sequence
        pack_count = pack_count + 1
        amts_seq_to_test.append(sorted_amts.iloc[i].values)

amts_to_test = pd.DataFrame(amts_seq_to_test, columns = sorted_amts.columns)

#Sequence Diversity
nmts_h = seqs_df[seqs_df['name'].str.contains('HUMAN')].reset_index(drop=True)
from Levenshtein import distance as lv

dist_closest_nmts = []
for j in range(np.shape(amts_to_test)[0]):
    AA_distance = []
    for i in range(np.shape(nmts_h)[0]):
        AA_distance.append(lv(nmts_h['sequence'][i],amts_to_test['sequence'][j]))

    dist_closest_nmts.append(np.min(AA_distance))
    
amts_to_test['Distance_to_closest_natural_mts'] = dist_closest_nmts
amts_to_test.to_csv('MTS/data/amts_to_test_h.csv',index=False)

#Tobacco
amts_org = pd.read_csv('MTS/data/amts_final_nt.csv')
curr_amts = amts_org[['name','sequence']] 

embd_for_amts = []
amts_data_embd_arranged = []
for i in list(arrays.keys()):
    if i in list(curr_amts['name']):
        amts_data_embd_arranged.append(list(curr_amts.loc[curr_amts['name'] == i].values[0]))
        embd_for_amts.append(arrays[i].item()['avg'])
        
amts_data_embd_arranged_df = pd.DataFrame(amts_data_embd_arranged, columns = ['name','sequence'])

amts_label = []
amts_distance = []
for i in range(np.shape(curr_amts)[0]):
    curr_d = []
    for j in range(org_number):
        curr_d.append(distance.euclidean(embd_for_amts[i], cluster_center[j])) 
        
    if np.argmin(curr_d)+1 == 4:
        amts_distance.append(curr_d[3])
        
amts_org['distance_cluster_center'] =  amts_distance
sorted_amts = amts_org.sort_values('distance_cluster_center').reset_index(drop=True)

#AMTS selection
pack_count = 0

amts_seq_to_test = []

for i in range(np.shape(sorted_amts)[0]):
    if pack_count < 8:  
        pack_count = pack_count + 1
        amts_seq_to_test.append(sorted_amts.iloc[i].values)
            
amts_to_test = pd.DataFrame(amts_seq_to_test, columns = sorted_amts.columns)

#Sequence Diversity
nmts_nt = seqs_df[seqs_df['name'].str.contains('TOBAC')].reset_index(drop=True)
from Levenshtein import distance as lv

dist_closest_nmts = []
for j in range(np.shape(amts_to_test)[0]):
    AA_distance = []
    for i in range(np.shape(nmts_nt)[0]):
        AA_distance.append(lv(nmts_nt['sequence'][i],amts_to_test['sequence'][j]))

    dist_closest_nmts.append(np.min(AA_distance))
    
amts_to_test['Distance_to_closest_natural_mts'] = dist_closest_nmts
amts_to_test.to_csv('MTS/data/amts_to_test_nt.csv',index=False)