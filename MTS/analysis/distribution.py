import pandas as pd
import numpy as np
import pickle
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

with open('MTS/data/tv_sim_split_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('MTS/data/tv_sim_split_valid.pkl', 'rb') as f:
    X_valid = pickle.load(f)

final_df = pd.concat([X_train, X_valid], ignore_index=True).reset_index(drop=True)
mtss_len = list(final_df['length'])

print(np.min(mtss_len))
print(np.max(mtss_len))
print(np.mean(mtss_len))

plt.figure(figsize=(9, 6))
plt.xlabel('Length (in AA)', fontsize=12)
plt.ylabel('Density',fontsize=12)

#sns.distplot(mtss_len, hist=True, kde=True, rug=False, label = 'Curated MTS Dataset')
sns.distplot(list(X_train['length']), hist=True, kde=True, rug=False, label = 'Train')
sns.distplot(list(X_valid['length']), hist=True, kde=True, rug=False, label = 'Valid')
plt.legend(fontsize=12)
plt.savefig('MTS/data/Curated_MTS_dataset_stratified_split.png', dpi=400, bbox_inches = "tight")
plt.clf()

uniprot = pd.read_excel('MTS/data/uniprot_transit_peptide.xlsx', header = None) 
uniprot_tp = []
for i in range(1,np.shape(uniprot)[0]):
    
    #filtering for Mitochondrion and defined transit peptide length
    tp_end = uniprot[3][i].split('..')[1].split(';')[0]
    organelle = uniprot[3][i].split('note="')[1].split('";')[0]
    
    if tp_end.find('?') == -1:
        if organelle.find('Mitochondrion') > -1 and int(tp_end) > 5:
            tp_name = uniprot[1][i]
            tp_seq = uniprot[2][i][:int(tp_end)]

            uniprot_tp.append([tp_name,tp_seq])
    
uniprot_tp = pd.DataFrame(uniprot_tp, columns = ['name', 'sequence'])
print(len(uniprot_tp))
species = [item.split('_')[1] for item in list(uniprot_tp['name'])]
'''
set1 = set(list(final_df['name']))
set2 = set([item.split('_')[0] for item in list(uniprot_tp['name'])])

# Check the overlap
overlap = set1.intersection(set2)
print(len(overlap))

uniprot_len = list(uniprot_tp['sequence'].str.len())
sns.distplot(uniprot_len, hist=True, kde=True, rug=False, label = 'UniProt (unfiltered)')
'''

# Count occurrences of each species
species_counts = Counter(species)
#print(species_counts)
filtered_species_counts = {species: count for species, count in species_counts.items() if count > 100}
filtered_species_counts = dict(sorted(filtered_species_counts.items(), key=lambda item: item[1], reverse=True))

others_count = sum(count for count in species_counts.values() if count <= 100)
filtered_species_counts['Others (' + str(len(species_counts)-len(filtered_species_counts)) + ')'] = others_count

# Set a lighter color palette using Seaborn
sns.set_palette("pastel")

plt.pie(filtered_species_counts.values(), labels=filtered_species_counts.keys(), autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8}, pctdistance=0.8)
plt.axis('equal')
plt.savefig('MTS/data/UniProt_Species_proportion.png', dpi=400, bbox_inches = "tight")


