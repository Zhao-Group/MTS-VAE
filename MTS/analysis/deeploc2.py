import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
import random
from collections import Counter

vae_tp = pd.read_csv('MTS/data/results_20231212-090514.csv')
phmm_tp = pd.read_csv('MTS/data/results_20231212-093420.csv')
modlamp_tp = pd.read_csv('MTS/data/results_20241208-163526.csv')

vae_prob = list(vae_tp['Mitochondrion'])
hmm_prob = list(phmm_tp['Mitochondrion'])
modlamp_prob = list(modlamp_tp['Mitochondrion'])

hmm_prob_f = random.sample(hmm_prob, len(vae_prob))

#Percent probability of being functional (>0.5)
print('Probability of pHMM generated MTS to be functional:')
print(len([i for i,v in enumerate(hmm_prob_f) if v > 0.6373])*100/len(hmm_prob_f))

print('Probability of modlAMP Helices generated MTS to be functional:')
print(len([i for i,v in enumerate(modlamp_prob) if v > 0.6373])*100/len(modlamp_prob))

print('Probability of VAE generated MTS to be functional:')
print(len([i for i,v in enumerate(vae_prob) if v > 0.6373])*100/len(vae_prob))

plt.figure(figsize=(9, 6))
plt.xticks(fontsize=18)  # Font size for x-axis labels
plt.yticks(fontsize=18)  # Font size for y-axis labels

plt.axvline(x=0.6373, color='black', linestyle='--', label='Cut-Off')

num_bins = 20 
sns.distplot(vae_prob, hist=True, bins = num_bins, kde=False, rug=False, label='VAE', kde_kws={'cut': 0})
sns.distplot(hmm_prob_f, hist=True, bins = num_bins, kde=False, rug=False, label='pHMM', kde_kws={'cut': 0})
sns.distplot(modlamp_prob, hist=True, bins = num_bins, kde=False, rug=False, label='modlAMP Helices', kde_kws={'cut': 0})

plt.legend(fontsize=12)
plt.savefig('MTS/data/insilico_DeepLoc_prob_MTS_distribution.png', dpi=400, bbox_inches = "tight")

vae_prob = list(vae_tp['Plastid'])
hmm_prob = list(phmm_tp['Plastid'])

hmm_prob_f = random.sample(hmm_prob, len(vae_prob))

#Percent probability of being functional (>0.5)
print('Probability of pHMM generated CTP to be functional:')
print(len([i for i,v in enumerate(hmm_prob) if v > 0.6586])*100/len(hmm_prob_f))

print('Probability of VAE generated CTP to be functional:')
print(len([i for i,v in enumerate(vae_prob) if v > 0.6586])*100/len(vae_prob))

plt.figure(figsize=(9, 6))
plt.xticks(fontsize=18)  # Font size for x-axis labels
plt.yticks(fontsize=18)  # Font size for y-axis labels

plt.axvline(x=0.6373, color='black', linestyle='--', label='Cut-Off')

num_bins = 20 
sns.distplot(vae_prob, hist=True, bins = num_bins, kde=False, rug=False, label='VAE', kde_kws={'cut': 0})
sns.distplot(hmm_prob_f, hist=True, bins = num_bins, kde=False, rug=False, label='pHMM', kde_kws={'cut': 0})

plt.legend(fontsize=12)
#plt.savefig('MTS/data/insilico_DeepLoc_prob_CTP_distribution.png', dpi=400, bbox_inches = "tight")

predicted_localization = list(vae_tp['Localizations'])
print('Probability of VAE generated MTS-GFP protein to be in mitochondrion:')
print(len([i for i,v in enumerate(predicted_localization) if v == 'Mitochondrion'])*100/len(predicted_localization))
label_counts = Counter(predicted_localization)

# Sort labels based on counts in descending order
sorted_labels = sorted(label_counts, key=lambda x: label_counts[x], reverse=True)

sns.set(style="white")
plt.figure(figsize=(9, 6))
ax = sns.countplot(x=predicted_localization, color='#A2DEA5', order=sorted_labels)

# Remove top and right spines
sns.despine(top=True, right=True)

# Rotate xticks by 75 degrees
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right", fontsize = 15)
ax.set_xlim(left=-0.7)

# Set labels and title
#plt.xlabel('Localization')
plt.ylabel('Count')

# Show the plot
plt.savefig('MTS/data/insilico_DeepLoc_prob_Localization_VAE_distribution.png', dpi=400, bbox_inches = "tight")

predicted_localization_all = list(phmm_tp['Localizations'])
predicted_localization = random.sample(predicted_localization_all, len(vae_prob))

print('Probability of pHMM generated MTS-GFP protein to be in mitochondrion:')
print(len([i for i,v in enumerate(predicted_localization) if v == 'Mitochondrion'])*100/len(predicted_localization))

label_counts = Counter(predicted_localization)

# Sort labels based on counts in descending order
sorted_labels = sorted(label_counts, key=lambda x: label_counts[x], reverse=True)

sns.set(style="white")
plt.figure(figsize=(9, 6))
ax = sns.countplot(x=predicted_localization, color='#A2DEA5', order=sorted_labels)

# Remove top and right spines
sns.despine(top=True, right=True)

# Rotate xticks by 75 degrees
ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha="right", fontsize = 15)
ax.set_xlim(left=-0.7)

# Set labels and title
#plt.xlabel('Localization')
plt.ylabel('Count')

# Show the plot
plt.savefig('MTS/data/insilico_DeepLoc_prob_Localization_pHMM_distribution.png', dpi=400, bbox_inches = "tight")


