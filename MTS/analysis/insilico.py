import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

vae = pd.read_excel('MTS/data/mitofates_pred_amts_vae.xlsx', header = None) 
hmm = pd.read_excel('MTS/data/mitofates_pred_amts_hmm.xlsx', header = None) 

vae_prob = []
for i in range(1, len(vae)):
    vae_prob.append(float(vae[0][i].split()[1]))

hmm_prob = []
for i in range(1, len(hmm)):
    hmm_prob.append(float(hmm[0][i].split()[1]))

hmm_prob_f = random.sample(hmm_prob, len(vae_prob))

#Percent probability of being functional (>0.5)
print('Probability of pHMM generated MTS to be functional:')
print(len([i for i,v in enumerate(hmm_prob) if v > 0.5])*100/len(hmm_prob_f))

print('Probability of VAE generated MTS to be functional:')
print(len([i for i,v in enumerate(vae_prob) if v > 0.5])*100/len(vae_prob))


plt.figure(figsize=(9, 6))
plt.xticks(fontsize=18)  # Font size for x-axis labels
plt.yticks(fontsize=18)  # Font size for y-axis labels

num_bins = 20 
sns.distplot(vae_prob, hist=True, bins = num_bins, kde=False, rug=False, label='VAE', kde_kws={'cut': 0})
sns.distplot(hmm_prob_f, hist=True, bins = num_bins, kde=False, rug=False, label='pHMM', kde_kws={'cut': 0})

plt.legend(fontsize=12)
plt.savefig('MTS/data/insilico_prob_MTS_distribution.png', dpi=400, bbox_inches = "tight")



