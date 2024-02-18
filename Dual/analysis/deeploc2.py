import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO

dual_tp = pd.read_csv('Dual/data/results_20230702-120651.csv')

def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()])
            
    return data

df = pd.DataFrame(read_fasta('Dual/data/selective_linear_interpolation_0.4_f'), columns = ['Name','Sequence'])

I0_m_prob = []
I1_m_prob = []
I2_m_prob = []
I3_m_prob = []
I4_m_prob = []

I0_c_prob = []
I1_c_prob = []
I2_c_prob = []
I3_c_prob = []
I4_c_prob = []

count = 0
for i in range(int(len(dual_tp)/5)):
    if dual_tp['Plastid'][5*i+2] > 0.6586 and dual_tp['Mitochondrion'][5*i+2] > 0.6373:
        I0_m_prob.append(dual_tp['Mitochondrion'][5*i])
        I1_m_prob.append(dual_tp['Mitochondrion'][5*i+1])
        I2_m_prob.append(dual_tp['Mitochondrion'][5*i+2])
        I3_m_prob.append(dual_tp['Mitochondrion'][5*i+3])
        I4_m_prob.append(dual_tp['Mitochondrion'][5*i+4])

        I0_c_prob.append(dual_tp['Plastid'][5*i])
        I1_c_prob.append(dual_tp['Plastid'][5*i+1])
        I2_c_prob.append(dual_tp['Plastid'][5*i+2])
        I3_c_prob.append(dual_tp['Plastid'][5*i+3])
        I4_c_prob.append(dual_tp['Plastid'][5*i+4])

        count = count + 1

print(count)
print(int(len(dual_tp)/5))

prob_m_list = I0_m_prob + I1_m_prob + I2_m_prob + I3_m_prob +I4_m_prob
label_m_list = ['M']*len(I0_m_prob) + ['I1']*len(I0_m_prob) + ['I2']*len(I0_m_prob) + ['I3']*len(I0_m_prob) + ['C']*len(I0_m_prob)
prob_m_df = pd.DataFrame(list(zip(label_m_list, prob_m_list)), columns = ['Label', 'Probability (Mitochondria)'])

ax = sns.boxplot(data=prob_m_df, y="Probability (Mitochondria)", x="Label", palette = 'RdYlGn')
ax = sns.swarmplot(x="Label", y="Probability (Mitochondria)", data=prob_m_df, color="0", alpha=.35, s = 2)
figure = ax.get_figure()
figure.savefig('Dual/data/selective_linear_interpolation_0.4_dual_MTP_probability.png', dpi=400, bbox_inches = "tight")
plt.clf()

prob_c_list = I0_c_prob + I1_c_prob + I2_c_prob + I3_c_prob +I4_c_prob
label_c_list = ['M']*len(I0_c_prob) + ['I1']*len(I0_c_prob) + ['I2']*len(I0_c_prob) + ['I3']*len(I0_c_prob) + ['C']*len(I0_c_prob)
prob_c_df = pd.DataFrame(list(zip(label_c_list, prob_c_list)), columns = ['Label', 'Probability (Plastid)'])

ax = sns.boxplot(data=prob_c_df, y="Probability (Plastid)", x="Label", palette = 'RdYlGn')
ax = sns.swarmplot(x="Label", y="Probability (Plastid)", data=prob_c_df, color="0", alpha=.35, s = 2)
figure = ax.get_figure()
figure.savefig('Dual/data/selective_linear_interpolation_0.4_dual_CTP_probability.png', dpi=400, bbox_inches = "tight")
plt.clf()

indices = []
for i in range(int(len(dual_tp)/5)):
    if dual_tp['Plastid'][5*i+2] > 0.6586 and dual_tp['Mitochondrion'][5*i+2] > 0.6373:
        for j in range(5):
            indices.append(5*i+j)

dual_tp_df = df.iloc[indices].reset_index(drop=True)
#dual_tp_df.to_csv('Dual/data/dual_tps.csv', index = False)

indices = []
for i in range(int(len(dual_tp)/5)):
    if dual_tp['Plastid'][5*i+2] > 0.6586 and dual_tp['Mitochondrion'][5*i+2] > 0.6373:
        indices.append(5*i+2)

dtp_df = df.iloc[indices].reset_index(drop=True)
dtp_df['Name'] = dtp_df['Name'].astype(int)

merged_df = pd.merge(dtp_df, dual_tp, left_on='Name', right_on='Protein_ID', how='inner')[['Name', 'Sequence', 'Localizations','Mitochondrion', 'Plastid']]

GFP_seq = 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK'
merged_df['Sequence'] = merged_df['Sequence'].str.replace(GFP_seq[1:], '')
merged_df.to_csv('Dual/data/dual_tps_manuscript.csv', index = False)

