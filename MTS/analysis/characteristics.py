##Analyzing peptide composition, net charge, secondary structure properties and more
#Modules 
import numpy as np
import pandas as pd 
import re
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from matplotlib.patches import Patch

#Function
def calculate_amino_acid_fraction(peptide):
    prot_param = ProteinAnalysis(peptide)
    amino_acid_fraction = prot_param.get_amino_acids_percent()
    return amino_acid_fraction

def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()])
            
    return data

amts_df = pd.DataFrame(read_fasta('MTS/data/amts_exploration'), columns = ['Name','Sequence'])
mts_df = pd.DataFrame(read_fasta('MTS/data/mts_train'), columns = ['Name','Sequence'])
amts_df['Label'] = 'AMTS'
mts_df['Label'] = 'MTS Training'

df = pd.concat([amts_df, mts_df], ignore_index=True).reset_index(drop = True)

#Biopython
ss_fraction = []
net_charge = []
gravy = []
eisenberg_hydrophobicity = []
for i in range(len(df)):
    prot_seq = ProteinAnalysis(df['Sequence'][i])
    net_charge.append(prot_seq.charge_at_pH(7.0))
    gravy.append(prot_seq.gravy())
    eisenberg_hydrophobicity.append(prot_seq.gravy(scale='Eisenberg'))

df['Net Charge'] = net_charge
df['GRAVY'] = gravy
df['Eisenberg hydrophobicity'] = eisenberg_hydrophobicity
df = pd.concat([df, df['Sequence'].apply(calculate_amino_acid_fraction).apply(pd.Series)], axis=1)

labels = np.unique(df['Label'])
label_colors = {'MTS Training': '#A2DEA5', 'AMTS': 'mistyrose'}
colors_dict = {'MTS Training': '#58a365', 'AMTS': '#d4664f'}

#Net charge
for i in range(len(labels)):
    # Select the data for the current label
    data = df[df['Label'] == labels[i]]['Net Charge']
    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, kde_kws={'bw': 0.2}, color = colors_dict[labels[i]], hist_kws={"alpha": 0.25}) 

# Set the title and labels for the plot
plt.xlabel('Net Charge')
plt.ylabel('Count')
plt.legend()
plt.savefig('MTS/data/diversity_net_charge_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Amino acid composition
positions = list(range(1, 21))
gap = 0.2
widths = 0.15
amino_acid_columns = df.columns[-20:]

plt.figure(figsize=(12, 4.8))
plt.boxplot(df[df['Label'] == labels[0]][amino_acid_columns], positions=[pos-0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor= label_colors[labels[0]]), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == labels[1]][amino_acid_columns], positions=[pos+0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor= label_colors[labels[1]]), flierprops={'markersize': 2})
    
# Set the title and labels for the plot
plt.xticks(range(1, len(amino_acid_columns) + 1), amino_acid_columns)
plt.xlabel('Amino Acids')
plt.ylabel('Fraction')
plt.grid(visible=False, axis='both')

legend_elements = [
    Patch(facecolor= label_colors[labels[0]], label=labels[0]),
    Patch(facecolor= label_colors[labels[1]], label=labels[1])
]
plt.legend(handles=legend_elements)
plt.savefig('MTS/data/diversity_aa_fraction_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

plt.figure(figsize=(6.4, 4.8))
#GRAVY
for i in range(len(labels)):
    # Select the data for the current label
    data = df[df['Label'] == labels[i]]['GRAVY']

    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, color = colors_dict[labels[i]], hist_kws={"alpha": 0.25})

# Set the title and labels for the plot
plt.xlabel('GRAVY')
plt.ylabel('Count')
plt.legend()
plt.savefig('MTS/data/diversity_gravy_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Eisenberg hydrophobicity
for i in range(len(labels)):
    # Select the data for the current label
    data = df[df['Label'] == labels[i]]['Eisenberg hydrophobicity']

    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, color = colors_dict[labels[i]], hist_kws={"alpha": 0.25})

# Set the title and labels for the plot
plt.xlabel('Eisenberg hydrophobicity')
plt.ylabel('Count')
plt.legend()
plt.savefig('MTS/data/diversity_Eisenberg_hydrophobicity_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Length
df['Length'] = df['Sequence'].apply(len)
for l in df['Label'].unique():
    class_data = df[df['Label'] == l]
    sns.distplot(list(class_data['Length']), bins = 10, hist=True, kde=True, rug=False, label = l, color = colors_dict[l])

plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('MTS/data/diversity_Length.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#s4pred secondary structure
# Function to calculate secondary structure element percentages
def calculate_secondary_structure_percentages(structure):
    length = len(structure)
    c_count = structure.count("C")
    h_count = structure.count("H")
    e_count = structure.count("E")
    
    c_percentage = (c_count / length) * 100
    h_percentage = (h_count / length) * 100
    e_percentage = (e_count / length) * 100
    
    return c_percentage, h_percentage, e_percentage

data = {"Name": [], "Sequence": [], "Structure": []}
with open("MTS/data/diversity_cluster_ss.fas", "r") as file:
    lines = file.readlines()

    for i in range(0, len(lines), 3):
        protein_name = lines[i].strip()[1:]
        protein_sequence = lines[i + 1].strip()
        protein_structure = lines[i + 2].strip()

        data["Name"].append(protein_name)
        data["Sequence"].append(protein_sequence)
        data["Structure"].append(protein_structure)

s4pred_df = pd.DataFrame(data)
def lowercase_sample(name):
    if 'SAMPLE' in name:
        return name.lower()
    return name

# Apply the function to the 'name' column
s4pred_df['Name'] = s4pred_df['Name'].apply(lambda x: lowercase_sample(x))

combined_df = df.merge(s4pred_df[['Name', 'Structure']], on='Name', how='left')

coil = []
helix = []
strand = []
# Calculate secondary structure percentages for each peptide
for i in range(len(combined_df)):
    c_percentage, h_percentage, e_percentage = calculate_secondary_structure_percentages(combined_df['Structure'][i])
    coil.append(c_percentage)
    helix.append(h_percentage)
    strand.append(e_percentage)

combined_df['Coil'] = coil
combined_df['Helix'] = helix
combined_df['Strand'] = strand

positions = [1, 2, 3]
gap = 0.2
widths = 0.15

plt.boxplot(combined_df[combined_df['Label'] == labels[0]][['Coil','Helix','Strand']], positions=[pos-0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=label_colors[labels[0]]), flierprops={'markersize': 2})
plt.boxplot(combined_df[combined_df['Label'] == labels[1]][['Coil','Helix','Strand']], positions=[pos+0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=label_colors[labels[1]]), flierprops={'markersize': 2})
    
# Set the x-axis labels
plt.xticks(positions, combined_df.columns[-3:])
plt.xlabel("Secondary Structure")
plt.ylabel('Percentage')
plt.grid(visible=False, axis='both')

legend_elements = [
    Patch(facecolor= label_colors[labels[0]], label=labels[0]),
    Patch(facecolor= label_colors[labels[1]], label=labels[1])
]

plt.legend(handles=legend_elements)
plt.savefig('MTS/data/diversity_ss_fraction_s4pred.png', dpi = 400, bbox_inches = "tight")