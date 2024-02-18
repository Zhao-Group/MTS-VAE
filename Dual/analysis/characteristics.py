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
from modlamp.descriptors import PeptideDescriptor

#Function
def calculate_amino_acid_fraction(peptide):
    prot_param = ProteinAnalysis(peptide)
    amino_acid_fraction = prot_param.get_amino_acids_percent()
    return amino_acid_fraction

def write_fasta(name, sequence_df):
    out_file = open(name, "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + str(sequence_df['Name'][i]) + '\n')
        out_file.write(sequence_df['Sequence'][i][:-237] + '\n')
    out_file.close()

#Data (AMTS)
df = pd.read_csv('Dual/data/dual_tps.csv')
write_fasta('Dual/data/dtps.fas', df)

#Add labels
label = []
sequence = []
for i in range(int(len(df)/5)):
    for j in range(5):
        sequence.append(df['Sequence'][5*i+j][:-237])
    label.append('M')
    label.append('I1')
    label.append('I2')
    label.append('I3')
    label.append('C')

df['Label'] = label
df['Sequence'] = sequence

#Biopython
ss_fraction = []
net_charge = []
gravy = []
eisenberg_hydrophobicity = []
for i in range(len(df)):
    prot_seq = ProteinAnalysis(df['Sequence'][i])
    ss_fraction.append(prot_seq.secondary_structure_fraction()) # helix, turn, sheet
    net_charge.append(prot_seq.charge_at_pH(7.0))
    gravy.append(prot_seq.gravy())
    eisenberg_hydrophobicity.append(prot_seq.gravy(scale='Eisenberg'))

df['Helix'], df['Turn'], df['Sheet'] = zip(*ss_fraction)
df['Net Charge'] = net_charge
df['GRAVY'] = gravy
df['Eisenberg hydrophobicity'] = eisenberg_hydrophobicity
df = pd.concat([df, df['Sequence'].apply(calculate_amino_acid_fraction).apply(pd.Series)], axis=1)

#Hydrophobic Moment
descr = PeptideDescriptor(np.array(df['Sequence']),'eisenberg')
descr.calculate_moment()
dts_mu = list(descr.descriptor.flatten())

df['Moment'] = dts_mu
color_dict = {'M': '#d4664f', 'I1': '#ecbd81', 'I2': '#f6f7c6', 'I3': '#b4d382', 'C': '#58a365'} 
for l in df['Label'].unique():
    class_data = df[df['Label'] == l]
    sns.distplot(list(class_data['Moment']), bins = 6, hist=True, kde=True, rug=False, label = l, color = color_dict[l], norm_hist=False)

plt.xlabel('Hydrophobic Moment')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('Dual/data/moment_modlamp.png', dpi = 400, bbox_inches = "tight")
plt.clf()

df.to_csv('Dual/data/dtps_properties_biopython.csv', index = False)

color_dict = {'M': '#d4664f', 'I1': '#ecbd81', 'I2': '#f6f7c6', 'I3': '#b4d382', 'C': '#58a365'} 
for l in df['Label'].unique():
    class_data = df[df['Label'] == l]
    sns.distplot(list(class_data['Net Charge']), bins = 6, hist=True, kde=True, rug=False, label = l, color = color_dict[l])

plt.xlabel('Net Charge')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('Dual/data/dtp_net_charge_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()
    
for l in df['Label'].unique():
    class_data = df[df['Label'] == l]
    sns.distplot(list(class_data['GRAVY']), bins = 6, hist=True, kde=True, rug=False, label = l, color = color_dict[l])

plt.xlabel('GRAVY')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('Dual/data/dtp_GRAVY_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

for l in df['Label'].unique():
    class_data = df[df['Label'] == l]
    sns.distplot(list(class_data['Eisenberg hydrophobicity']), bins = 6, hist=True, kde=True, rug=False, label = l, color = color_dict[l])

plt.xlabel('Eisenberg hydrophobicity')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('Dual/data/dtp_Eisenberg_hydrophobicity_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Amino acid composition
positions = list(range(1, 21))
gap = 0.12
widths = 0.09
amino_acid_columns = df.columns[-20:]

plt.figure(figsize=(12, 4.8))
plt.boxplot(df[df['Label'] == 'M'][amino_acid_columns], positions=[pos-2.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['M']), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == 'I1'][amino_acid_columns], positions=[pos-1.25*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['I1']), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == 'I2'][amino_acid_columns], positions=[pos for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['I2']), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == 'I3'][amino_acid_columns], positions=[pos+1.25*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['I3']), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == 'C'][amino_acid_columns], positions=[pos+2.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['C']), flierprops={'markersize': 2})
    
# Set the title and labels for the plot
plt.xticks(range(1, len(amino_acid_columns) + 1), amino_acid_columns)
plt.xlabel('Amino Acids')
plt.ylabel('Fraction')
plt.grid(visible=False, axis='both')

legend_elements = [
    Patch(facecolor=color_dict['M'], label='M'),
    Patch(facecolor=color_dict['I1'], label='I1'),
    Patch(facecolor=color_dict['I2'], label='I2'),
    Patch(facecolor=color_dict['I3'], label='I3'),
    Patch(facecolor=color_dict['C'], label='I4')
]
plt.legend(handles=legend_elements)
plt.savefig('Dual/data/dtp_organism_aa_fraction_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Structure
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
with open("Dual/data/dtps_ss.fas", "r") as file:
    lines = file.readlines()

    for i in range(0, len(lines), 3):
        protein_name = lines[i].strip()[1:]
        protein_sequence = lines[i + 1].strip()
        protein_structure = lines[i + 2].strip()

        data["Name"].append(protein_name)
        data["Sequence"].append(protein_sequence)
        data["Structure"].append(protein_structure)

structure_df = pd.DataFrame(data)
structure_df['Label'] = label

coil = []
helix = []
strand = []
# Calculate secondary structure percentages for each peptide
for i in range(len(structure_df)):
    c_percentage, h_percentage, e_percentage = calculate_secondary_structure_percentages(structure_df['Structure'][i])
    coil.append(c_percentage)
    helix.append(h_percentage)
    strand.append(e_percentage)

structure_df['Coil'] = coil
structure_df['Helix'] = helix
structure_df['Strand'] = strand

#structure_df.to_csv('Dual/data/dtps_properties_s4pred.csv', index = False)

positions = [1, 2, 3]
gap = 0.12
widths = 0.09

plt.figure(figsize=(6.4, 4.8))
plt.boxplot(structure_df[structure_df['Label'] == 'M'][['Coil','Helix','Strand']], positions=[pos-2.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['M']), flierprops={'markersize': 2})
plt.boxplot(structure_df[structure_df['Label'] == 'I1'][['Coil','Helix','Strand']], positions=[pos-1.25*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['I1']), flierprops={'markersize': 2})
plt.boxplot(structure_df[structure_df['Label'] == 'I2'][['Coil','Helix','Strand']], positions=[pos for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['I2']), flierprops={'markersize': 2})
plt.boxplot(structure_df[structure_df['Label'] == 'I3'][['Coil','Helix','Strand']], positions=[pos+1.25*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['I3']), flierprops={'markersize': 2})
plt.boxplot(structure_df[structure_df['Label'] == 'C'][['Coil','Helix','Strand']], positions=[pos+2.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor=color_dict['C']), flierprops={'markersize': 2})

plt.xticks(positions, structure_df.columns[-3:])
plt.xlabel("Secondary Structure")
plt.ylabel('Percentage')
plt.grid(visible=False, axis='both')
legend_elements = [
    Patch(facecolor=color_dict['M'], label='M'),
    Patch(facecolor=color_dict['I1'], label='I1'),
    Patch(facecolor=color_dict['I2'], label='I2'),
    Patch(facecolor=color_dict['I3'], label='I3'),
    Patch(facecolor=color_dict['C'], label='I4')
]
plt.legend(handles=legend_elements)
plt.savefig('Dual/data/dtp_ss_fraction_s4pred.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Length
df['Length'] = df['Sequence'].apply(len)
for l in df['Label'].unique():
    class_data = df[df['Label'] == l]
    sns.distplot(list(class_data['Length']), bins = 6, hist=True, kde=True, rug=False, label = l, color = color_dict[l])

plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('Dual/data/dtp_Length.png', dpi = 400, bbox_inches = "tight")