##Analyzing peptide composition, net charge, secondary structure properties 
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
def read_fasta(name):
    fasta_seqs = SeqIO.parse(open(name + '.fasta'),'fasta')
    data = []
    for fasta in fasta_seqs:
        data.append([fasta.id, str(fasta.seq).strip()])
            
    return data

def calculate_amino_acid_fraction(peptide):
    prot_param = ProteinAnalysis(peptide)
    amino_acid_fraction = prot_param.get_amino_acids_percent()
    return amino_acid_fraction

def get_color_by_charge(amino_acid):
    # Define color mapping based on charge
    if amino_acid in ['R', 'K', 'H']:
        return 'red'  # Positive charge
    elif amino_acid in ['D', 'E']:
        return 'blue'  # Negative charge
    else:
        return 'gray'  # Neutral charge

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
    out_file = open(name, "w")
    for i in range(len(sequence_df)):
        out_file.write('>' + sequence_df['Name'][i] + '\n')
        out_file.write(sequence_df['Sequence'][i] + '\n')
    out_file.close()

df = pd.DataFrame(read_fasta('MTS/data/model_organism_sequences_mts'), columns = ['Name', 'Sequence'])

org_label = []
index_label = [] 
for i in range(np.shape(df)[0]):
    if 'HUMAN' in df['Name'][i]:
        org_label.append(1)
        index_label.append(i)
    elif 'Rhoto' in df['Name'][i]:
        org_label.append(2)
        index_label.append(i)
    elif 'TOBAC' in df['Name'][i]:
        org_label.append(3)
        index_label.append(i)
    elif 'YEAST' in df['Name'][i]:
        org_label.append(4)
        index_label.append(i)
        
df = df.iloc[index_label].reset_index(drop=True)
df['Label'] = org_label

df_f = df[df.Sequence.str.startswith('M')].reset_index(drop=True)
df_f = clean(df_f)

print('Total number of sequences remaining:', len(df_f))
df_f = df_f.drop_duplicates(subset='Sequence').reset_index().drop('index', axis=1)
print('Total sequences remaining after duplicate removal', len(df_f))
#write_fasta('MTS/data/model_organism_sequences_mts.fas', df_f)

#Biopython
ss_fraction = []
net_charge = []
gravy = []
eisenberg_hydrophobicity = []
for i in range(len(df_f)):
    prot_seq = ProteinAnalysis(df_f['Sequence'][i])
    ss_fraction.append(prot_seq.secondary_structure_fraction()) # helix, turn, sheet
    net_charge.append(prot_seq.charge_at_pH(7.0))
    gravy.append(prot_seq.gravy())
    eisenberg_hydrophobicity.append(prot_seq.gravy(scale='Eisenberg'))

df_f['Helix'], df_f['Turn'], df_f['Sheet'] = zip(*ss_fraction)
df_f['Net Charge'] = net_charge
df_f['GRAVY'] = gravy
df_f['Eisenberg hydrophobicity'] = eisenberg_hydrophobicity
df_f = pd.concat([df_f, df_f['Sequence'].apply(calculate_amino_acid_fraction).apply(pd.Series)], axis=1)
labels = ['H. sapiens', 'R. toruloides', 'N. tabacum', 'S. cerevisiae']

#Hydrophobic Moment
descr = PeptideDescriptor(np.array(df_f['Sequence']),'eisenberg')
descr.calculate_moment()
organism_mu = list(descr.descriptor.flatten())

df_f['Moment'] = organism_mu
for i in range(len(labels)):
    # Select the data for the current label
    data = df_f[df_f['Label'] == i+1]['Moment']

    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, hist_kws={'alpha': 0.25})

plt.xlabel('Hydrophobic Moment')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('MTS/data/organism_moment_modlamp.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Net charge
for i in range(len(labels)):
    # Select the data for the current label
    data = df_f[df_f['Label'] == i+1]['Net Charge']

    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, hist_kws={'alpha': 0.25})

# Set the title and labels for the plot
plt.xlabel('Net Charge')
plt.ylabel('Count')
plt.legend()
plt.savefig('MTS/data/organism_net_charge_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Amino acid composition
positions = list(range(1, 21))
gap = 0.2
widths = 0.15
amino_acid_columns = df_f.columns[-20:]

plt.figure(figsize=(12, 6))
plt.boxplot(df_f[df_f['Label'] == 1][amino_acid_columns], positions=[pos-1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='lightblue'), flierprops={'markersize': 2})
plt.boxplot(df_f[df_f['Label'] == 2][amino_acid_columns], positions=[pos-0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='bisque'), flierprops={'markersize': 2})
plt.boxplot(df_f[df_f['Label'] == 3][amino_acid_columns], positions=[pos+0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='#A2DEA5'), flierprops={'markersize': 2})
plt.boxplot(df_f[df_f['Label'] == 4][amino_acid_columns], positions=[pos+1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='mistyrose'), flierprops={'markersize': 2})
    
# Set the title and labels for the plot
plt.xticks(range(1, len(amino_acid_columns) + 1), amino_acid_columns)
plt.xlabel('Amino Acids')
plt.ylabel('Fraction')
plt.grid(visible=False, axis='both')

legend_elements = [
    Patch(facecolor='lightblue', label=labels[0]),
    Patch(facecolor='bisque', label=labels[1]),
    Patch(facecolor='#A2DEA5', label=labels[2]),
    Patch(facecolor='mistyrose', label=labels[3])
]
plt.legend(handles=legend_elements)
plt.savefig('MTS/data/organism_aa_fraction_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

plt.figure(figsize=(6.4, 4.8))

#GRAVY
for i in range(len(labels)):
    # Select the data for the current label
    data = df_f[df_f['Label'] == i+1]['GRAVY']

    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, hist_kws={'alpha': 0.25})

# Set the title and labels for the plot
plt.xlabel('GRAVY')
plt.ylabel('Count')
plt.legend()
plt.savefig('MTS/data/organism_gravy_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Eisenberg hydrophobicity
for i in range(len(labels)):
    # Select the data for the current label
    data = df_f[df_f['Label'] == i+1]['Eisenberg hydrophobicity']

    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, hist_kws={'alpha': 0.25})

# Set the title and labels for the plot
plt.xlabel('Eisenberg hydrophobicity')
plt.ylabel('Count')
plt.legend()
plt.savefig('MTS/data/organism_eisenberg_hydrophobicity_biopython.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#s4pred secondary structure
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
with open("MTS/data/model_organism_sequences_mts_ss.fas", "r") as file:
    lines = file.readlines()

    for i in range(0, len(lines), 3):
        protein_name = lines[i].strip()[1:]
        protein_sequence = lines[i + 1].strip()
        protein_structure = lines[i + 2].strip()

        data["Name"].append(protein_name)
        data["Sequence"].append(protein_sequence)
        data["Structure"].append(protein_structure)

df = pd.DataFrame(data)
org_label = []
index_label = [] 
for i in range(np.shape(df)[0]):
    if 'HUMAN' in df['Name'][i]:
        org_label.append(1)
        index_label.append(i)
    elif 'RHOTO' in df['Name'][i]:
        org_label.append(2)
        index_label.append(i)
    elif 'TOBAC' in df['Name'][i]:
        org_label.append(3)
        index_label.append(i)
    elif 'YEAST' in df['Name'][i]:
        org_label.append(4)
        index_label.append(i)
        
df = df.iloc[index_label].reset_index(drop=True)
df['Label'] = org_label

coil = []
helix = []
strand = []
# Calculate secondary structure percentages for each peptide
for i in range(len(df)):
    c_percentage, h_percentage, e_percentage = calculate_secondary_structure_percentages(df['Structure'][i])
    coil.append(c_percentage)
    helix.append(h_percentage)
    strand.append(e_percentage)

df['Coil'] = coil
df['Helix'] = helix
df['Strand'] = strand

positions = [1, 2, 3]
gap = 0.2
widths = 0.15

plt.boxplot(df[df['Label'] == 1][['Coil','Helix','Strand']], positions=[pos-1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='lightblue'), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == 2][['Coil','Helix','Strand']], positions=[pos-0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='bisque'), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == 3][['Coil','Helix','Strand']], positions=[pos+0.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='#A2DEA5'), flierprops={'markersize': 2})
plt.boxplot(df[df['Label'] == 4][['Coil','Helix','Strand']], positions=[pos+1.5*gap for pos in positions], widths=widths, patch_artist=True, boxprops=dict(facecolor='mistyrose'), flierprops={'markersize': 2})
    
# Set the x-axis labels
plt.xticks(positions, df.columns[-3:])
plt.xlabel("Secondary Structure")
plt.ylabel('Fraction')
plt.grid(visible=False, axis='both')

legend_elements = [
    Patch(facecolor='lightblue', label=labels[0]),
    Patch(facecolor='bisque', label=labels[1]),
    Patch(facecolor='#A2DEA5', label=labels[2]),
    Patch(facecolor='mistyrose', label=labels[3])
]
plt.legend(handles=legend_elements)
plt.savefig('MTS/data/organism_ss_fraction_s4pred.png', dpi = 400, bbox_inches = "tight")
plt.clf()

#Length
df['Length'] = df['Sequence'].apply(len)
for i in range(len(labels)):
    # Select the data for the current label
    data = df[df['Label'] == i+1]['Length']

    # Plot the histogram
    sns.distplot(data, label= labels[i], bins = 10, hist=True, kde=True, rug=False, hist_kws={'alpha': 0.25})

plt.xlabel('Length')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('MTS/data/organism_length.png', dpi = 400, bbox_inches = "tight")