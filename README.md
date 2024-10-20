# MTS-VAE
### THE PAPER!
This repository accompanies the work ["Design of diverse, functional mitochondrial targeting sequences across eukaryotic organisms using Variational Autoencoders"](https://www.google.com). This work utilizes local software packages of [TargetP 2.0](https://services.healthtech.dtu.dk/services/TargetP-2.0/) and [DeepLoc 2.0](https://services.healthtech.dtu.dk/services/DeepLoc-2.0/) for synthetic data generation and analysis of artificial peptides, respectively. It also employs [s4pred](https://github.com/psipred/s4pred) and [modlamp](https://modlamp.org) for analyzing the characteristics of generated peptides, while [UniRep](https://github.com/churchlab/UniRep) embeddings are used for sampling. 

### Model:
![Model](Model.png)

Install the requirements using: 
```
conda env create -f environment.yml
conda activate py3
```

For model training, run:
```
python MTS/scripts/train.py
```

To generate artificial mitochondrial targeting sequences, run:
```
python MTS/scripts/generate.py
```

To sample artificial MTS near the cluster center:
```
python MTS/scripts/sample.py
```

To train the Dual-VAE model, run:
```
python Dual/scripts/train.py
```

To generate dual targeting sequences using latent interpolation, run:
```
python Dual/scripts/interpolate.py
```

Replicate the analysis/data processing performed in the paper using:
```
python MTS/analysis/distribution.py
python MTS/analysis/seq_identity.py
python MTS/analysis/deeploc2.py
python MTS/analysis/characteristics.py
python MTS/analysis/organism.py
python MTS/scripts/in_vivo.py


python Dual/analysis/preprocess.py
python Dual/analysis/split.py
python Dual/analysis/deeploc2.py
python Dual/analysis/characteristics.py
```

### Reference
<details>
<summary>If you using these scripts, please cite us:</summary>

```bibtex

```
</details>