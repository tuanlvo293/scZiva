# scZiva: Imputation method for single-cell RNA-seq data with Zero-Inflated Variational Autoencoder
This is a code repository associated with the paper: "scZiva: Imputation method for single-cell RNA-seq data with Zero-Inflated Variational Autoencoder". This paper is under review for the journal.
# 1. Introduction
Single-cell RNA sequencing (scRNA-seq) has enabled detailed gene expression analysis but is severely affected by technical dropouts, which introduce noise and hinder downstream analyses. We propose scZiva, a variational autoencoder–based imputation method that models overdispersed counts with a zero-inflated negative binomial distribution and captures local gene dependencies using one-dimensional convolutional layers. Experiments on simulated and real scRNA-seq datasets show that scZiva consistently outperforms existing imputation methods.

# 2. About this repository
This repository is structured as follows
```bash
.
├── README.md
├── requirements.txt
├── data
└── src
    ├── main.py
    ├── preprocessing
    ├── metrics
    └── ZIVAimputation
```

# 3. Usages

## 3.1. Datasets 
All datasets used in this work can be directly downloaded here: https://zenodo.org/uploads/18263992.

## 3.2. Example
- The input is located in `data` folder, and the output is located in `results/imputed_data`.
- Run the main.py, it will find all datasets and process them.

Here is an example with a specific dataset:

```bash
from ZIVA import ZIVAimpute

# Loading your dataset
df = pd.read_csv(data/Lung.csv, index_col=0)
X = df.values

# scZiva imputation
X_imp, _ = ZIVAimpute(X)
```

There are some options in `ZIVAimpute` function:

- `Xmiss`: Input data matrix with missing values (cells × genes)

- `seed`: Random seed for reproducibility (default: `1`)

- `device`: Computing device (`cuda` or `cpu`).  If `None`, the default device is automatically selected (default: `None`)

- `num_epochs`: Number of training epochs for optimizing the ZIVA model (default: `200`)

- `lr`: Learning rate for optimization (default: `1e-3`)

- `hidden_dim`: Dimension of hidden layers in the encoder/decoder network (default: `128`)

- `latent_dim`: Dimension of latent representation (default: `64`)

- `verbose`: Whether to print training logs (default: `False`)

- `tau`: Threshold for dropout probability `π` used during imputation (default: `0.001`)

## References and contact
We recommend you cite our paper when using these codes for further investigation:
```bash
@article{vo2026scziva,
  title={scZiva: Imputation method for single-cell RNA-seq data with Zero-Inflated Variational Autoencoder},
  author={Vo, Tuan L and Le Van, Vinh and Ha Quoc, Toan and Nguyen Anh, Quoc},
  journal={},
  year={2026}
}
```
You can just send additional requests directly to the first author Tuan L. Vo (tuanvl@hcmute.edu.vn) or the corresponding author Vinh Le Van (vinhlv@hcmute.edu.vn), for the appropriate permission.
