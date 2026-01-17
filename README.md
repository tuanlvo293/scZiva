# scZiva: Imputation method for single-cell RNA-seq data with Zero-Inflated Variational Autoencoder
This is a code repository associated with the paper: "scZiva: Imputation method for single-cell RNA-seq data with Zero-Inflated Variational Autoencoder". This paper is under review for the journal.
# 1. Introduction
**Background:** Single-cell RNA sequencing (scRNA-seq) is considered a revolution in gene expression studies and offers significant benefits across various fields of biomedical or clinical applications. However, a high proportion of technical dropouts in scRNA-seq data leads to increased noise and reduces the performance of downstream analyses such as cell clustering, differential expression analysis, and cell trajectory inference. Numerous recent imputation methods utilize deep learning to recover missing gene expression values in scRNA-seq data. Despite the research efforts, existing methods have limitations in capturing local co-expression patterns in scRNA-seq data and handling the uncertainty in distinguishing technical zeros from true biological zeros.

**Results:** This work proposes a novel imputation method based on Variational Autoencoder (VAE), called scZiva, for scRNA-seq data. The proposed method employs a Zero-Inflated Negative Binomial distribution to account for overdispersion in data, and integrates one-dimensional convolutional layers into the encoder to learn local gene dependency patterns. Additionally, an auxiliary reconstruction term and a probability-guided imputation strategy are also utilized to enhance imputation accuracy. Comprehensive experiments conducted on both simulated and real datasets demonstrate the strength of scZiva compared with other baseline methods.

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
All datasets used in this work can be directly downloaded here: https://zenodo.org/uploads/18263992

## 3.2. Example
