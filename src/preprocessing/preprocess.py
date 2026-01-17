import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy import sparse as sp
#from utils import load_h5_data1, load_h5_data2


def load_real_data_forDEGs(file_path, min_cells=1, round_input=False):
    
    # --- Load file ---
    df = pd.read_csv(file_path, index_col=0)
    X_raw = df.values
    print(f"[INFO] Loaded data: {X_raw.shape[0]} × {X_raw.shape[1]} (before filtering)")

    # --- Detect orientation ---
    # We expect rows=cells, columns=genes. If not, transpose automatically.
    if X_raw.shape[0] < X_raw.shape[1]:
        print("[INFO] Detected genes as columns → assuming cells × genes format.")
        X = X_raw
    else:
        print("[INFO] Detected genes as rows → transposing to cells × genes.")
        X = X_raw.T
        df_temp = pd.DataFrame(X)
        df_temp.index=df.columns
        df_temp.columns=df.index
        df=df_temp

    # --- Filter out genes not expressed in any cell ---
    adata = sc.AnnData(X)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep = gene_counts >= min_cells
    X_filtered = X[:, genes_to_keep].astype(np.float32)
    df_filtered = df.loc[:, genes_to_keep]
    
    # --- Optional rounding ---
    if round_input:
        print("[INFO] Rounding input to nearest integer (for raw counts).")
        X_filtered = np.round(X_filtered).clip(min=0)

    print(f"[INFO] Filtered genes: {np.sum(genes_to_keep)} kept out of {len(genes_to_keep)}")

    print(df_filtered.columns)
    print(df_filtered.index)

    return X_filtered, df_filtered


def load_real_data(data_file, min_cells=1, round_input=False):
    """
    Load gene x cell CSV, convert to cells x genes, filter genes by presence >= min_cells,
    then (optionally) round the filtered matrix at the end.
    """

    # 1) Read CSV: genes × cells
    df = pd.read_csv(data_file, index_col=0, engine="python")

    X_gc = df.values  # genes × cells
    print(f"[INFO] Loaded data: {X_gc.shape[0]} genes × {X_gc.shape[1]} cells (before filtering)")

    # 2) Convert to cells × genes for ZIVA
    X_cg = X_gc.T  # cells × genes
    cell_names = df.columns
    gene_names = df.index
    df_cg = pd.DataFrame(X_cg, index=cell_names, columns=gene_names)

    # 3) Filter genes FIRST: keep genes expressed in >= min_cells cells (based on original values)
    gene_counts = (X_cg > 0).sum(axis=0)  # length = #genes
    genes_to_keep = gene_counts >= min_cells

    X_filtered = X_cg[:, genes_to_keep].astype(np.float32)
    df_filtered = df_cg.loc[:, genes_to_keep]

    print(f"[INFO] Filtered genes: {genes_to_keep.sum()} kept out of {len(genes_to_keep)}")

    # 4) round AFTER filtering
    if round_input:
        X_filtered = np.round(X_filtered).clip(min=0).astype(np.float32)
        df_filtered = pd.DataFrame(
            X_filtered,
            index=df_filtered.index,
            columns=df_filtered.columns
        )
    return X_filtered, df_filtered

def load_real_data_advanced(file_path, 
                   min_cells=1, 
                   min_cells_percent=0.1,
                   min_genes=500,
                   max_genes=None,
                   round_input=False, 
                   normalize=False,
                   log_transform=False,
                   verbose=True):
    """
    Load and preprocess single-cell RNA-seq data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file containing expression data
    min_cells : int, default=1
        Minimum number of cells a gene must be expressed in (absolute number)
    min_cells_percent : float, optional
        Min percentage of cells a gene must be expressed in (e.g., 0.1 for 10%)
        If specified, overrides min_cells
    min_genes : int, default=200
        Minimum genes a cell must express to be kept
    max_genes : int, optional
        Maximum genes a cell can express (filter potential doublets)
    round_input : bool, default=False
        Whether to round values to nearest integer (for raw counts)
    normalize : bool, default=False
        Normalize to 10,000 counts per cell
    log_transform : bool, default=False
        Apply log1p transformation
    verbose : bool, default=True
        Whether to print info messages
        
    Returns:
    --------
    X_filtered : np.ndarray
        Filtered expression matrix (cells × genes)
    df_filtered : pd.DataFrame
        Filtered DataFrame with proper index/columns (cell names × gene names)
    adata : sc.AnnData
        AnnData object for downstream analysis
    """
    
    # --- Load file ---
    df = pd.read_csv(file_path, index_col=0)
    X_raw = df.values
    
    if verbose:
        print(f"[INFO] Loaded data: {X_raw.shape[0]} × {X_raw.shape[1]} (before filtering)")

    # --- Detect orientation ---
    # We expect rows=cells, columns=genes. If not, transpose automatically.
    if X_raw.shape[0] < X_raw.shape[1]:
        if verbose:
            print("[INFO] Detected genes as columns → assuming cells × genes format.")
        X = X_raw
    else:
        if verbose:
            print("[INFO] Detected genes as rows → transposing to cells × genes.")
        X = X_raw.T
        # Cleaner transpose using .T attribute
        df = df.T

    # --- Optional rounding ---
    if round_input:
        if verbose:
            print("[INFO] Rounding input to nearest integer (for raw counts).")
        X = np.round(X).clip(min=0)

    # --- Create initial AnnData object ---
    adata = sc.AnnData(X)
    adata.obs_names = df.index
    adata.var_names = df.columns
    
    # --- Calculate initial QC metrics ---
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1)
    adata.obs['total_counts'] = np.sum(adata.X, axis=1)
    adata.var['n_cells'] = np.sum(adata.X > 0, axis=0)
    
    if verbose:
        print(f"\n[INFO] === Quality Control Filtering ===")
        print(f"[INFO] Initial: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # --- Step 1: Filter cells by gene count ---
    cells_before = adata.n_obs
    if verbose:
        print(f"[INFO] Filtering cells with < {min_genes} genes...")
    
    adata = adata[adata.obs['n_genes'] >= min_genes, :].copy()
    
    if verbose:
        print(f"[INFO] Cells after min_genes filter: {adata.n_obs} (removed {cells_before - adata.n_obs})")
    
    if max_genes is not None:
        cells_before = adata.n_obs
        if verbose:
            print(f"[INFO] Filtering cells with > {max_genes} genes...")
        adata = adata[adata.obs['n_genes'] <= max_genes, :].copy()
        if verbose:
            print(f"[INFO] Cells after max_genes filter: {adata.n_obs} (removed {cells_before - adata.n_obs})")
    
    # --- Step 2: Filter genes by cell count ---
    genes_before = adata.n_vars
    
    if min_cells_percent is not None:
        # Use percentage-based filtering
        min_cells_threshold = int(np.ceil(adata.n_obs * min_cells_percent))
        if verbose:
            print(f"[INFO] Filtering genes expressed in < {min_cells_percent*100:.1f}% of cells (< {min_cells_threshold} cells)...")
    else:
        # Use absolute number
        min_cells_threshold = min_cells
        if verbose:
            print(f"[INFO] Filtering genes expressed in < {min_cells_threshold} cells...")
    
    # Recalculate n_cells after cell filtering
    adata.var['n_cells'] = np.sum(adata.X > 0, axis=0)
    adata = adata[:, adata.var['n_cells'] >= min_cells_threshold].copy()
    
    if verbose:
        print(f"[INFO] Genes after filtering: {adata.n_vars} (removed {genes_before - adata.n_vars})")
    
    # --- Recalculate metrics after filtering ---
    adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1)
    adata.obs['total_counts'] = np.sum(adata.X, axis=1)
    adata.var['n_cells'] = np.sum(adata.X > 0, axis=0)
    
    # --- Normalization ---
    if normalize:
        if verbose:
            print("[INFO] Normalizing to 10,000 counts per cell")
        sc.pp.normalize_total(adata, target_sum=1e4)
    
    # --- Log transformation ---
    if log_transform:
        if verbose:
            print("[INFO] Applying log1p transformation")
        sc.pp.log1p(adata)
    
    # --- Prepare outputs ---
    X_filtered = adata.X.copy()
    df_filtered = pd.DataFrame(
        X_filtered,
        index=adata.obs_names,
        columns=adata.var_names
    )
    
    if verbose:
        print(f"\n[INFO] === Final Data ===")
        print(f"[INFO] Shape: {adata.n_obs} cells × {adata.n_vars} genes")
        print(f"[INFO] Mean genes per cell: {adata.obs['n_genes'].mean():.1f}")
        print(f"[INFO] Mean counts per cell: {adata.obs['total_counts'].mean():.1f}")
        print(f"[INFO] Mean cells per gene: {adata.var['n_cells'].mean():.1f}")
        print(f"\n[INFO] First 5 genes: {list(df_filtered.columns[:5])}")
        print(f"[INFO] First 5 cells: {list(df_filtered.index[:5])}")
    
    return X_filtered, df_filtered, adata


def load_and_filter_data(dropout_rate):
    """Load and filter data for a specific dropout rate"""
    # Load the counts data
    counts_file = f"sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_counts.csv"
    df3 = pd.read_csv(counts_file)
    Xmiss_original = df3.iloc[:, 1:].values.T

    # Load true counts
    truecounts_file = f"sim.Tung/sim.Tung.drop{dropout_rate}/SplatDrop_TrueCounts.csv"
    df1 = pd.read_csv(truecounts_file)
    X_original = df1.iloc[:, 1:].values.T
    
    # Apply gene filtering
    adata = sc.AnnData(Xmiss_original)
    gene_counts = np.sum(adata.X > 0, axis=0)
    genes_to_keep_mask = gene_counts >= 1
    
    # Filter all matrices consistently
    Xmiss = Xmiss_original[:, genes_to_keep_mask].astype('float32')
    X = X_original[:, genes_to_keep_mask]
    
    Xmiss = np.rint(Xmiss).clip(min=0).astype('float32')
    return X, Xmiss

def normalize_and_log_single(
        X,
        do_normalize=True,
        do_log=True,
        target_sum=1e4,
        highly_genes=None,   # <-- thêm tham số
        copy=True
    ):
    """
    Chuẩn hoá + log + OPTIONAL: chọn top N HVGs (sau khi normalize/log).
    Trả về duy nhất 1 ma trận đã qua mọi bước xử lý.
    """

    # (1) Normalize + log như cũ
    adata = ad.AnnData(X.copy() if copy else X)

    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=False)

    if do_log:
        sc.pp.log1p(adata)

    X_processed = adata.X

    # Ép kiểu
    if sp.issparse(X_processed):
        X_processed = X_processed.astype(np.float32)
    else:
        X_processed = np.asarray(X_processed, dtype=np.float32)

    # (2) Nếu không chọn HVG → return
    if highly_genes is None:
        return X_processed

    # (3) TÍNH HVG SAU normalize/log
    adata2 = ad.AnnData(X_processed)

    sc.pp.highly_variable_genes(
        adata2,
        n_top_genes=highly_genes,
        subset=True,           # <-- LẤY TOP N GENE
        min_mean=0.0125,          # vì tính trên normalized data
        max_mean=3,
        min_disp=0.5           # để không loại gene do dispersion quá thấp
    )

    # (4) Kết quả cuối: X_processed đã được subset
    X_subset = adata2.X

    # Ép kiểu lần cuối
    if sp.issparse(X_subset):
        X_subset = X_subset.astype(np.float32)
    else:
        X_subset = np.asarray(X_subset, dtype=np.float32)

    return X_subset


def normalize_and_log(
        X, X_imp,
        do_normalize=True,
        do_log=True,
        target_sum=1e4,
        highly_genes=None,
        copy=True
    ):
    """
    Normalize + log X, rồi tính HVG trên chính X_norm,
    và subset cả X_norm và X_imp_scaled theo HVG mask đầy đủ.
    """

    # --- (1) Total counts raw ---
    if sp.issparse(X):
        total_counts_X = np.asarray(X.sum(axis=1)).ravel()
    else:
        total_counts_X = X.sum(axis=1).ravel()

    safe_total = total_counts_X.copy()
    safe_total[safe_total == 0] = 1.0

    # --- (2) Normalize + log X ---
    adata = ad.AnnData(X.copy() if copy else X)

    if do_normalize:
        sc.pp.normalize_total(adata, target_sum=target_sum, exclude_highly_expressed=False)
    if do_log:
        sc.pp.log1p(adata)

    X_norm = adata.X
    X_norm = np.asarray(X_norm, dtype=np.float32)

    # --- (3) Apply SAME normalization/log to X_imp ---
    if do_normalize:
        factors = safe_total / float(target_sum)
        if sp.issparse(X_imp):
            X_imp_scaled = X_imp.multiply(1.0 / factors[:, None])
        else:
            X_imp_scaled = X_imp / factors[:, None]
    else:
        X_imp_scaled = X_imp.copy()

    if do_log:
        if sp.issparse(X_imp_scaled):
            coo = X_imp_scaled.tocoo()
            coo.data = np.log1p(coo.data)
            X_imp_scaled = coo.tocsr()
        else:
            X_imp_scaled = np.log1p(X_imp_scaled)

    X_imp_scaled = np.asarray(X_imp_scaled, dtype=np.float32)

    # --- (4) Không chọn HVG thì trả luôn ---
    if highly_genes is None:
        return X_norm, X_imp_scaled

    # --- (5) TÍNH HVG trên full gene set (subset=False) ---
    adata_hvg = ad.AnnData(X_norm)
    sc.pp.highly_variable_genes(
        adata_hvg,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        n_top_genes=highly_genes,
        subset=False     # <-- QUAN TRỌNG: GIỮ NGUYÊN 21073 genes
    )

    # Full-length boolean mask (length = số gene gốc)
    mask_full = adata_hvg.var['highly_variable'].values

    # --- (6) Subset X_norm & X_imp_scaled ---
    X_norm_hvg = X_norm[:, mask_full]
    X_imp_hvg = X_imp_scaled[:, mask_full]

    return X_norm_hvg.astype(np.float32), X_imp_hvg.astype(np.float32)

