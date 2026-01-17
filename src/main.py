import numpy as np
import pandas as pd
from pathlib import Path
import torch, gc

from ZIVA import ZIVAimpute
from preprocessing import normalize_and_log_single


def print_gpu_mem(tag=""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        #print(f"[GPU] {tag} allocated={alloc:.2f}GB | reserved={reserved:.2f}GB")

def run_ziva_imputation(
    input_folder="data",
    output_folder="results/imputed_data",
    seed=1,
    round_input=True,
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(input_folder.glob("*.csv"))
    if not files:
        print(f"[WARN] No CSV files found in {input_folder.resolve()}")
        return

    print(f"[ZIVA] Found {len(files)} file(s) in {input_folder}")

    for csv_file in files:
        print(f"\n[ZIVA] Processing: {csv_file.name}")

        try:
            # --------------------------------------------------
            # 1️⃣ Load filtered data (cells × genes)
            # --------------------------------------------------
            df = pd.read_csv(csv_file, index_col=0)

            cell_names = df.index.copy()
            gene_names = df.columns.copy()

            X = df.values.astype(np.float32)
            print("[ZIVA] Input shape (cells × genes):", X.shape)

            # --------------------------------------------------
            # 2️⃣ Optional rounding
            # --------------------------------------------------
            if round_input:
                X = np.round(X).clip(min=0)

            # --------------------------------------------------
            # 3️⃣ Decide device (AUTO)
            # --------------------------------------------------
            n_elements = X.size
            use_gpu = torch.cuda.is_available() and n_elements < 8e7
            device = torch.device("cuda" if use_gpu else "cpu")

            print(f"[ZIVA] Using device: {device}")
            print_gpu_mem("before ZIVA")

            # --------------------------------------------------
            # 4️⃣ ZIVA imputation
            # --------------------------------------------------
            zero_ratio = (X == 0).sum() / X.size
            print(f"[ZIVA] Zero ratio before impute: {zero_ratio:.2%}")

            X_imp, _ = ZIVAimpute(
                X,
                seed=seed,
                device=device
            )

            print_gpu_mem("after ZIVA")

            zero_ratio_imp = (X_imp == 0).sum() / X_imp.size
            print(f"[ZIVA] Zero ratio after impute: {zero_ratio_imp:.2%}")

            # --------------------------------------------------
            # 5️⃣ Normalize + log (cells × genes)
            # --------------------------------------------------
            X_imp = normalize_and_log_single(X_imp)
            print("[ZIVA] Normalization & log done.")

            # --------------------------------------------------
            # 6️⃣ Save output
            # --------------------------------------------------
            out_df = pd.DataFrame(
                X_imp,
                index=cell_names,
                columns=gene_names
            )

            out_path = output_folder / f"{csv_file.stem}_imputed.csv"
            out_df.to_csv(out_path)

            print(f"[ZIVA] ✅ Saved: {out_path}")

        except Exception as e:
            print(f"[ERROR] ❌ Failed on {csv_file.name}: {e}")

        finally:
            if 'X' in locals():
                del X
            if 'X_imp' in locals():
                del X_imp

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print_gpu_mem("after cleanup")

# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    run_ziva_imputation(
        input_folder="data",
        output_folder="results/imputed_data",
        round_input=True
    )

