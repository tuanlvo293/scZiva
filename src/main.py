import numpy as np
import pandas as pd
from pathlib import Path
import torch, gc
from ZIVA import ZIVAimpute
from preprocessing import normalize_and_log_single


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

    device = torch.device("cuda")
    print(f"[ZIVA] Using device: {device}")

    for csv_file in files:
        print(f"\n[ZIVA] Processing: {csv_file.name}")

        try:
            # --------------------------------------------------
            # 1️⃣ Load data (cells × genes)
            # --------------------------------------------------
            df = pd.read_csv(csv_file, index_col=0)

            cell_names = df.index
            gene_names = df.columns

            X = df.values.astype(np.float32)
            print("[ZIVA] Input shape (cells × genes):", X.shape)

            # --------------------------------------------------
            # 2️⃣ Optional rounding
            # --------------------------------------------------
            if round_input:
                X = np.round(X).clip(min=0)

            # --------------------------------------------------
            # 3️⃣ ZIVA imputation (GPU)
            # --------------------------------------------------
            zero_ratio = (X == 0).sum() / X.size
            print(f"[ZIVA] Zero ratio before impute: {zero_ratio:.2%}")

            X_imp, _ = ZIVAimpute(
                X,
                seed=seed,
                device=device
            )

            zero_ratio_imp = (X_imp == 0).sum() / X_imp.size
            print(f"[ZIVA] Zero ratio after impute: {zero_ratio_imp:.2%}")

            # --------------------------------------------------
            # 4️⃣ Normalize + log
            # --------------------------------------------------
            X_imp = normalize_and_log_single(X_imp)
            print("[ZIVA] Normalization & log done.")

            # --------------------------------------------------
            # 5️⃣ Save output
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
            del X, X_imp
            gc.collect()
            torch.cuda.empty_cache()


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    run_ziva_imputation(
        input_folder="data",
        output_folder="results/imputed_data",
        round_input=True
    )
