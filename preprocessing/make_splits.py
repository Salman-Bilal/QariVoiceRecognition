# preprocessing/make_splits.py
import pandas as pd
import json

def create_splits(manifest_df, held_out_val_surah, held_out_test_surah):
    """
    held_out_val_surah / held_out_test_surah: surah names (without extension)
    used consistently across ALL Qaris for val/test.
    Must exactly match the surah_name values in your manifest.
    """
    def assign_split(surah_name):
        if surah_name == held_out_val_surah:
            return "val"
        elif surah_name == held_out_test_surah:
            return "test"
        else:
            return "train"

    manifest_df = manifest_df.copy()
    manifest_df["split"] = manifest_df["surah_name"].apply(assign_split)
    return manifest_df

# Example — replace with your actual surah filenames (no extension)
HELD_OUT_VAL = "Surah_Kahf"
HELD_OUT_TEST = "Surah_Rahman"

manifest_df = pd.read_csv("/kaggle/working/chunks_manifest.csv")
manifest_df = create_splits(manifest_df, HELD_OUT_VAL, HELD_OUT_TEST)

manifest_df.to_csv("/kaggle/working/splits.csv", index=False)

print(manifest_df.groupby(["split", "qari_id"]).size().unstack(fill_value=0))