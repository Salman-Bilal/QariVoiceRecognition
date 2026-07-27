import pandas as pd

def create_splits(manifest_df, held_out_val_surah, held_out_test_surah):
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