# This code picks variables from our data to feed them into the model later

import pandas as pd
from typing import List
import os
import glob
from pathlib import Path

#Define your own paths if your data structure differs from github's
Data_path = Path("../Données")
Variables_path = Path("../Output")

#Class to standardize models, we can select variables, create the dataset, and choose parameters. 
class Model:
    def __init__(self) -> None:
        pass

    # Prendre tous les csv files dans Data_path qui ont une colonne dans l'objet variables & les mettre dans un dataframe
    def Dataset_maker(self, Data_path) -> pd.DataFrame:
        print("▶ Loading base communes file")
        df = pd.read_csv(
            Data_path / "Législatives" / "leg2002_csv__leg2002comm.csv",
            usecols=["codecommune", "nomcommune"],
            dtype={"codecommune": str},
            low_memory=False
        )

        self.variables = input(
            "▶ Enter variables (comma-separated): "
        ).split(",")
        self.variables = [v.strip() for v in self.variables if v.strip()]
        print(f"▶ Variables selected: {self.variables}")

        files = list(Data_path.rglob("*.csv"))
        print(f"▶ {len(files)} CSV files found")

        for file in files:
            if file.name.startswith("._"):
                continue

            print(f"\n▶ Reading: {file.name}")
            try:
                data = pd.read_csv(file, dtype={"codecommune": str}, encoding="utf-8", low_memory=False)
            except UnicodeDecodeError:
                print("  ↪ UTF-8 failed, trying latin1")
                data = pd.read_csv(file, dtype={"codecommune": str}, encoding="latin1", low_memory=False)

            if "codecommune" not in data.columns:
                print("  ↪ Skipped (no codecommune)")
                continue

            cols = [v for v in self.variables if v in data.columns]
            if not cols:
                print("  ↪ Skipped (no selected variables)")
                continue

            print(f"  ↪ Merging columns: {cols}")
            df = df.merge(data[["codecommune"] + cols], on="codecommune", how="outer")
            df = df[df["codecommune"] != "."]
            df = df[df["codecommune"].notna()]

        print(f"\n▶ Final dataset shape: {df.shape}")
        return df


if __name__ == "__main__":
    First_model = Model()
    df = First_model.Dataset_maker(Data_path)
