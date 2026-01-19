from pathlib import Path
from Scripts.model import Model 

def main():
    data_path = Path("../Données")
    output_path = Path("../Output") 

    model = Model()

    df = model.Dataset_maker(data_path)

    print("Final dataframe shape:", df.shape)
    print(df.head())

    # optional: save result
    output_path.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(model.variables)
    filename = f"dataset_{suffix}.csv"
    df.to_csv(output_path / filename, index=False)
    print(f"Dataset saved to {output_path / filename}")

if __name__ == "__main__":
    main()
