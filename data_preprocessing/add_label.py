import pandas as pd


def add_label(subset: str) -> None:
    df = pd.read_csv(f"asl_citizen/splits/{subset}.csv")
    print("original:")
    print(df)

    glosses = sorted(list(set(df["Gloss"])))
    print(f"Total different signs: {len(glosses)}")
    gloss_to_label = {gloss: index for index, gloss in enumerate(glosses)}

    # Add a new column "Label" based on the gloss index
    df = df.sort_values(by="Gloss")
    df["Label"] = df["Gloss"].map(gloss_to_label)

    # Print the updated DataFrame
    print("with labels:")
    print(df)
    df.to_csv(f"asl_citizen/splits/{subset}.csv", index=False)


add_label("test")
