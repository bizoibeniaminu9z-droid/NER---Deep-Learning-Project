import pandas as pd
from pathlib import Path

DATA_PATH = Path("data") / "NER dataset.csv"

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Primele 5 randuri:")
print(df.head())

print("\nColoane:")
print(list(df.columns))

df["Sentence #"] = df["Sentence #"].ffill()

# Grupam pe propoziții
sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
labels = df.groupby("Sentence #")["Tag"].apply(list).tolist()

print("\nNumăr de propoziții:", len(sentences))

i = 0
print(f"\nExemplu propozitie [{i}] (primele 30 token-uri):")
print(sentences[i][:30])
print(labels[i][:30])

# Verificare
print("\nLungimi (tokens vs labels) pentru exemplul 0:", len(sentences[i]), len(labels[i]))
