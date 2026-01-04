import pandas as pd
from pathlib import Path

DATA_PATH = Path("data") / "NER dataset.csv"

# Uneori CSV-ul ăsta are caractere ciudate, de asta folosim latin1
df = pd.read_csv(DATA_PATH, encoding="latin1")

print("Primele 5 rânduri:")
print(df.head())

print("\nColoane:")
print(list(df.columns))

# În multe versiuni, "Sentence #" are NaN pe rândurile următoare din aceeași propoziție.
# ffill() copiază ultimul Sentence # valid în jos.
df["Sentence #"] = df["Sentence #"].ffill()

# Grupăm pe propoziții
sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
labels = df.groupby("Sentence #")["Tag"].apply(list).tolist()

print("\nNumăr de propoziții:", len(sentences))

# Afișăm 1 propoziție (primele 30 cuvinte ca să nu fie perete de text)
i = 0
print(f"\nExemplu propoziție [{i}] (primele 30 token-uri):")
print(sentences[i][:30])
print(labels[i][:30])

# Verificare: trebuie să aibă aceeași lungime
print("\nLungimi (tokens vs labels) pentru exemplul 0:", len(sentences[i]), len(labels[i]))
