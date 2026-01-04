import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer

DATA_PATH = Path("data") / "NER dataset.csv"
MODEL_NAME = "bert-base-cased"  # bun pentru NER pe engleză

# 1) Citim CSV-ul și reconstruim propozițiile
df = pd.read_csv(DATA_PATH, encoding="latin1")
df["Sentence #"] = df["Sentence #"].ffill()

sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
labels = df.groupby("Sentence #")["Tag"].apply(list).tolist()

unique_labels = sorted({tag for sent_tags in labels for tag in sent_tags})
label2id = {lab: i for i, lab in enumerate(unique_labels)}
id2label = {i: lab for lab, i in label2id.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

i = 0
words = sentences[i]
word_labels = labels[i]

print(f"Propozitie [{i}] (primele 30 cuvinte):")
print(words[:30])
print(word_labels[:30])
print()

# 4) Tokenizam "pe cuvinte"
enc = tokenizer(
    words,
    is_split_into_words=True,
    truncation=True,
    return_offsets_mapping=True,
)

word_ids = enc.word_ids()
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

# 5) Aliniem label-urile la tokeni
aligned_label_ids = []
prev_word_id = None
for wid in word_ids:
    if wid is None:
        aligned_label_ids.append(-100)  # [CLS], [SEP], padding etc.
    elif wid != prev_word_id:
        aligned_label_ids.append(label2id[word_labels[wid]])
    else:
        aligned_label_ids.append(-100)
    prev_word_id = wid

print("Tokenizare + aliniere label-uri (primele 80 token-uri):\n")
print(f"{'tok_idx':>6}  {'token':<18}  {'word_idx':>7}  {'word':<18}  {'label':<10}")
print("-" * 70)

for t_idx, (tok, wid, lab_id) in enumerate(zip(tokens, word_ids, aligned_label_ids)):
    if t_idx >= 80:
        break
    word_str = "" if wid is None else words[wid]
    lab_str = "IGN(-100)" if lab_id == -100 else id2label[lab_id]
    print(f"{t_idx:>6}  {tok:<18}  {str(wid):>7}  {word_str:<18}  {lab_str:<10}")

# Mică verificare
print("\nVerificare rapida:")
print("Numar tokens:", len(tokens))
print("Numar aligned labels:", len(aligned_label_ids))
print("Conteaza la loss (labels != -100):", sum(l != -100 for l in aligned_label_ids))
