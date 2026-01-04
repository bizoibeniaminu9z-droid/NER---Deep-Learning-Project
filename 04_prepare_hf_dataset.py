import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# -----------------------
# Config
# -----------------------
DATA_PATH = Path("data") / "NER dataset.csv"
MODEL_NAME = "bert-base-cased"
MAX_LEN = 128
TEST_SIZE = 0.1
RANDOM_STATE = 42


def main():
    # -----------------------
    # 1) Load CSV (robust)
    # -----------------------
    df = pd.read_csv(DATA_PATH, encoding="latin1")

    # Curățăm rânduri stricate: Word/Tag lipsă
    df = df.dropna(subset=["Word", "Tag"])

    # Forțăm tip string. Ex: dacă un “Word” e 123, să nu îl trateze ca int.
    df["Word"] = df["Word"].astype(str)
    df["Tag"] = df["Tag"].astype(str)

    # Uneori Sentence # are NaN pe rândurile care aparțin aceleiași propoziții
    df["Sentence #"] = df["Sentence #"].ffill()

    # (opțional) eliminăm whitespace urât
    df["Word"] = df["Word"].str.strip()
    df["Tag"] = df["Tag"].str.strip()

    # -----------------------
    # 2) Group into sentences
    # -----------------------
    sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
    labels_str = df.groupby("Sentence #")["Tag"].apply(list).tolist()

    print("Număr propoziții:", len(sentences))
    print("Exemplu tokens:", sentences[0][:10])
    print("Exemplu labels:", labels_str[0][:10])

    # Verificare: aceeași lungime tokens vs labels
    bad = [(i, len(s), len(l)) for i, (s, l) in enumerate(zip(sentences, labels_str)) if len(s) != len(l)]
    if bad:
        print("\nATENȚIE: există propoziții cu lungimi diferite tokens vs labels! Primele 5:")
        for x in bad[:5]:
            print("idx:", x[0], "len(tokens):", x[1], "len(labels):", x[2])
        raise ValueError("Dataset inconsistent: tokens și labels nu au aceeași lungime pentru unele propoziții.")

    # -----------------------
    # 3) Build label mapping
    # -----------------------
    unique_labels = sorted({tag for sent_tags in labels_str for tag in sent_tags})
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    print("\nNumăr label-uri:", len(unique_labels))
    print("Primele label-uri:", unique_labels[:10])

    # Convertim label-urile din string -> int
    labels = [[label2id[t] for t in sent_tags] for sent_tags in labels_str]

    # -----------------------
    # 4) Train/Validation split
    # -----------------------
    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Construirea DatasetDict
    ds = DatasetDict({
        "train": Dataset.from_dict({"tokens": train_sents, "ner_tags": train_labels}),
        "validation": Dataset.from_dict({"tokens": val_sents, "ner_tags": val_labels}),
    })

    # -----------------------
    # 5) Tokenize + align
    # -----------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_and_align(batch):
        enc = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=MAX_LEN,
        )

        all_labels = []
        for i, word_label_ids in enumerate(batch["ner_tags"]):
            word_ids = enc.word_ids(batch_index=i)

            aligned = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    aligned.append(-100)  # [CLS], [SEP], etc.
                elif wid != prev:
                    aligned.append(word_label_ids[wid])  # primul sub-token primește label-ul cuvântului
                else:
                    aligned.append(-100)  # restul sub-tokenilor ignorați
                prev = wid

            all_labels.append(aligned)

        enc["labels"] = all_labels
        return enc

    ds_tok = ds.map(tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"])

    print("\nDupă tokenizare:")
    print(ds_tok)
    print("\nColoane:", ds_tok["train"].column_names)

    # -----------------------
    # 6) Save mappings
    # -----------------------
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    (out_dir / "label2id.txt").write_text(
        "\n".join([f"{k}\t{v}" for k, v in label2id.items()]),
        encoding="utf-8"
    )
    (out_dir / "id2label.txt").write_text(
        "\n".join([f"{k}\t{v}" for k, v in id2label.items()]),
        encoding="utf-8"
    )

    print("\nSalvat mapping în outputs/label2id.txt și outputs/id2label.txt")


if __name__ == "__main__":
    main()
