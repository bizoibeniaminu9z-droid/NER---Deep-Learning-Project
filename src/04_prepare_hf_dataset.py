import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

DATA_PATH = Path("data") / "NER dataset.csv"
MODEL_NAME = "bert-base-cased"
MAX_LEN = 128
TEST_SIZE = 0.1
RANDOM_STATE = 42


def main():

    df = pd.read_csv(DATA_PATH, encoding="latin1")

    df = df.dropna(subset=["Word", "Tag"])


    df["Word"] = df["Word"].astype(str)
    df["Tag"] = df["Tag"].astype(str)


    df["Sentence #"] = df["Sentence #"].ffill()


    df["Word"] = df["Word"].str.strip()
    df["Tag"] = df["Tag"].str.strip()


    sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
    labels_str = df.groupby("Sentence #")["Tag"].apply(list).tolist()

    print("Numar propozitii:", len(sentences))
    print("Exemplu tokens:", sentences[0][:10])
    print("Exemplu labels:", labels_str[0][:10])


    bad = [(i, len(s), len(l)) for i, (s, l) in enumerate(zip(sentences, labels_str)) if len(s) != len(l)]
    if bad:
        print("\nATENTIE: exista propozitii cu lungimi diferite tokens vs labels! Primele 5:")
        for x in bad[:5]:
            print("idx:", x[0], "len(tokens):", x[1], "len(labels):", x[2])
        raise ValueError("Dataset inconsistent: tokens si labels nu au aceeasi lungime pentru unele propozitii.")


    unique_labels = sorted({tag for sent_tags in labels_str for tag in sent_tags})
    label2id = {lab: i for i, lab in enumerate(unique_labels)}
    id2label = {i: lab for lab, i in label2id.items()}

    print("\nNumar label-uri:", len(unique_labels))
    print("Primele label-uri:", unique_labels[:10])

    labels = [[label2id[t] for t in sent_tags] for sent_tags in labels_str]

    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    ds = DatasetDict({
        "train": Dataset.from_dict({"tokens": train_sents, "ner_tags": train_labels}),
        "validation": Dataset.from_dict({"tokens": val_sents, "ner_tags": val_labels}),
    })

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
                    aligned.append(-100)
                elif wid != prev:
                    aligned.append(word_label_ids[wid])
                else:
                    aligned.append(-100)
                prev = wid

            all_labels.append(aligned)

        enc["labels"] = all_labels
        return enc

    ds_tok = ds.map(tokenize_and_align, batched=True, remove_columns=["tokens", "ner_tags"])

    print("\nDupa tokenizare:")
    print(ds_tok)
    print("\nColoane:", ds_tok["train"].column_names)

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

    print("\nSalvat mapping în outputs/label2id.txt si outputs/id2label.txt")


if __name__ == "__main__":
    main()
