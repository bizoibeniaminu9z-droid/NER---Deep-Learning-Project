import os
import json
from pathlib import Path

import numpy as np
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate

# -----------------------
# Config
# -----------------------
MODEL_NAME = "bert-base-cased"
MAX_LEN = 128

# Pentru 3060 12GB: safe + "serios"
BATCH_SIZE = 16              # dacă dă OOM, scazi la 8
GRAD_ACCUM = 2               # batch = 16*2 = 32
LR = 2e-5
EPOCHS = 4                   # 3-5 e ok
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06

OUT_DIR = Path("outputs") / "bert_ner"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL2ID_PATH = Path("outputs") / "label2id.txt"
ID2LABEL_PATH = Path("outputs") / "id2label.txt"

# Ca să fie determinist-ish
SEED = 42


def load_label_maps():
    label2id = {}
    id2label = {}

    # label2id.txt: "LABEL<TAB>ID"
    for line in LABEL2ID_PATH.read_text(encoding="utf-8").splitlines():
        k, v = line.split("\t")
        label2id[k] = int(v)

    # id2label.txt: "ID<TAB>LABEL"
    for line in ID2LABEL_PATH.read_text(encoding="utf-8").splitlines():
        k, v = line.split("\t")
        id2label[int(k)] = v

    return label2id, id2label


def build_dataset_again():
    """
    Refacem tokenizarea exact ca în 04_prepare_hf_dataset.py
    (ca să nu depindem de caching intern).
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from datasets import Dataset

    DATA_PATH = Path("data") / "NER dataset.csv"

    df = pd.read_csv(DATA_PATH, encoding="latin1")
    df = df.dropna(subset=["Word", "Tag"])
    df["Word"] = df["Word"].astype(str).str.strip()
    df["Tag"] = df["Tag"].astype(str).str.strip()
    df["Sentence #"] = df["Sentence #"].ffill()

    sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
    labels_str = df.groupby("Sentence #")["Tag"].apply(list).tolist()

    label2id, id2label = load_label_maps()
    labels = [[label2id[t] for t in sent_tags] for sent_tags in labels_str]

    train_sents, val_sents, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.1, random_state=SEED
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
    return ds_tok


def main():
    label2id, id2label = load_label_maps()
    num_labels = len(label2id)

    print("Număr label-uri:", num_labels)
    print("Exemple label-uri:", list(label2id.items())[:10])

    # Dataset tokenizat
    ds_tok = build_dataset_again()
    print(ds_tok)

    # Model
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # Metrici NER (seqeval)
    seqeval = evaluate.load("seqeval")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels = []
        true_preds = []

        for pred_seq, label_seq in zip(preds, labels):
            cur_true = []
            cur_pred = []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                cur_true.append(id2label[int(l)])
                cur_pred.append(id2label[int(p)])
            true_labels.append(cur_true)
            true_preds.append(cur_pred)

        results = seqeval.compute(predictions=true_preds, references=true_labels)

        # results conține: overall_precision/recall/f1/accuracy + pe clase uneori
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    # Training args
    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        seed=SEED,

        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,

        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,

        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,

        fp16=True,  # 3060 -> mare boost
        report_to="none",

        # Ca să nu-ți moară RAM-ul pe Windows
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n--- TRAIN START ---")
    train_result = trainer.train()
    print("\n--- TRAIN DONE ---")

    print("\nEvaluare finală:")
    metrics = trainer.evaluate()
    print(metrics)

    # Salvăm modelul final
    trainer.save_model(str(OUT_DIR / "final"))
    tokenizer.save_pretrained(str(OUT_DIR / "final"))

    # Salvăm metricile în fișier
    (OUT_DIR / "final_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n✅ Model salvat în: {OUT_DIR / 'final'}")
    print(f"✅ Metrici salvate în: {OUT_DIR / 'final_metrics.json'}")


if __name__ == "__main__":
    main()
