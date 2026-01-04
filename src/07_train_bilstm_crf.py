# src/07_train_bilstm_crf.py
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from seqeval.metrics import f1_score, precision_score, recall_score



CSV_PATH = Path("data") / "NER dataset.csv"
OUT_DIR = Path("outputs") / "bilstm_crf"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_LEN = 128  

EMB_DIM = 100    
HID_DIM = 128
DROPOUT = 0.2

TRAIN_SPLIT = 0.9
GRAD_CLIP = 1.0



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_conll_csv(csv_path: Path):

    df = pd.read_csv(csv_path, encoding="latin1")
    df["Sentence #"] = df["Sentence #"].ffill()

    df["Word"] = df["Word"].astype(str)
    df["Tag"] = df["Tag"].astype(str)

    sentences = df.groupby("Sentence #")["Word"].apply(list).tolist()
    labels = df.groupby("Sentence #")["Tag"].apply(list).tolist()
    return sentences, labels


def split_train_val(sentences, labels, split=0.9, seed=42):
    idx = list(range(len(sentences)))
    random.Random(seed).shuffle(idx)
    cut = int(len(idx) * split)
    tr_idx, va_idx = idx[:cut], idx[cut:]

    tr_s = [sentences[i] for i in tr_idx]
    tr_l = [labels[i] for i in tr_idx]
    va_s = [sentences[i] for i in va_idx]
    va_l = [labels[i] for i in va_idx]
    return tr_s, tr_l, va_s, va_l


def build_maps(sentences, labels):
    word2id = {"<PAD>": 0, "<UNK>": 1}
    for sent in sentences:
        for w in sent:
            if w not in word2id:
                word2id[w] = len(word2id)

    uniq_labels = sorted({lab for seq in labels for lab in seq})
    if "O" not in uniq_labels:
        uniq_labels = ["O"] + uniq_labels

    label2id = {lab: i for i, lab in enumerate(uniq_labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return word2id, label2id, id2label


class NERDataset(Dataset):
    def __init__(self, sentences, labels, word2id, label2id, max_len):
        self.sentences = sentences
        self.labels = labels
        self.word2id = word2id
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words = self.sentences[idx][: self.max_len]
        tags = self.labels[idx][: self.max_len]

        x = [self.word2id.get(w, self.word2id["<UNK>"]) for w in words]
        y = [self.label2id[t] for t in tags]

        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def collate_fn(batch, pad_word_id=0, pad_label_id=0):

    xs, ys = zip(*batch)
    lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
    max_len = int(lengths.max().item())

    x_pad = torch.full((len(xs), max_len), pad_word_id, dtype=torch.long)
    y_pad = torch.full((len(xs), max_len), pad_label_id, dtype=torch.long)
    mask = torch.zeros((len(xs), max_len), dtype=torch.bool)

    for i, (x, y) in enumerate(zip(xs, ys)):
        L = len(x)
        x_pad[i, :L] = x
        y_pad[i, :L] = y
        mask[i, :L] = True

    return x_pad, y_pad, mask, lengths


def log_sum_exp(tensor, dim=-1):
    max_score, _ = tensor.max(dim)
    return max_score + torch.log(torch.sum(torch.exp(tensor - max_score.unsqueeze(dim)), dim))


class LinearCRF(nn.Module):

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags

        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):

        log_den = self._compute_log_partition(emissions, mask)
        log_num = self._compute_score(emissions, tags, mask)
        nll = (log_den - log_num)
        return nll.mean()

    def decode(self, emissions, mask):
        return self._viterbi_decode(emissions, mask)

    def _compute_score(self, emissions, tags, mask):
        B, T, C = emissions.size()
        mask = mask.float()


        score = self.start_transitions[tags[:, 0]] + emissions[:, 0, :].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, T):
            emit_t = emissions[:, t, :].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_t = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (emit_t + trans_t) * mask[:, t]


        lengths = mask.long().sum(dim=1) 
        last_tag = tags.gather(1, (lengths - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tag]
        return score

    def _compute_log_partition(self, emissions, mask):
        B, T, C = emissions.size()
        mask = mask.bool()


        alpha = self.start_transitions + emissions[:, 0, :] 

        for t in range(1, T):
            emit_t = emissions[:, t, :].unsqueeze(1)  
            trans = self.transitions.unsqueeze(0)     

            scores = alpha.unsqueeze(2) + trans + emit_t  
            new_alpha = log_sum_exp(scores, dim=1)       

            alpha = torch.where(mask[:, t].unsqueeze(1), new_alpha, alpha)

        alpha = alpha + self.end_transitions
        return log_sum_exp(alpha, dim=1)  

    def _viterbi_decode(self, emissions, mask):
        B, T, C = emissions.size()
        mask = mask.bool()


        score = self.start_transitions + emissions[:, 0, :]  
        history = []

        for t in range(1, T):
            broadcast_score = score.unsqueeze(2)            
            broadcast_trans = self.transitions.unsqueeze(0) 
            next_score = broadcast_score + broadcast_trans  
            best_score, best_path = next_score.max(dim=1)  
            best_score = best_score + emissions[:, t, :]  

            history.append(best_path)

            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)

        score = score + self.end_transitions
        best_last_score, best_last_tag = score.max(dim=1)  


        lengths = mask.long().sum(dim=1)
        best_paths = []

        for b in range(B):
            L = int(lengths[b].item())
            last_tag = int(best_last_tag[b].item())
            path = [last_tag]

            for t in range(L - 1, 1, -1):
                last_tag = int(history[t - 1][b, last_tag].item())
                path.append(last_tag)


            if L > 1:
                last_tag = int(history[0][b, last_tag].item())
                path.append(last_tag)

            path.reverse()
            best_paths.append(path)

        return best_paths


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, num_labels, emb_dim, hid_dim, dropout, pad_id):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 2, num_labels)
        self.crf = LinearCRF(num_labels)

    def forward(self, x, lengths):

        z = self.emb(x)
        packed = pack_padded_sequence(z, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = self.drop(out)
        emissions = self.fc(out)
        return emissions

    def loss(self, x, y, mask, lengths):
        emissions = self.forward(x, lengths)
        T_prime = emissions.size(1)
        y = y[:, :T_prime]
        mask = mask[:, :T_prime]
        return self.crf(emissions, y, mask)

    def decode(self, x, mask, lengths):
        emissions = self.forward(x, lengths)
        T_prime = emissions.size(1)
        mask = mask[:, :T_prime]
        return self.crf.decode(emissions, mask)


def evaluate(model, loader, id2label, device):
    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y, mask, lengths in loader:
            x, y, mask, lengths = x.to(device), y.to(device), mask.to(device), lengths.to(device)

            pred_ids = model.decode(x, mask, lengths)

            y_np = y.cpu().numpy()
            mask_np = mask.cpu().numpy()

            for b in range(x.size(0)):
                true_seq = []
                pred_seq = []

                L = int(mask_np[b].sum())
                for t in range(L):
                    true_seq.append(id2label[int(y_np[b, t])])
                    pred_seq.append(id2label[int(pred_ids[b][t])])

                all_true.append(true_seq)
                all_pred.append(pred_seq)

    p = precision_score(all_true, all_pred)
    r = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    sentences, labels = read_conll_csv(CSV_PATH)
    print("Număr propoziții:", len(sentences))
    print("Exemplu:", sentences[0][:12], labels[0][:12])

    tr_s, tr_l, va_s, va_l = split_train_val(sentences, labels, TRAIN_SPLIT, SEED)

    word2id, label2id, id2label = build_maps(tr_s, tr_l)
    print("Vocab words:", len(word2id), "| Num labels:", len(label2id))

    train_ds = NERDataset(tr_s, tr_l, word2id, label2id, MAX_LEN)
    val_ds = NERDataset(va_s, va_l, word2id, label2id, MAX_LEN)

    pad_word_id = word2id["<PAD>"]
    pad_label_id = label2id["O"] 

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_word_id, pad_label_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_word_id, pad_label_id),
    )

    model = BiLSTMCRF(
        vocab_size=len(word2id),
        num_labels=len(label2id),
        emb_dim=EMB_DIM,
        hid_dim=HID_DIM,
        dropout=DROPOUT,
        pad_id=pad_word_id,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_f1 = -1.0
    best_path = OUT_DIR / "best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y, mask, lengths in train_loader:
            x, y, mask, lengths = x.to(device), y.to(device), mask.to(device), lengths.to(device)

            loss = model.loss(x, y, mask, lengths)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        metrics = evaluate(model, val_loader, id2label, device)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  train_loss:    {avg_loss:.4f}")
        print(f"  val_precision: {metrics['precision']:.4f}")
        print(f"  val_recall:    {metrics['recall']:.4f}")
        print(f"  val_f1:        {metrics['f1']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_path)
            print(f"  ✅ saved best -> {best_path}")


    (OUT_DIR / "word2id.json").write_text(json.dumps(word2id, indent=2), encoding="utf-8")
    (OUT_DIR / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
    (OUT_DIR / "id2label.json").write_text(json.dumps({str(k): v for k, v in id2label.items()}, indent=2), encoding="utf-8")

    cfg = {
        "csv": str(CSV_PATH),
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "emb_dim": EMB_DIM,
        "hid_dim": HID_DIM,
        "dropout": DROPOUT,
        "seed": SEED,
        "best_f1": best_f1,
    }
    (OUT_DIR / "train_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print("\n--- DONE ---")
    print("Best F1:", best_f1)
    print("Model:", best_path)
    print("Maps saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
