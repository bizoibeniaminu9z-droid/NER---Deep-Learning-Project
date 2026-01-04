# src/06_infer.py
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


BASE_MODEL = "bert-base-cased"
MODEL_DIR = Path("outputs") / "bert_ner" / "final"
ID2LABEL_PATH = Path("outputs") / "id2label.txt"

SHOW_SUBTOKENS = False
MIN_SCORE = 0.0


def normalize_outside(label: str) -> str:

    return "O" if label.strip() in ("0", "O") else label.strip()


def load_id2label(path: Path):

    if not path.exists():
        return None

    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return None


    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            out = {int(k): normalize_outside(str(v)) for k, v in obj.items()}
            return out
        if isinstance(obj, list):
            return {i: normalize_outside(str(v)) for i, v in enumerate(obj)}
    except Exception:
        pass

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # 4) detect 2-coloane
    two_col = True
    parsed_pairs = []
    for ln in lines:
        parts = ln.replace("\t", " ").split()
        if len(parts) != 2:
            two_col = False
            break
        parsed_pairs.append(parts)

    if two_col:
        id2label = {}
        for a, b in parsed_pairs:

            a_is_int = a.lstrip("-").isdigit()
            b_is_int = b.lstrip("-").isdigit()

            if a_is_int and not b_is_int:
                idx = int(a)
                lab = b
            elif b_is_int and not a_is_int:
                idx = int(b)
                lab = a
            elif a_is_int and b_is_int:

                idx = int(a)
                lab = b
            else:

                continue

            id2label[idx] = normalize_outside(lab)
        return id2label if id2label else None


    return {i: normalize_outside(line) for i, line in enumerate(lines)}


def merge_wordpieces(tokens, labels, scores):
    merged_tokens, merged_labels, merged_scores = [], [], []

    for tok, lab, sc in zip(tokens, labels, scores):
        if tok.startswith("##") and merged_tokens:
            merged_tokens[-1] = merged_tokens[-1] + tok[2:]

        else:
            merged_tokens.append(tok)
            merged_labels.append(lab)
            merged_scores.append(sc)

    return merged_tokens, merged_labels, merged_scores


def extract_entities(tokens, labels):
    """
    BIO:
      B-geo I-geo -> ("London", "geo")
    """
    entities = []
    current_tokens = []
    current_type = None

    def flush():
        nonlocal current_tokens, current_type
        if current_tokens and current_type:
            entities.append((" ".join(current_tokens), current_type))
        current_tokens = []
        current_type = None

    for tok, lab in zip(tokens, labels):
        lab = normalize_outside(str(lab))

        if lab == "O":
            flush()
            continue

        if "-" in lab:
            prefix, ent_type = lab.split("-", 1)
        else:
            prefix, ent_type = "B", lab

        if prefix == "B":
            flush()
            current_tokens = [tok]
            current_type = ent_type
        elif prefix == "I":
            if current_type == ent_type and current_tokens:
                current_tokens.append(tok)
            else:
                flush()
                current_tokens = [tok]
                current_type = ent_type

    flush()
    return entities


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    print("\nIntrodu textul:")
    user_text = input("> ").strip()

    if user_text:
        TEXT = user_text
    else:
        TEXT = (
            "Thousands of demonstrators marched through London to protest "
            "the war in Iraq and demand the withdrawal of British troops "
            "from that country."
        )


    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    id2label = load_id2label(ID2LABEL_PATH)
    if id2label is None:
        id2label = {int(k): normalize_outside(v) for k, v in model.config.id2label.items()}

    # tokenizare
    enc = tokenizer(TEXT, return_tensors="pt", truncation=True, padding=False)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)
        pred_ids = torch.argmax(probs, dim=-1)[0].tolist()
        pred_scores = torch.max(probs, dim=-1).values[0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    labels = [normalize_outside(str(id2label.get(i, "O"))) for i in pred_ids]

    # scoatem [CLS]/[SEP]
    cleaned = []
    for tok, lab, sc in zip(tokens, labels, pred_scores):
        if tok in ("[CLS]", "[SEP]"):
            continue
        cleaned.append((tok, lab, sc))

    if not cleaned:
        print("Nimic dupa curatare.")
        return

    tokens, labels, scores = map(list, zip(*cleaned))

    if not SHOW_SUBTOKENS:
        tokens, labels, scores = merge_wordpieces(tokens, labels, scores)

    # prag scor: sub MIN_SCORE => O
    labels_filtered = []
    for lab, sc in zip(labels, scores):
        lab = normalize_outside(lab)
        labels_filtered.append("O" if sc < MIN_SCORE else lab)

    print("\nTEXT:\n", TEXT)
    print("\nToken -> Label (primele 80):")
    for i, (tok, lab, sc) in enumerate(zip(tokens, labels_filtered, scores)):
        if i >= 80:
            print("... (trunchiat)")
            break
        print(f"{tok:20} {lab:10}  score={sc:.3f}")

    entities = extract_entities(tokens, labels_filtered)

    print("\nEntitati detectate:")
    if not entities:
        print("(nimic)")
    else:
        for ent_text, ent_type in entities:
            print(f"- {ent_text}  [{ent_type}]")

if __name__ == "__main__":
    main()
