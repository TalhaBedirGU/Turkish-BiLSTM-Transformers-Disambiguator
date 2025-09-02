from preprocess import train_sentences, test_sentences, dev_sentences
from bilstm_preprocess import train_char2id, train_id2char, train_upos2id, train_id2upos, sentence_to_char_ids
from bilstm_upos_process import sentences_to_upos_ids, SentenceUPOSProbe
from bilstm_character_word_encoder import CharWordEncoder, collate_sentences

import random

import torch
import torch.nn as nn
import torch.optim as optim

def masked_token_cross_entropy(logits, gold, word_mask):
    B,T,C = logits.shape
    logits_f = logits.view(B*T, C)
    gold_f   = gold.view(B*T)
    mask_f   = word_mask.view(B*T).bool()
    return nn.CrossEntropyLoss()(logits_f[mask_f], gold_f[mask_f])

@torch.no_grad()
def masked_accuracy(logits, gold, word_mask):
    pred = logits.argmax(dim=-1)
    correct = ((pred == gold) & word_mask.bool()).sum().item()
    total   = word_mask.sum().item()
    return correct / max(1, total)

# --------------- Data loaders (simple) -------------
def make_batches(data, batch_size, shuffle=True):
    idxs = list(range(len(data)))
    if shuffle: random.shuffle(idxs)
    for i in range(0, len(data), batch_size):
        yield [data[j] for j in idxs[i:i+batch_size]]

def to_device(t, device):
    return t.to(device) if torch.is_tensor(t) else t

# --------------- Train / Eval loops ----------------
def run_epoch(sentences, batch_size, char2id, upos2id, char_enc, upos_head, device, train=True, lr=1e-3):
    pad_id = char2id['<pad>']
    if train:
        char_enc.train(); upos_head.train()
        opt = optim.Adam(list(char_enc.parameters()) + list(upos_head.parameters()), lr=lr)
    else:
        char_enc.eval(); upos_head.eval()
    total_loss, total_acc, total_tokens = 0.0, 0.0, 0

    for batch_raw in make_batches(sentences, batch_size, shuffle=train):
        # chars
        batch_char = [sentence_to_char_ids(s, char2id) for s in batch_raw]
        char_ids, word_mask = collate_sentences(batch_char, pad_id)            # [B,T,L],[B,T]
        gold = sentences_to_upos_ids(batch_raw, upos2id, pad_val=0)            # [B,T]
        char_ids = char_ids.to(device); word_mask = word_mask.to(device); gold = gold.to(device)

        # forward
        word_vecs = char_enc(char_ids)                                         # [B,T,D_word]
        logits, H_ctx = upos_head(word_vecs)                                         # [B,T,n_upos]
        loss = masked_token_cross_entropy(logits, gold, word_mask)
        acc  = masked_accuracy(logits, gold, word_mask)

        if train:
            opt.zero_grad(); loss.backward(); opt.step()

        total_loss += loss.item() * word_mask.sum().item()
        total_acc  += acc * word_mask.sum().item()
        total_tokens += word_mask.sum().item()

    avg_loss = total_loss / max(1, total_tokens)
    avg_acc  = total_acc  / max(1, total_tokens)
    return avg_loss, avg_acc

# --------------- Main ------------------------------
def main():

    # ---- models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pad_id = train_char2id['<pad>']
    char_enc = CharWordEncoder(vocab_size=len(train_char2id), pad_id=pad_id, d_char=64, h_char=128, dropout=0.2).to(device)
    D_word = 2*128
    upos_head = SentenceUPOSProbe(d_word=D_word, h_word=256, n_upos=len(train_upos2id), dropout=0.2).to(device)

    # ---- train a few epochs
    batch_size = 32
    epochs = 5
    best_dev = -1.0
    for ep in range(1, epochs+1):
        tr_loss, tr_acc = run_epoch(train_sentences, batch_size, train_char2id, train_upos2id, char_enc, upos_head, device, train=True, lr=1e-3)
        dv_loss, dv_acc = run_epoch(dev_sentences,   batch_size, train_char2id, train_upos2id, char_enc, upos_head, device, train=False)
        print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | dev loss {dv_loss:.4f} acc {dv_acc:.4f}")
        if dv_acc > best_dev:
            best_dev = dv_acc
            torch.save({"char_enc": char_enc.state_dict(),
                        "upos_head": upos_head.state_dict(),
                        "char2id": train_char2id,
                        "upos2id": train_upos2id}, "best_upos_baseline.pt")

    # ---- test with best model (optional reload)
    ckpt = torch.load("best_upos_baseline.pt", map_location=device)
    char_enc.load_state_dict(ckpt["char_enc"])
    upos_head.load_state_dict(ckpt["upos_head"])

    te_loss, te_acc = run_epoch(test_sentences, batch_size, train_char2id, train_upos2id, char_enc, upos_head, device, train=False)
    print(f"TEST  | loss {te_loss:.4f} acc {te_acc:.4f}")

if __name__ == "__main__":
    main()