from preprocess import train_sentences, test_sentences, dev_sentences
from bilstm_preprocess import train_char2id, train_id2char, train_upos2id, train_id2upos, sentence_to_char_ids
from bilstm_upos_process import sentences_to_upos_ids, SentenceUPOSProbe
from bilstm_character_word_encoder import CharWordEncoder, collate_sentences

import torch

def load_models(ckpt_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device)

    char2id = ckpt["char2id"]
    upos2id = ckpt["upos2id"]
    id2upos = {i:t for t,i in upos2id.items()}
    pad_id = char2id["<pad>"]

    # same hyperparams as training
    d_char = 64
    h_char = 128
    d_word = 2*h_char
    h_word = 256
    n_upos = len(upos2id)

    char_enc = CharWordEncoder(len(char2id), pad_id, d_char=d_char, h_char=h_char).to(device)
    upos_head = SentenceUPOSProbe(d_word, h_word=h_word, n_upos=n_upos).to(device)

    char_enc.load_state_dict(ckpt["char_enc"])
    upos_head.load_state_dict(ckpt["upos_head"])

    char_enc.eval()
    upos_head.eval()
    return char_enc, upos_head, char2id, id2upos, device

# ---------- single-word predictor ----------
def predict_upos_for_word(word, char2id, char_encoder, upos_head, id2upos, device="cpu", max_len=30):
    # chars -> ids (pad/truncate)
    ids = [char2id.get(ch, char2id["<unk>"]) for ch in word[:max_len]]
    if len(ids) < max_len:
        ids += [char2id["<pad>"]] * (max_len - len(ids))
    x = torch.tensor([[ids]], dtype=torch.long, device=device)  # [1,1,L]

    with torch.no_grad():
        word_vecs = char_encoder(x)              # [1,1,D_word]
        logits, H_ctx = upos_head(word_vecs)            # [1,1,n_upos]  (returns ONLY logits)
        pred_id = int(logits.argmax(dim=-1).item())
    return id2upos[pred_id]

# ---------- optional: sentence helper ----------
def predict_upos_for_sentence(tokens, char2id, char_encoder, upos_head, id2upos, device="cpu", max_len=30):
    # tokens: list of strings (already tokenized words)
    ids_batch = []
    for w in tokens:
        ids = [char2id.get(ch, char2id["<unk>"]) for ch in w[:max_len]]
        if len(ids) < max_len:
            ids += [char2id["<pad>"]] * (max_len - len(ids))
        ids_batch.append(ids)
    x = torch.tensor([ids_batch], dtype=torch.long, device=device)  # [1,T,L]
    with torch.no_grad():
        word_vecs = char_encoder(x)           # [1,T,D_word]
        logits, H_ctx = upos_head(word_vecs)         # [1,T,n_upos]
        pred_ids = logits.argmax(dim=-1).squeeze(0).tolist()
    return [id2upos[i] for i in pred_ids]

if __name__ == "__main__":
    # path to the checkpoint you saved during training
    ckpt_path = "best_upos_baseline.pt"
    char_enc, upos_head, char2id, id2upos, device = load_models(ckpt_path)

    # TRYING A WORD
    print(predict_upos_for_word("çevredekilere", char2id, char_enc, upos_head, id2upos, device))
    print(predict_upos_for_word("koşuyorum", char2id, char_enc, upos_head, id2upos, device))
    sent = ["Ben", "geldim", "."]
    print(list(zip(sent, predict_upos_for_sentence(sent, char2id, char_enc, upos_head, id2upos, device))))