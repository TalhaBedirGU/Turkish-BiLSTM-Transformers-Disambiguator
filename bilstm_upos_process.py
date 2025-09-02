from preprocess import train_sentences, test_sentences, dev_sentences
from bilstm_preprocess import build_upos_vocab, train_upos2id, train_id2upos, train_char2id, train_id2char, sentence_to_char_ids
from bilstm_character_word_encoder import CharWordEncoder, collate_sentences

import torch
import torch.nn as nn

def sentences_to_upos_ids(batch_raw_sentences, upos2id, pad_val=0):
    # pad each sentence on the right to the longest length in the batch
    T_max = max(len(sentence) for sentence in batch_raw_sentences) if batch_raw_sentences else 0

    Y = []
    for sent in batch_raw_sentences:
        row = [upos2id[token['upos']] for token in sent]
        row += [pad_val] * (T_max - len(row))
        Y.append(row)

    return torch.tensor(Y, dtype=torch.long)  # [B, T_max]

class SentenceUPOSProbe(nn.Module):
    """
    Sentence-level BiLSTM + UPOS classifier.

    IT TAKES:
        - word_vecs:    FloatTensor [B, T, D_word]
            D_word typically = 2*h_char from your char-level encoder (e.g., 256)
    
    IT RETURNS:
        - logits:       FloatTensor [B, T, n_upos]
        - context_states    FloatTensor [B, T, 2*h_word] (contextual token representations)
    """

    def __init__(self, d_word, h_word=256, n_upos=17, dropout=0.2):
        """
        d_word  : dimension of each input word vector (from CharWordEncoder; e.g., 256)
        h_word  : hidden size of the WORD-LEVEL LSTM per direction
                  (BiLMST -> output per token is 2*h_word)
        n_upos  : number of UPOS classes (build from training set)
        dropout : dropout prob applied to LSTM outputs before classification
        """
        super().__init__()

        # Word-level BiLSTM (sequence over TOKENS within EACH SENTENCE)
        # batch_first=True → expects input as [batch, seq_len, feat_dim] = [B, T, D_word] RATHER THAN the default [seq, batch, feat]
        self.word_lstm = nn.LSTM(
            input_size = d_word,
            hidden_size = h_word,
            num_layers = 1,
            batch_first = True,
            bidirectional = True
        )

        # Applying dropout to reduce overfitting
        self.dropout = nn.Dropout(p=dropout)

        #Linear classifier: maps each contextual token vector (2*h_word) -> logits over UPOS
        self.classifier = nn.Linear(in_features=2*h_word, out_features=n_upos)
            # So here;
            #	•	in_features=2*h_word: the size of the input vector per token (from BiLSTM).
	        #   •	out_features=n_upos: the size of the output vector we want (one logit per UPOS class).
	        #   •	PyTorch will automatically learn a weight matrix of shape [n_upos, 2*h_word] plus a bias of shape [n_upos].
            # IN SHORT: Each token vector is transformed into a logit vector of length n_upos.
            #           Like... “given this contextual vector, how much does it look like a NOUN, how much like a VERB, …?”

    def forward(self, word_vecs):
        """
        IT TAKES:
            word_vecs: [B, T, D_word] 

        IT RETURNS:
            logits: [B, T, n_upos]
            H_context: [B, T, 2*h_word] (you can reuse these for other heads later)
        """
        # Run the BiLSTM over tokens. It reads left→right and right→left, (bidirectional yay!)
        # then concatenates the per-token hidden states from both directions.
        # Output H_context has one contextual vector per token: [B, T, 2*h_word]
        H_context, _ = self.word_lstm(word_vecs)    # [B, T, 2*h_word]

        # Dropout before classification
        H_context = self.dropout(H_context)         # [B, T, 2*h_word]

        # Classify each token independently: a single linear layer shared across timesteps.
        logits = self.classifier(H_context)         # [B, T, n_upos]

        return logits, H_context
        


def masked_token_cross_entropy(logits, gold, word_mask):
    """
    logits:    [B, T, C]  (C = n_upos)
    gold:      [B, T]     (int class ids; arbitrary value at padded positions)
    word_mask: [B, T]     (1 for real tokens, 0 for padded)
    returns: scalar loss over *only* real tokens
    """
    B, T, C = logits.shape

    # Flatten time & batch so we can select only real tokens easily
    logits_flat = logits.view(B*T, C)        # [B*T, C]
    gold_flat   = gold.view(B*T)             # [B*T]
    mask_flat   = word_mask.view(B*T).bool() # [B*T]

    # Select real-token rows; ignore padded positions
    logits_sel = logits_flat[mask_flat]      # [N_real, C]
    gold_sel   = gold_flat[mask_flat]        # [N_real]

    # Standard cross-entropy on the selected rows
    return nn.CrossEntropyLoss()(logits_sel, gold_sel)

@torch.no_grad()
def masked_accuracy(logits, gold, word_mask):
    """
    Same shapes as above. Computes accuracy over non-padded tokens only.
    """
    pred = logits.argmax(dim=-1)                 # [B, T]
    correct = ((pred == gold) & word_mask.bool()).sum().item()
    total   = word_mask.sum().item()
    return correct / max(1, total)

def upos_probe_step(batch_raw_sentences,  # list of sentences (token dicts) length B
                    char2id, pad_id, upos2id,
                    char_encoder, upos_probe):
    # 1) char-IDs for the same sentences
    batch_char = [sentence_to_char_ids(s, char2id) for s in batch_raw_sentences]
    char_ids, word_mask = collate_sentences(batch_char, pad_id)   # [B,T,L], [B,T]

    # 2) Char → Word vectors
    word_vecs = char_encoder(char_ids)                            # [B,T,D_word]

    # 3) Gold UPOS ids (padded on the right to T_max)
    gold_upos = sentences_to_upos_ids(batch_raw_sentences, upos2id, pad_val=0)  # [B,T]

    # 4) Predict + loss
    logits, _ = upos_probe(word_vecs)                             # [B,T,n_upos]
    loss = masked_token_cross_entropy(logits, gold_upos, word_mask)
    acc  = masked_accuracy(logits, gold_upos, word_mask)
    return loss, acc, logits


# SANITY CHECK!!!

print("Sanity check for UPOS Probe with dummy data:\n")

pad_id = 0
char_enc = CharWordEncoder(vocab_size=len(train_char2id), pad_id=pad_id, d_char=64, h_char=128, dropout=0.2)
D_word = 2*128
upos_probe = SentenceUPOSProbe(d_word=D_word, h_word=256, n_upos=len(train_upos2id), dropout=0.2)

# tiny batch (e.g., 2 sentences)
batch_raw = [train_sentences[0], train_sentences[1]]
loss, acc, logits = upos_probe_step(batch_raw, train_char2id, pad_id, train_upos2id, char_enc, upos_probe)

print("logits shape:", tuple(logits.shape))  # expect (B, T_max, n_upos)
print("loss:", float(loss))
print("acc:", acc)


        
