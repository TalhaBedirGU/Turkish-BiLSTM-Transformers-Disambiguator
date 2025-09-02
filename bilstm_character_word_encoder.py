from preprocess import train_sentences, test_sentences, dev_sentences
from bilstm_preprocess import build_char_vocab, train_char2id, train_id2char, sentence_to_char_ids


import torch
import torch.nn as nn

def collate_sentences(batch_sentences, pad_id):
    """
    IT TAKES:
    batch_sentences: List[Sentence], where a Sentence = List[Word],
                     and a Word = List[int] (char IDs, variable length)
    pad_id: int index for <pad> in char2id

    IT RETURNS:
        char_ids:  LongTensor [B, T_max, L_max]
        word_mask: LongTensor [B, T_max]  (1=real token, 0=pad)
    """
    B = len(batch_sentences)
    if B == 0:
        return (torch.zeros(0, 0, 0, dtype=torch.long),
                torch.zeros(0, 0, dtype=torch.long))

    T_max = max(len(sent) for sent in batch_sentences) # Longest sentence in the batch
    L_max = 0 # Longest word in the batch
    for sent in batch_sentences:
        for word in sent:
            if len(word) > L_max:
                L_max = len(word)

    char_ids = []
    word_mask = []

    for sent in batch_sentences:
        padded_sent = []
        mask_row = []
        # pad/truncate each real word to L_max
        for word in sent:
            w = word[:]  # copy to avoid mutating input
            if len(w) < L_max:
                w = w + [pad_id] * (L_max - len(w))
            else:
                w = w[:L_max]
            padded_sent.append(w)
            mask_row.append(1)

        # pad sentence to T_max words with full pad-words
        num_pad_words = T_max - len(sent)
        if num_pad_words > 0:
            pad_word = [pad_id] * L_max
            for _ in range(num_pad_words):
                padded_sent.append(pad_word)
                mask_row.append(0)

        char_ids.append(padded_sent)
        word_mask.append(mask_row)

    char_ids = torch.tensor(char_ids, dtype=torch.long)   # [B, T_max, L_max]
    word_mask = torch.tensor(word_mask, dtype=torch.long) # [B, T_max]
    return char_ids, word_mask

class CharWordEncoder(nn.Module):
    """
    CharWordEncoder
    ----------------
    IT TAKES: char_ids LongTensor of shape [B, T, L]
              - B = [B]atch size (number of sentences in the batch)
              - T = Number of [T]okens words per sentence (padded to T_max in given batch)
              - L = Number of [L]etters per word (padded to L_max in given batch)

    IT RETURNS: word_vecs FloatTensor of shape [B, T, 2*h_char] (Why 2*h_char --> because this is Birectional LSTM!)
              - one vector per token, produced by a BiLSTM over that token's characters

    """

    def __init__(self, vocab_size, pad_id, d_char=64, h_char=128, dropout=0.2):
        """
        vocab_size  : size of the character vocabulary = len(char2id)
        pad_id      : index of <pad> in the char vocabulary (e.g., 0)
        d_char      : dimension of *char-level* LSTM per direction
        h_char      : hidden size for the *char-level* LSTM per direction
                      (since it's bidirectional, output per word will be 2*h_char)
        dropout     : dropout probability applied to embeddings and outputs
        """
        super(). __init__()

        # Remember the padding id (useful in masks)
        self.pad_id = pad_id

        # 1) Character embedding look-up table.
        #   - nn.Embedding maps integer IDs -> learned vectors.
        #   - padding_idx=pad_id tells PyTorch to keep the <pad> row fixed at zeros
        #     and exclude it from gradient updates.
        self.embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_char,
            padding_idx=pad_id
        )

        # 2) A character-level BiLSTM
        #   - We will feed one word at a time (as a sequence of char embeddings).
        #   - batch_first=True means LSTM expects [batch, seq_len, feat_dim] INSTEAD OF default [seq_len, batch, feat_dim]
        #   - hidden_size=h_char is PER DIRECTION (forward/backward)
        #     So the concatenated final state will be size 2*h_char.
        self.char_lstm = nn.LSTM(
            input_size=d_char, # Recall: d_char is dimension of CHAR-LEVEL LSTM per direction
            hidden_size=h_char,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # 3) Dropout for regularization (this randomly zeroes some activation nodes to help with the generalization)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: torch.LongTensor) -> torch.Tensor:
        """
        IT TAKES:
            char_ids: LongTensor[B,T,L]

        IT RETURSN:
            FloatTensor [B,T,2*h_char]
        """

        # Unpack sizes for readability
        B, T, L = char_ids.shape # batch size, tokens per sentence, chars per token

        # (A) Character embeddings
        #   - INPUT: [B, T, L] (integers) -> OUTPUT: [B, T, L, d_char] (floats)
        #   - Each integer char id becomes a d_char-dimensuonal vector
        E = self.embed(char_ids)        # [B, T, L, d_char]
        E = self.dropout(E)             # applying dropout to training

        # (B) Prepare for char-LSTM!!!
        #   - The char-LSTM operates over WORDS as sequences of chars.
        #   - We currently have sentences x tokens x chars. 
        #   -- Flatten sentences x tokens into ONE BIG "BATCH OF WORDS" so LSTM can process ALL WORDS in parallel.
        #   - INPUT: [B, T, L, d_char] -> [B*T, L, d_char]

        BT = B * T
        E2 = E.view(BT, L, E.size(-1))  # E.size(-1) == d_char

        # (C) Run the char-level BiLSTM
        #   - H: hidden states at each char position (we don't need them here)
        #   - h_n: final gidden states (last timestep) for both directions
        #     SHAPE: [num_directions=2, batch=B*T, h_char]
        #   - We'll concatenate forward and backward final states to get a single fixed-size vector per word.
        H, (h_n, c_n) = self.char_lstm(E2)  # h_n: [2, B*T, h_char]

        # (D) Concatenate final forward/backward hidden states
        #   - h_n[0] is the last hidden state from the forward LSTM
        #   - h_n[1] is the last hidden state from the backward LSTM
        #   Concatenation along the feature dim yields [B*T, 2*h_char]
        word_vec_flat = torch.cat([h_n[0], h_n[1]], dim=1)  # [B*T, 2*h_char]

        # (E) Restore sentence structure
        #   - We currently havve one vector per word for a flattened batch of B*T words.
        #   - Reshape back to [B,T,2*h_char] so we have words grouped by sentence again.
        word_vecs = word_vec_flat.view(B, T, -1)            # Shape: [B, T, 2*h_char]
        word_vecs = self.dropout(word_vecs)                 # Extra dropout because... why not!

        return word_vecs


# ------------------- sanity demo -------------------
if __name__ == "__main__":
    # Toy batch: two sentences, char-idized already
    # sentence1: 3 words with lengths 3,2,1
    # sentence2: 2 words with lengths 2,4
    pad_id = 0
    batch = [
        [[5, 8, 2], [7, 4], [9]],
        [[12, 6], [3, 4, 7, 8]]
    ]

    X, M = collate_sentences(batch, pad_id)
    print("char_ids shape:", tuple(X.shape))   # expect (2, 3, 4)
    print("word_mask:\n", M)                   # expect [[1,1,1],[1,1,0]]

    vocab_size = 100  # example
    enc = CharWordEncoder(vocab_size=vocab_size, pad_id=pad_id, d_char=64, h_char=128, dropout=0.1)
    W = enc(X)
    print("word_vecs shape:", tuple(W.shape))  # expect (2, 3, 256) since 2*h_char = 256