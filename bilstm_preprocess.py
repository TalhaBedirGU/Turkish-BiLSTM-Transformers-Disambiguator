from preprocess import train_sentences, test_sentences, dev_sentences
from collections import Counter
from collections import defaultdict


# THE CHARACTER VOCABULARY
def build_char_vocab(sentences):

    """
    This function takes a sentences that are presentend as set of 
    dicts that has UD style keys: form, lemma, feats etc. 
    And it returns all unique characters with indices as a dict.
    I also added padding and unknown tokens at the beginning.
    """

    all_chars = ""
    for entry in sentences:
        for dict in entry:
            all_chars += dict['form']

    unique_all_chars = sorted(list(set(all_chars)))
    unique_all_chars = ['<pad>', '<unk>'] + unique_all_chars

    char2id = {char: index for index, char in enumerate(unique_all_chars)}
    id2char = {index: char for index, char in enumerate(unique_all_chars)}

    return char2id, id2char

# Applying char extraction to train_sentences
train_char2id, train_id2char = build_char_vocab(train_sentences)
print(f"All characters in our inventory: {train_char2id}")



def sentence_to_char_ids(sentence, char2id, max_len=None):
    """
    sentence: List[Dict], one UD sentence (tokens)
    return: List[List[int]] (one char-ID list per token)
    """
    current_sentence = []
    for token in sentence:
        ids = [char2id.get(ch, char2id['<unk>']) for ch in token['form']]
        
        if max_len is not None:
            if len(ids) > max_len: 
                ids = ids[:max_len] # Truncate
            else:
                ids += [char2id['<pad>']] * (max_len - len(ids)) # Padding
        
        current_sentence.append(ids)
    
    return current_sentence

#Trying our function
print(f"An example sentence in id indexes: {sentence_to_char_ids(train_sentences[0], train_char2id, max_len=20)}")


# THE UPOS VOCABULARY
def build_upos_vocab(sentences):
    """
    This function gives out all unique UPOS tags with indices
    """

    all_upos = []
    for entry in sentences:
        for dict in entry:
            all_upos += [dict['upos']]
    
    unique_all_uposes = sorted(list(set(all_upos)))

    upos2id = {upos: index for index, upos in enumerate(unique_all_uposes)}
    id2upos = {index: upos for index, upos in enumerate(unique_all_uposes)}

    return upos2id, id2upos

# Applying upos extraction to train_sentences
train_upos2id, train_id2upos = build_upos_vocab(train_sentences)
print(f"All UPOS tags in our inventory: {train_upos2id}")

# THE FEATS VOCABULARY
def build_feat_vocab(sentences):
    """
    This function gives out all unique feature tags with indices
    """
    
    all_features = set()
    for entry in sentences:
        for dict in entry:
            if dict['feats']:
                for feat, value in dict['feats'].items():
                    all_features.add(f"{feat}={value}")
    
    all_features = sorted(all_features)
    feat2id = {feat: index for index, feat in enumerate(all_features)}
    id2feat = {index: feat for index, feat in enumerate(all_features)}

    return feat2id, id2feat

train_feat2id, train_id2feat = build_feat_vocab(train_sentences)
print(f"All features in our inventory: {train_feat2id}")