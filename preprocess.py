from conllu import parse_incr
from pathlib import Path

def read_conllu(filepath):
    
    """ 
    This function preprocesses UD-annotated sentences from a conllu file and gives out list of dicts.

    takes: 
        a connllu file specified with a file path

    returns: 
        a list of dicts where keys are forms, lemma, upos, an most importantly... feats!
    """


    sentences = []
    with open(filepath, 'r', encoding='utf-8') as our_file:
        for tokenlist in parse_incr(our_file):
            sentence = []
            for token in tokenlist:
                if isinstance(token['id'], int): # to skip multiword tokens or empty nodes
                    sentence.append(
                        {
                        'form': token['form'],
                        'lemma': token['lemma'],
                        'upos': token['upostag'],
                        'feats': token['feats'] or {} 
                        }
                    )
            sentences.append(sentence)

    return sentences


# Our data:

data_directory = Path("ud_turkish_boun")
train_path = data_directory / "tr_boun-ud-train.conllu"
test_path = data_directory / "tr_boun-ud-test.conllu"
dev_path = data_directory / "tr_boun-ud-dev.conllu"

train_sentences = read_conllu(train_path)
test_sentences = read_conllu(test_path)
dev_sentences = read_conllu(dev_path)

print(f"An example entry: \n {train_sentences[0]}")

all_words = []
for sentence in train_sentences:
    for word in sentence:
        all_words.append(word['form'])

print("\nSome data about the input\n")
print(f"The total count of words: {len(all_words)}")
print(f"The lengthg of the longest words is: {len(max(all_words, key=len))}")
        









            


