**GOAL**: Given one word and one context, I wanted to run a bidirectional LSTM to disambiguate a Turkish word in terms of two features in Universal Dependencies:
UPOS tag such as NUM, NOUN, ADJ, ADV, etc.
Features such as Person=3, Case=Dat, etc.

**GOAL 2**: I wanted to try the same in Transformers architecture as well.

**Current state**:
- I have downloaded the train, test, and dev files from UD_Turkish-BOUN Treebank - a treebank I annotated so much of during my Master’s! Then I preprocess the conllu files into neat dictionaries so that I can see word forms, lemmas, UPOS tags and feats. (preprocess.py)
- Then I have started preparing the file to BiLSTM. I have made character-based, UPOS-based and Feats-based vocabularies that output feature, index pairs. (bilstm_preprocess.py)
- I created a character word encoder and started integrating it with BiLSTM sentence encoder. (bilstm_character_word_encoder.py)
- Similarly, I have created sentence UPOS probe that took UPOS tags with indices and trained it along with the word sequences. (bilstm_upos_process.py)
- I trained train_sentences using character encoder and UPOS probe—and calculated loss & evaluation scores after epochs after testing on test_sentences. (bilstm_char_upos_loss_eval.py)
- I also created a file where we can enter a word and a sentence and the disambiguator would predict the individual and sequential UPOS tags.


**What’s missing**:
- Feature based disambiguation
- Transformers and comparison
- Comparing results against the state of art (My current training returns around 87% accuracy)
