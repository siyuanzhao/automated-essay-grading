import re
import os as os
import numpy as np
import itertools
import pandas as pd
from collections import Counter

def load_training_data(training_path, essay_set=1):
    training_df = pd.read_csv(training_path, delimiter='\t')
    # resolved score for essay set 1
    resolved_score = training_df[training_df['essay_set'] == essay_set]['domain1_score']
    essay_ids = training_df[training_df['essay_set'] == essay_set]['essay_id']
    essays = training_df[training_df['essay_set'] == essay_set]['essay']
    essay_list = []
    # turn an essay to a list of words
    for idx, essay in essays.iteritems():
        essay = clean_str(essay)
        #essay_list.append([w for w in tokenize(essay) if is_ascii(w)])
        essay_list.append(tokenize(essay))
    return essay_list, resolved_score.tolist(), essay_ids.tolist()
    
def load_glove(token_num=6, dim=50):
    word2vec = []
    word_idx = {}
    # first word is nil
    word2vec.append([0]*dim)
    count = 1
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "glove/glove."+str(token_num)+
                           "B." + str(dim) + "d.txt")) as f:
        for line in f:
            l = line.split()
            word = l[0]
            vector = map(float, l[1:])
            word_idx[word] = count
            word2vec.append(vector)
            count += 1

    print "==> glove is loaded"

    return word_idx, word2vec

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    >>> tokenize('I don't know')
    ['I', 'don', '\'', 'know']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " ( ", string)
    string = re.sub(r"\)", " ) ", string)
    string = re.sub(r"\?", " ? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()

def build_vocab(sentences, vocab_limit):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    print 'Total size of vocab is {}'.format(len(word_counts.most_common()))
    # Mapping from index to word
    # vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_limit)]
    vocabulary_inv = [x[0] for x in word_counts.most_common(vocab_limit)]
    
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

# data is DataFrame
def vectorize_data(data, word_idx, sentence_size):
    E = []
    for essay in data:
        ls = max(0, sentence_size - len(essay))
        wl = []
        for w in essay:
            if w in word_idx:
                wl.append(word_idx[w])
            else:
                #print '{} is not in vocab'.format(w)
                wl.append(0)
        wl += [0]*ls
        E.append(wl)
    return E
