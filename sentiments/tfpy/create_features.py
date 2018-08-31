import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
n_lines = 1000000

def create_lexicon(files):
    lexicon = []
    for fi in files:
        with open (fi, 'r') as f:
            contents = f.readlines()
            for l in contents:
                try:
                    all_words = word_tokenize(
                        l.lower()
                    )
                except UnicodeDecodeError:
                    pass
                lexicon += list(all_words)

    l_lexicon = []
    for w in lexicon:
        try:
            l = lemmatizer.lemmatize(w)
            l_lexicon.append(l)
        except UnicodeDecodeError:
            pass

    w_counts = Counter(l_lexicon)

    l2 = []
    for w in w_counts:
        # Only keep common words, but not too common
        if 1000 > w_counts[w] > 50:
            l2.append(w)

    return l2

def sample_handling(sample, lexicon, classification):
    featureset = []

    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:n_lines]:
            try:
                current_words = word_tokenize(l.lower())
                current_words_l = []
                for w in current_words:
                    try:
                        l = lemmatizer.lemmatize(w)
                        current_words_l.append(l)
                    except UnicodeDecodeError:
                        pass

                features = np.zeros(len(lexicon))
                for word in current_words_l:
                    if word.lower() in lexicon:
                        index_value = lexicon.index(word.lower())
                        features[index_value] += 1

                features = list(features)
                featureset.append([features, classification])
            except UnicodeDecodeError:
                pass

    return featureset

def create_featureset_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon([pos, neg])
    features = []
    features += sample_handling('pos.txt', lexicon, [1,0])
    features += sample_handling('neg.txt', lexicon, [0,1])
    random.shuffle(features)

    features = np.array(features)

    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x, train_y, test_x, test_y

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_featureset_and_labels('pos.txt', 'neg.txt')
    with open('sentiment_set.pickle', 'wb') as f:
        pickle.dump([
            train_x,
            train_y,
            test_x,
            test_y
        ], f)
