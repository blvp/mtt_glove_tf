import argparse
import os
import pickle
from collections import defaultdict, Counter
from time import clock
from typing import List, Dict

import numpy as np
import pandas as pd


def load_data(path: str):
    with open(path, 'r') as f:
        for line in f:
            yield tokenize(line)


def tokenize(sentence_str: str):
    # TODO: find better solution for this
    return sentence_str.split(" ")


def generate_contexts(sentence: List[str], context_size):
    last_index = len(sentence)

    def window(start, end):
        return sentence[max(start, 0): min(end, last_index)]

    for i, word in enumerate(sentence):
        yield window(i - context_size, i), word, window(i + 1, i + context_size)


def prepare_corpus(path: str, context_size: int = 5, min_count: int = 3) -> (Dict[str, int], np.ndarray):
    """
    :param min_count:
    :param context_size:
    :param path:
    :return: vocab and cooccurrence matrix
    """
    word_counts = Counter()
    cooccurrence = defaultdict(float)
    st = clock()
    for sentence in load_data(path):
        word_counts.update(sentence)
        for l_ctx, word, r_ctx in generate_contexts(sentence, context_size):
            contexts = map(lambda t: (1 / (t[0] + 1), t[1]), list(enumerate(l_ctx[::-1])) + list(enumerate(r_ctx)))
            for coocur, context_word in contexts:
                cooccurrence[(word, context_word)] += coocur
                cooccurrence[(context_word, word)] += coocur
    print("coocurrence calc time: ", clock() - st)
    # filter rare words and index every word so we get id -> word
    st = clock()
    df = pd.DataFrame(list(word_counts.items()), columns=['word', 'counts'])
    df = df[df['counts'] > min_count]
    df = df.reset_index(drop=True)
    vocab = dict([(w, idx) for idx, w in df['word'].to_dict().items()])
    vocab_size = len(vocab)
    print("vocab get time: ", clock() - st)
    print("vocab size", vocab_size)
    st = clock()
    coocur_new = []
    for key, coocur in cooccurrence.items():
        word, context_word = key
        word_id = vocab.get(word, None)
        context_word_id = vocab.get(context_word, None)
        if word_id is not None and context_word_id is not None:
            coocur_new.append((word_id, context_word_id, coocur))
    print("coocurrence calc time", clock() - st)
    return vocab, coocur_new


def get_wiki_corpus_and_dump(
        wiki_file_path,
        context_size=5,
        min_occurences=3,
        save_path='./data/wiki_prepared/',
        overwrite=False
):
    vocab_file = os.path.join(save_path, 'vocab.pkl')
    coocur_file = os.path.join(save_path, 'coocur.pkl')
    if not overwrite:
        if os.path.exists(vocab_file) and os.path.exists(coocur_file):
            with open(vocab_file, 'rb+') as f:
                vocab = pickle.load(f)
            with open(coocur_file, 'rb+') as f:
                coocur = pickle.load(f)
        else:
            raise EnvironmentError("wrong usage of method: when using overwrite=False, please make sure that "
                                   "vocab.pkl and coocur.pkl exists in path %s".format(save_path))
    else:
        vocab, coocur = prepare_corpus(wiki_file_path, context_size, min_occurences)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(vocab_file, 'wb+') as f:
            pickle.dump(vocab, f, protocol=4)
        with open(coocur_file, 'wb+') as f:
            pickle.dump(coocur, f, protocol=4)
    return vocab, coocur


if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--wiki-txt-file', type=str, help='path to wiki file')
    par.add_argument('--save-path', type=str, help='where to store calculated coocurrences')
    par.add_argument('--context-size', type=int, help='context size for coocurrence calculation')
    par.add_argument('--min-occurrences', type=int, help='min occurrences of word pairs to occur in ds for training')
    args = par.parse_args()
    get_wiki_corpus_and_dump(
        args.wiki_txt_file,
        args.context_size,
        args.min_occurrences,
        args.save_path,
        overwrite=True
    )
