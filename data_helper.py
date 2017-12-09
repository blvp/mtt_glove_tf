from typing import List, Dict, Tuple

from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import pickle
import os

import argparse

import sys


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


def prepare_corpus(path: str, context_size: int = 5, min_occurrences: int = 3) -> (Dict[str, int], np.ndarray):
    """
    :param min_occurrences:
    :param context_size:
    :param path:
    :return: vocab and cooccurrence matrix
    """
    word_counts = Counter()
    cooccurrence = defaultdict(float)
    for sentence in load_data(path):
        word_counts.update(sentence)
        for l_ctx, word, r_ctx in generate_contexts(sentence, context_size):
            contexts = map(lambda t: (1 / (t[0] + 1), t[1]), list(enumerate(l_ctx)) + list(enumerate(r_ctx)))
            for coocur, context_word in contexts:
                cooccurrence[(word, context_word)] += coocur
    # filter rare words and index every word so we get id -> word
    df = pd.DataFrame(word_counts.items(), columns=['word', 'counts'])
    df = df[df['counts'] > min_occurrences]
    vocab_size = len(df)
    coocur_matrix = np.zeros((vocab_size, vocab_size))
    for key, coocur in cooccurrence.items():
        word, context_word = key
        context_word_id = df.index[df['word'] == context_word]
        word_id = df.index[df['word'] == word]
        if word_id.empty or context_word_id.empty:
            print("handle rare word {} or {}, skipping.".format(word, context_word))
            continue

        coocur_matrix[context_word_id[0], word_id[0]] += coocur

    return df['word'].to_dict(), coocur_matrix


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
                coocur_mat = pickle.load(f)
        else:
            raise EnvironmentError("wrong usage of method: when using overwrite=False, please make sure that "
                                   "vocab.pkl and coocur.pkl exists in path %s".format(save_path))
    else:
        vocab, coocur_mat = prepare_corpus(wiki_file_path, context_size, min_occurences)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(vocab_file, 'wb+') as f:
            pickle.dump(vocab, f)
        with open(coocur_file, 'wb+') as f:
            pickle.dump(coocur_mat, f)
    return vocab, coocur_mat


if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--wiki-txt-file', type=str, help='path to wiki file')
    par.add_argument('--save-path', type=str, help='where to store calculated coocurances')
    par.add_argument('--context-size', type=int, help='context size for coocurance calculation')
    par.add_argument('--min-occurences', type=int, help='min occurences of word pairs to occur in ds for training')
    args = par.parse_args(sys.argv)
    get_wiki_corpus_and_dump(
        args.wiki_txt_file,
        args.context_size,
        args.min_occurences,
        args.save_path,
        overwrite=True
    )
