from typing import List, Dict, Tuple

from collections import defaultdict, Counter
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
