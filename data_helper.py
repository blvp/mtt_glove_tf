import argparse
import os
import pickle
from collections import defaultdict, Counter
from time import clock
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess


def get_line_no(fp):
    p = subprocess.Popen(['wc', '-l', fp], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])


def load_data(path: str):
    with open(path, 'r') as f:
        for line in tqdm(f, total=get_line_no(path)):
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
    # filter rare words and index every word so we get id -> word
    st = clock()
    word_counts = Counter()
    for sentence in load_data(path):
        word_counts.update(sentence)
    df = pd.DataFrame(list(word_counts.items()), columns=['word', 'counts'])
    df = df[df['counts'] > min_count]
    df = df.reset_index(drop=True)
    vocab = dict([(w, idx) for idx, w in df['word'].to_dict().items()])
    vocab_size = len(vocab)
    print("vocab get time: ", clock() - st)
    print("vocab size", vocab_size)

    st = clock()
    cooccurrence = defaultdict(float)
    for sentence in load_data(path):
        for l_ctx, word, r_ctx in generate_contexts(sentence, context_size):
            contexts = map(lambda t: (1 / (t[0] + 1), t[1]), list(enumerate(l_ctx[::-1])) + list(enumerate(r_ctx)))
            word_id = vocab.get(word, None)
            for coocur, context_word in contexts:
                context_word_id = vocab.get(context_word, None)
                if word_id is not None and context_word_id is not None:
                    cooccurrence[(word_id, context_word_id)] += coocur
                    cooccurrence[(context_word_id, word_id)] += coocur
    print("coocurrence calc time: ", clock() - st)
    return vocab, [(k[0], k[1], v) for k, v in cooccurrence]


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
