import argparse
import pickle

from web.datasets.similarity import fetch_WS353
from web.embeddings import load_embedding
from web.evaluate import evaluate_similarity

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--word-vectors')
    args = parser.parse_args()

    ws353 = fetch_WS353()

    embedding = load_embedding(args.word_vectors, lower=True, clean_words=True, format='dict')
    print('Spearman`s rank on WS353 ', evaluate_similarity(embedding, ws353.X, ws353.y))



