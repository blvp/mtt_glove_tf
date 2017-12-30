import pickle
import json
import codecs
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict-pkl')
    parser.add_argument('--out-path')
    args = parser.parse_args()

    with open(args.dict_pkl, 'rb+') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    vector_size = max(map(len, vocab.values()))
    result = {
        'vocabularySize': vocab_size,
        'vectorSize': vector_size
    }
    vocabulary = {}
    for k, v in vocab.items():
        vocabulary[k] = v.tolist()
    result['vocabulary'] = vocabulary
    json.dump(
        result,
        codecs.open(args.out_path, 'w', encoding='utf-8'),
        separators=(',', ':'), sort_keys=True, indent=4
    )


