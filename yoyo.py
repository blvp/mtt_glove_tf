import pickle
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--id-to-vector-path')
    parser.add_argument('--vocab-path')
    parser.add_argument('--output-path', default='wordvectors.pkl')

    args = parser.parse_args()

    with open(args.id_to_vector_path, 'rb+') as f:
        id_to_vec = pickle.load(f)
    with open(args.vocab_path, 'rb+') as f:
        vocab = pickle.load(f)
    id_to_word = {v: k for k, v in vocab.items()}
    output = {id_to_word[k]: v for k, v in id_to_vec.items()}
    with open(args.output_path, 'wb+') as f:
        pickle.dump(output, f, protocol=4)
