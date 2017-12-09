import codecs
import email
import glob
import re

import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import os
import tflearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



def cleaned(content):
    # First remove inline JavaScript/CSS:
    cleaned_content = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", content)
    # Then remove html comments.
    cleaned_content = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned_content)
    # Next remove the remaining tags:
    cleaned_content = re.sub(r"(?s)<.*?>", " ", cleaned_content)
    # Finally deal with whitespace
    cleaned_content = re.sub(r"&nbsp;", " ", cleaned_content)
    cleaned_content = re.sub(r"^$", "", cleaned_content)
    cleaned_content = re.sub("''|,", "", cleaned_content)
    cleaned_content = re.sub(r"  ", " ", cleaned_content)
    cleaned_content = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", cleaned_content)
    cleaned_content = re.sub(r"\'s", " \'s", cleaned_content)
    cleaned_content = re.sub(r"\'ve", " \'ve", cleaned_content)
    cleaned_content = re.sub(r"n\'t", " n\'t", cleaned_content)
    cleaned_content = re.sub(r"\'re", " \'re", cleaned_content)
    cleaned_content = re.sub(r"\'d", " \'d", cleaned_content)
    cleaned_content = re.sub(r"\'ll", " \'ll", cleaned_content)
    cleaned_content = re.sub(r",", " , ", cleaned_content)
    cleaned_content = re.sub(r"!", " ! ", cleaned_content)
    cleaned_content = re.sub(r"\(", " \( ", cleaned_content)
    cleaned_content = re.sub(r"\)", " \) ", cleaned_content)
    cleaned_content = re.sub(r"\?", " \? ", cleaned_content)
    cleaned_content = re.sub(r"\s{2,}", " ", cleaned_content)
    cleaned_content = re.sub(r"\d+", "", cleaned_content)
    cleaned_content = re.sub(r"[\r\n]+", " ", cleaned_content)
    return cleaned_content.strip().lower()


def proccess_message(text):
    msg = email.message_from_string(text)
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append(part.get_payload())
    content = ''.join(parts)
    return cleaned(content)


def load_from_path(path):
    data = []

    def proccess_dir(glob_re_path, label):
        res = []
        for file_path in glob.glob(glob_re_path):
            with codecs.open(file_path, "r", encoding='utf-8', errors='ignore') as f:
                res.append([label, proccess_message(f.read())])
        return res

    data += proccess_dir(path + '/ham/*.txt', 0)
    data += proccess_dir(path + '/spam/*.txt', 1)
    return data


def space_tokenizer(w):
    return w.split(' ')


class BagOfWords(object):
    def __init__(self,
                 tokenize=space_tokenizer,
                 normalize=True,
                 remove_stop_words=True):
        self._vocab = dict(UNK=0)
        self.normalize = normalize
        self._tokenize = tokenize
        self._stemmer = SnowballStemmer('english')
        self.remove_stop_words = remove_stop_words

    def fit(self, X, y=None):
        if self.remove_stop_words:
            stops = stopwords.words('english')
            words = [word for d in X for word in self._tokenize(d) if word not in stops]
        else:
            words = [word for d in X for word in self._tokenize(d)]

        words = set([self._stemmer.stem(w) for w in set(words)])

        self._vocab.update(
            dict([(v, k + 1) for k, v in enumerate(words)]))

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def transform(self, X):
        data = []
        for d in X:
            res = np.zeros(len(self._vocab), dtype=np.float32)
            for word in self._tokenize(d):
                res[self.__get_word_index(self._stemmer.stem(word))] += 1
            if self.normalize:
                res = np.log(res + 1)

            data.append(res)
        return np.array(data)

    def __get_word_index(self, word):
        return self._vocab.get(word, 0)

    def __len__(self):
        return len(self._vocab)


def get_network(incoming_shape):
    input_x = tflearn.input_data([None, incoming_shape])
    net = tflearn.fully_connected(input_x, 1024, activation='relu')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 1024, activation='relu')
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy', name='target')
    return net


if __name__ == '__main__':
    print('Loading dataset')
    data = load_from_path('./data/emails/raw')
    df = pd.DataFrame(data, columns=['label', 'content'])
    df = shuffle(df)
    labels_encoded = pd.get_dummies(df['label'], columns=['ham', 'spam'])
    print('Split dataset to train/test')
    X_train, X_test, y_train, y_test = train_test_split(df['content'], labels_encoded, test_size=0.3)
    print('Train size: ', len(X_train))
    print('Test size: ', len(X_test))
    bow = BagOfWords(remove_stop_words=True)
    print('Fitting BoW')
    X_train_encoded = bow.fit_transform(X_train)
    X_test_encoded = bow.transform(X_test)
    vocab_size = len(bow)
    print('Vocabulary size: ', len(bow))
    print('Network initialization')
    network = get_network(vocab_size)
    model = tflearn.DNN(
        network,
        tensorboard_verbose=0,
        tensorboard_dir='./logs/bow_mlp/'
    )
    model.fit(
        X_train_encoded,
        y_train,
        show_metric=True,
        n_epoch=25,
        validation_set=(X_test_encoded, y_test),
        shuffle=True,
        batch_size=1024
    )
