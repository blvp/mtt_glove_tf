from __future__ import division

import datetime
import pickle
import sys
from random import shuffle
import tensorflow as tf
import data_helper
from glove_module import config


class GloveModule:
    def batchify(self, *matrix):
        for i in range(0, len(matrix[0]), config.BATCH_SIZE):
            yield tuple(sequence[i:i + config.BATCH_SIZE] for sequence in matrix)

    def __init__(self):
        print("Load vocab and cooc matrix...")
        self.vocab, cooccurrence_matrix = \
            data_helper.get_wiki_corpus_and_dump(
                config.DATASET_FILE,
                config.CONTEXT_SIZE,
                config.MIN_OCCURRENCES,
                save_path='./data/wiki_prepared_250m_2/',
                overwrite=False
            )
        self.cooccurrence_matrix = list(cooccurrence_matrix.items())
        vocab_size = len(self.vocab)
        print("Done loading vocab and cooc.")
        print("Creating tensorflow graph...")
        self.graph = tf.Graph()

        with self.graph.as_default():
            with self.graph.device("/cpu:0"):
                count_max = tf.constant([config.COUNT_MAX], dtype=tf.float32)  # tf константа -
                alpha = tf.constant([config.SCALING_FACTOR], dtype=tf.float32)  # tf константа -
                self.focal_input = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE])  # tf переменная -
                self.context_input = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE])  # tf переменная -
                self.cooccurrence_count = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE])  # tf переменная -
                self.embeddings = tf.Variable(
                    tf.random_uniform([vocab_size, config.EMBEDDING_SIZE], 1.0, -1.0)
                )
                # focal_embeddings = tf.Variable(
                #     tf.random_uniform([vocab_size, config.EMBEDDING_SIZE], 1.0, -1.0)
                # )

                # context_embeddings = tf.Variable(
                #     tf.random_uniform([vocab_size, config.EMBEDDING_SIZE], 1.0, -1.0)
                # )

                focal_biases = tf.Variable(
                    tf.random_uniform([vocab_size], 1.0, -1.0)
                )

                context_biases = tf.Variable(
                    tf.random_uniform([vocab_size], 1.0, -1.0)
                )

                focal_embedding = tf.nn.embedding_lookup([self.embeddings], self.focal_input)
                context_embedding = tf.nn.embedding_lookup([self.embeddings], self.context_input)
                focal_bias = tf.nn.embedding_lookup([focal_biases], self.focal_input)
                context_bias = tf.nn.embedding_lookup([context_biases], self.context_input)

                weighting_factor = tf.minimum(
                    1.0,
                    tf.pow(
                        tf.div(self.cooccurrence_count, count_max),
                        alpha
                    )
                )

                embedding_product = tf.reduce_sum(tf.multiply(focal_embedding, context_embedding), 1)

                log_cooccurrences = tf.log(tf.to_float(self.cooccurrence_count))

                distance_expr = tf.square(tf.add_n([
                    embedding_product,
                    focal_bias,
                    context_bias,
                    tf.negative(log_cooccurrences)
                ]))

                single_losses = tf.multiply(weighting_factor, distance_expr)
                self.total_loss = tf.reduce_sum(single_losses)
                # with self.graph.device("/cpu:0"):
                tf.summary.scalar('loss', self.total_loss)
                # self.optimizer = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.total_loss)
                self.optimizer = tf.train.AdagradOptimizer(config.LEARNING_RATE).minimize(self.total_loss)
        print("Done creating")

    def begin(self):
        print("Prepare cooccurrence matrix...")

        #i_indices, j_indices, counts = zip(*self.cooccurrence_matrix)
        print("Done preparing")
        print("Get batches...")
        #self.batches = list(self.batchify(i_indices, j_indices, cooc))
        print("Done get batches.")
        print("Begin training: {}".format(datetime.datetime.now().time()))
        print("=================")
        sys.stdout.flush()
        with tf.Session(graph=self.graph) as session:
            merged = tf.summary.merge_all()
            batch_writer = tf.summary.FileWriter('./logs/batch')
            tf.initialize_all_variables().run()
            for epoch in range(config.NUM_EPOCHS):
                #shuffle(self.batches)
                print("Batches shuffled")
                print("-----------------")
                sys.stdout.flush()
                accumulated_loss = 0
                total = len(self.cooccurrence_matrix)
                num_batches = total/config.BATCH_SIZE
                batch_index = 0
                for i in range(0, total, config.BATCH_SIZE):
                    batch_index += 1
                    ww, counts = zip(*self.cooccurrence_matrix[i:i+config.BATCH_SIZE])
                    i_s, j_s = zip(*ww)
                    #i_s, j_s, counts = batch
                    if len(counts) != config.BATCH_SIZE:
                        continue
                    feed_dict = {self.focal_input: i_s, self.context_input: j_s, self.cooccurrence_count: counts}
                    summaries, _, total_loss_, = session.run(
                        [merged, self.optimizer, self.total_loss], feed_dict=feed_dict)
                    accumulated_loss += total_loss_
                    if (batch_index + 1) % config.REPORT_BATCH_SIZE == 0:
                        print("Epoch: {0}/{1}".format(epoch + 1, config.NUM_EPOCHS))
                        print("Batch: {0}/{1}".format(batch_index + 1, num_batches))
                        print("Average loss: {}".format(accumulated_loss / config.REPORT_BATCH_SIZE))
                        print("-----------------")
                        batch_writer.add_summary(summaries, epoch * num_batches + batch_index)
                        sys.stdout.flush()
                        accumulated_loss = 0
                print("Epoch finished: {}".format(datetime.datetime.now().time()))
                print("=================")

                sys.stdout.flush()
            batch_writer.close()
            final_embeddings = self.embeddings.eval()
            print("End: {}".format(datetime.datetime.now().time()))
            final_dict = {}
            for word, idx in self.vocab.items():
                final_dict[word] = final_embeddings[idx, :]
            return final_dict


glove_instance = GloveModule()
embeddings = glove_instance.begin()

with open('data/emb/glove{}.pkl'.format(config.EMBEDDING_SIZE), 'wb+') as f:
    pickle.dump(embeddings, f, protocol=4)
