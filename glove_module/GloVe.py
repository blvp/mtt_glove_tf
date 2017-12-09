from __future__ import division

import datetime
import sys
from random import shuffle
import tensorflow as tf
import data_helper
from glove_module import config

class GloveModule:
    def batchify(self, batch_size):
        rows = self.cooccurrence_matrix.shape[0]
        for i in range(0, rows, batch_size):
            yield self.cooccurrence_matrix[i: i + batch_size, :]

    def __init__(self):
        self.vocab, self.cooccurrence_matrix = data_helper.get_wiki_corpus_and_dump('../small_text8',
                                                                                    config.CONTEXT_SIZE,
                                                                                    config.MIN_OCCURRENCES,
                                                                                    overwrite=True)
        vocab_size = len(self.vocab)

        self.graph = tf.Graph()

        with self.graph.as_default():
            with self.graph.device("/cpu:0"):
                count_max = tf.constant([config.COUNT_MAX], dtype=tf.float32)  # tf константа -
                scaling_factor = tf.constant([config.SCALING_FACTOR], dtype=tf.float32)  # tf константа -
                self.focal_input = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE])  # tf переменная -
                self.context_input = tf.placeholder(tf.int32, shape=[config.BATCH_SIZE])  # tf переменная -
                self.cooccurrence_count = tf.placeholder(tf.float32, shape=[config.BATCH_SIZE])  # tf переменная -

                focal_embeddings = tf.Variable(
                    tf.random_uniform([vocab_size, config.EMBEDDING_SIZE], 1.0, -1.0)
                )

                context_embeddings = tf.Variable(
                    tf.random_uniform([vocab_size, config.EMBEDDING_SIZE], 1.0, -1.0)
                )

                focal_biases = tf.Variable(
                    tf.random_uniform([vocab_size], 1.0, -1.0)
                )

                context_biases = tf.Variable(
                    tf.random_uniform([vocab_size], 1.0, -1.0)
                )

                focal_embedding = tf.nn.embedding_lookup([focal_embeddings], self.focal_input)
                context_embedding = tf.nn.embedding_lookup([context_embeddings], self.context_input)
                focal_bias = tf.nn.embedding_lookup([focal_biases], self.focal_input)
                context_bias = tf.nn.embedding_lookup([context_biases], self.context_input)

                weighting_factor = tf.minimum(
                    1.0,
                    tf.pow(
                        tf.div(self.cooccurrence_count, count_max),
                        scaling_factor
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
                self.optimizer = tf.train.AdagradOptimizer(config.LEARNING_RATE).minimize(self.total_loss)

                self.combined_embeddings = tf.add(focal_embeddings, context_embeddings)

    def begin(self):
        # self.cooccurrences = [(pos[0], pos[1], count) for pos, count in self.cooccurrence_matrix]
        # i_indices, j_indices, counts = zip(*self.cooccurrences)

        self.batches = list(self.batchify(config.BATCH_SIZE))

        print("Begin training: {}".format(datetime.datetime.now().time()))
        print("=================")
        sys.stdout.flush()
        with tf.Session(graph=self.graph) as session:
            tf.initialize_all_variables().run()
            for epoch in range(config.NUM_EPOCHS):
                shuffle(self.batches)
                print("Batches shuffled")
                print("-----------------")
                sys.stdout.flush()
                accumulated_loss = 0
                for batch_index, batch in enumerate(self.batches):
                    i_s, j_s, counts = batch
                    if len(counts) != config.BATCH_SIZE:
                        continue
                    feed_dict = {self.focal_input: i_s, self.context_input: j_s, self.cooccurrence_count: counts}
                    _, total_loss_, = session.run([self.optimizer, self.total_loss], feed_dict=feed_dict)
                    accumulated_loss += total_loss_
                    if (batch_index + 1) % config.REPORT_BATCH_SIZE == 0:
                        print("Epoch: {0}/{1}".format(epoch + 1, config.NUM_EPOCHS))
                        print("Batch: {0}/{1}".format(batch_index + 1, len(self.batches)))
                        print("Average loss: {}".format(accumulated_loss / config.REPORT_BATCH_SIZE))
                        print("-----------------")
                        sys.stdout.flush()
                        accumulated_loss = 0
                # if (epoch + 1) % config.TSNE_EPOCH_FREQ == 0:
                #     print("Outputting t-SNE: {}".format(datetime.datetime.now().time()))
                #     print("-----------------")
                #     sys.stdout.flush()
                #     current_embeddings = self.combined_embeddings.eval()
                    # output_tsne(current_embeddings, "epoch{:02d}.png".format(epoch + 1))
                print("Epoch finished: {}".format(datetime.datetime.now().time()))
                print("=================")
                sys.stdout.flush()
        final_embeddings = self.combined_embeddings.eval()
        print("End: {}".format(datetime.datetime.now().time()))
        return final_embeddings



glove_instance = GloveModule()
glove_instance.begin()