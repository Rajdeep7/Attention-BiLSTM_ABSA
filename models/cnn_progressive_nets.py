import tensorflow as tf
from utils.util import initialize_embeddings
from utils.evaluation_util import compute_f1_score
from config.settings import BASE_CHECKPOINT_DIR, CHECKPOINT_DIR
from models import model_components
import tensorflow.contrib.layers as layers
import os
from config.operation_names import *
from tensorflow.python.keras.layers import TimeDistributed, Dense


def get_previous_model(session, args):
    # load existing model
    print('Loading previous meta graph')
    restore_dir = os.path.join(BASE_CHECKPOINT_DIR, args.previous_restore_dir)
    meta_graph_file = os.path.join(restore_dir, args.previous_meta_file_path)

    # update the saver from an existing meta graph
    saver = tf.train.import_meta_graph(meta_graph_file)
    if not saver:
        print('ERROR! Can not load meta graph from {}'.format(meta_graph_file))

    # fetch the modified default graph
    model = tf.get_default_graph()

    # print operations
    # for op in model.get_operations():
    #     print(op.name)
    #     print(op.values()[0])

    operations = {
        'global_step': model.get_operation_by_name(GLOBAL_STEP).outputs[0],
        'summary_op': model.get_operation_by_name(SUMMARY_OP).outputs[0],
        'loss': model.get_operation_by_name(LOSS).outputs[0],
        'accuracy': model.get_operation_by_name(ACCURACY).outputs[0],
        'train_op': model.get_operation_by_name(TRAIN_OP).outputs[0],
        'predictions': model.get_operation_by_name(PREDICTIONS).outputs[0],
        'f1_score': model.get_operation_by_name(F1_SCORE).outputs[0],
        # 'precision': model.get_operation_by_name(PRECISION).outputs[0],
        # 'recall': model.get_operation_by_name(RECALL).outputs[0],
        'padded_labels': model.get_operation_by_name(LABELS).outputs[0],
        'masked_predictions': model.get_operation_by_name(MASKED_PREDICTIONS).outputs[0],
        'confusion_matrix': model.get_operation_by_name(CONFUSION_MATRIX).outputs[0],
        'f1_score_0': model.get_operation_by_name(F1_SCORE_0).outputs[0],
        'f1_score_1': model.get_operation_by_name(F1_SCORE_1).outputs[0],
        'f1_score_2': model.get_operation_by_name(F1_SCORE_2).outputs[0],
        'f1_score_3': model.get_operation_by_name(F1_SCORE_3).outputs[0],
        'label_weights': model.get_operation_by_name(LABEL_WEIGHTS).outputs[0],
        'word_level_inputs': model.get_operation_by_name(WORD_LEVEL_INPUTS).outputs[0],
        'aspect_embedded_encoder_output': model.get_operation_by_name(ASPECT_EMBEDDED_ENCODER_OUTPUT).outputs[0],
        'aspect_embedded_sentence_inputs': model.get_operation_by_name(ASPECT_EMBEDDED_SENTENCE_INPUTS).outputs[0],
        'birnn_output': model.get_operation_by_name(SENTENCE_ENCODER_OUTPUT).outputs[0]
    }
    checkpoint = tf.train.get_checkpoint_state(restore_dir)
    if checkpoint:
        checkpoint.model_checkpoint_path = meta_graph_file.split('.')[0]
        print("Reading model parameters from {}".format(checkpoint.model_checkpoint_path))
        saver.restore(session, checkpoint.model_checkpoint_path)

    # Initialize local variables. It is specially needed for tf.metrics
    session.run(tf.local_variables_initializer())

    return operations


def get_model(session, args, restore_only = False):
    prefix = 'current_'
    if args.load_graph:
        # load existing model
        print('Loading meta graph')
        restore_dir = os.path.join(BASE_CHECKPOINT_DIR, args.restore_dir)
        meta_graph_file = os.path.join(restore_dir, args.meta_file_path)

        # update the saver from an existing meta graph
        saver = tf.train.import_meta_graph(meta_graph_file)
        if not saver:
            print('ERROR! Can not load meta graph from {}'.format(meta_graph_file))

        # fetch the modified default graph
        model = tf.get_default_graph()

        # print operations
        # for op in model.get_operations():
        #     print(op.name)
        #     print(op.values()[0])

        operations = {
            'global_step': model.get_operation_by_name(prefix + GLOBAL_STEP).outputs[0],
            'summary_op': model.get_operation_by_name(prefix + SUMMARY_OP).outputs[0],
            'loss': model.get_operation_by_name(prefix + LOSS).outputs[0],
            'accuracy': model.get_operation_by_name(prefix + ACCURACY).outputs[0],
            'train_op': model.get_operation_by_name(prefix + TRAIN_OP).outputs[0],
            'predictions': model.get_operation_by_name(prefix + PREDICTIONS).outputs[0],
            'f1_score': model.get_operation_by_name(prefix + F1_SCORE).outputs[0],
            # 'precision': model.get_operation_by_name(PRECISION).outputs[0],
            # 'recall': model.get_operation_by_name(RECALL).outputs[0],
            'padded_labels': model.get_operation_by_name(prefix + LABELS).outputs[0],
            'masked_predictions': model.get_operation_by_name(prefix + MASKED_PREDICTIONS).outputs[0],
            'confusion_matrix': model.get_operation_by_name(prefix + CONFUSION_MATRIX).outputs[0],
            'f1_score_0': model.get_operation_by_name(prefix + F1_SCORE_0).outputs[0],
            'f1_score_1': model.get_operation_by_name(prefix + F1_SCORE_1).outputs[0],
            'f1_score_2': model.get_operation_by_name(prefix + F1_SCORE_2).outputs[0],
            'f1_score_3': model.get_operation_by_name(prefix + F1_SCORE_3).outputs[0],
            'label_weights': model.get_operation_by_name(prefix + LABEL_WEIGHTS).outputs[0],
            'word_level_inputs': model.get_operation_by_name(prefix + WORD_LEVEL_INPUTS).outputs[0],
            'aspect_embedded_encoder_output':
                model.get_operation_by_name(prefix + ASPECT_EMBEDDED_ENCODER_OUTPUT).outputs[0],
            'aspect_embedded_sentence_inputs':
                model.get_operation_by_name(prefix + ASPECT_EMBEDDED_SENTENCE_INPUTS).outputs[0],
            'birnn_output': model.get_operation_by_name(prefix + SENTENCE_ENCODER_OUTPUT).outputs[0]
        }
        checkpoint = tf.train.get_checkpoint_state(restore_dir)
        if checkpoint:
            checkpoint.model_checkpoint_path = meta_graph_file.split('.')[0]
            print("Reading model parameters from {}".format(checkpoint.model_checkpoint_path))
            saver.restore(session, checkpoint.model_checkpoint_path)
        elif restore_only:
            raise FileNotFoundError("Cannot restore model")

        # Initialize local variables. It is specially needed for tf.metrics
        session.run(tf.local_variables_initializer())
    else:
        # create a new model object
        model = Model(
            word_cell = model_components.cell_maker(num_units = args.word_cell_units,
                                                    num_of_stacks = args.word_cell_stacks),
            sentence_cell = model_components.cell_maker(num_units = args.sentence_cell_units,
                                                        num_of_stacks = args.sentence_cell_stacks),
            args = args
        )
        operations = {
            'global_step': model.global_step,
            'summary_op': model.summary_op,
            'loss': model.loss,
            'accuracy': model.accuracy,
            'train_op': model.train_op,
            'predictions': model.predictions,
            'padded_labels': model.labels,
            'masked_predictions': model.masked_predictions,
            'f1_score': model.f1_score,
            'confusion_matrix': model.confusion_matrix,
            'f1_score_0': model.f1_score_0,
            'f1_score_1': model.f1_score_1,
            'f1_score_2': model.f1_score_2,
            'f1_score_3': model.f1_score_3,
            'label_weights': model.label_weights,
            'word_level_inputs': model.word_level_inputs,
            'aspect_embedded_encoder_output': model.aspect_embedded_encoder_output,
            'aspect_embedded_sentence_inputs': model.aspect_embedded_sentence_inputs,
            'birnn_output': model.sentence_encoder_output
        }

        # define saver for saving/restoring global variables from checkpoints
        saver = tf.train.Saver(tf.global_variables())

        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if checkpoint:
            print("Reading model parameters from {}".format(checkpoint.model_checkpoint_path))
            saver.restore(session, checkpoint.model_checkpoint_path)
        elif restore_only:
            raise FileNotFoundError("Cannot restore model")
        else:
            print("Created model with fresh parameters")
            # Initialize all global variables in a session
            session.run(tf.global_variables_initializer())

        # Initialize local variables. It is specially needed for tf.metrics
        session.run(tf.local_variables_initializer())
    return model, saver, operations


class Model:
    def __init__(self, args, word_cell, sentence_cell):
        self.args = args
        self.word_cell = word_cell
        self.sentence_cell = sentence_cell

        with tf.variable_scope('current_absa'):
            # get_variable gets an existing variable with these parameters or create a new one.
            self.global_step = tf.get_variable(name = 'global_step', initializer = 0, trainable = False)

            self.is_training = tf.placeholder(name = 'is_training', dtype = tf.bool)

            self.label_weights = tf.placeholder(name = 'label_weights', shape = (None, None), dtype = tf.float32)

            # padded input reviews with shape [batch_size, max_sentence_count_per_doc, max_word_count_per_sentence]
            # value at (i,j,k) indicates the word identifier for the kth word in the jth sentence of the ith review doc.
            self.reviews = tf.placeholder(name = 'padded_reviews', dtype = tf.int32, shape = (None, None, None))

            # input aspect words [batch_size, max_aspect_words_count]
            # value at (i,j) indicates the word identifier for the jth word in the ith review doc.
            self.aspect_words = tf.placeholder(name = 'aspect_words', dtype = tf.int32, shape = (None, None))

            # [batch_size, max_sentence_count_per_doc]
            # value at (i,j) indicates the actual number of words in jth sentence of ith doc
            self.actual_word_count = tf.placeholder(name = 'actual_word_count', shape = (None, None), dtype = tf.int32)

            # [batch_size] | value at (i) indicates the number of sentences in the ith doc
            self.actual_sentence_count = tf.placeholder(name = 'actual_sentence_count', shape = (None,),
                                                        dtype = tf.int32)

            # [batch_size, max_sentence_count_per_doc] | value at (i,j) indicates the sentiment polarity and
            # aspect applicability for the jth sentence in ith review doc.
            self.labels = tf.placeholder(name = 'padded_labels', shape = (None, None), dtype = tf.int32)

            self.batch_size, self.max_sentence_count, self.max_word_count = tf.unstack(tf.shape(self.reviews))

            # [batch_size, max_sentence_count] | value at (i,j) indicates the mask value of jth sentence in the ith
            # review doc
            self.sentence_mask = tf.placeholder(name = 'sentence_mask', shape = (None, None), dtype = tf.int32)

            # [batch_size, max_sentence_count, max_word_count_per_sentence, word_embedding_size]
            # value at (i, j, k, l) indicates the mask value of kth word in jth sentence in the ith review doc.
            # Each word is represented by a vector of size l.
            self.word_mask = tf.placeholder(name = 'word_mask', shape = (None, None, None, None), dtype = tf.float32)

            self.previous_word_level_inputs = tf.placeholder(name = 'previous_word_level_inputs',
                                                             shape = (None, None,
                                                                      self.args.word_embedding_size + self.args.aspect_embedding_size),
                                                             dtype = tf.float32)
            self.previous_aspect_embedded_encoder_output = tf.placeholder(
                name = 'previous_aspect_embedded_encoder_output',
                shape = (None, None, 2 * args.word_cell_units + args.aspect_embedding_size),
                dtype = tf.float32)
            self.previous_aspect_embedded_sentence_inputs = tf.placeholder(
                name = 'previous_aspect_embedded_sentence_inputs',
                shape = (None, None, args.word_output_size + args.aspect_embedding_size),
                dtype = tf.float32)
            self.previous_birnn_output = tf.placeholder(name = 'previous_birnn_output',
                                                        shape = (None, None, 2 * args.sentence_cell_units),
                                                        dtype = tf.float32)

            channels = self.args.word_embedding_size + self.args.aspect_embedding_size
            self.previous_word_level_inputs_filter = tf.get_variable(
                name = 'previous_word_level_inputs_filter',
                shape = (3, 3, channels, channels),
                initializer = layers.xavier_initializer(),
                trainable = True, dtype = tf.float32)

            channels = 2 * self.args.word_cell_units + self.args.aspect_embedding_size
            self.previous_aspect_embedded_encoder_output_filter = tf.get_variable(
                name = 'previous_aspect_embedded_encoder_output_filter',
                shape = (3, 3, channels, channels),
                initializer = layers.xavier_initializer(),
                trainable = True, dtype = tf.float32)

            channels = self.args.word_output_size + self.args.aspect_embedding_size
            self.previous_aspect_embedded_sentence_inputs_filter = tf.get_variable(
                name = 'previous_aspect_embedded_sentence_inputs_filter',
                shape = (3, 3, channels, channels),
                initializer = layers.xavier_initializer(),
                trainable = True, dtype = tf.float32)

            channels = 2 * self.args.sentence_cell_units
            self.previous_birnn_output_filter = tf.get_variable(
                name = 'previous_birnn_output_filter',
                shape = (3, 3, channels, channels),
                initializer = layers.xavier_initializer(),
                trainable = True, dtype = tf.float32)

            # These are Alpha scalars from the progressive nets paper
            self.previous_word_level_inputs_scalar = tf.get_variable(name = 'previous_word_level_inputs_scalar',
                                                                     dtype = tf.float32,
                                                                     shape = 1,
                                                                     trainable = True)

            self.previous_aspect_embedded_encoder_output_scalar = tf.get_variable(
                name = 'previous_aspect_embedded_encoder_output_scalar',
                dtype = tf.float32,
                shape = 1,
                trainable = True)

            self.previous_aspect_embedded_sentence_inputs_scalar = tf.get_variable(
                name = 'previous_aspect_embedded_sentence_inputs_scalar',
                dtype = tf.float32,
                shape = 1,
                trainable = True)

            self.previous_birnn_output_scalar = tf.get_variable(
                name = 'previous_birnn_output_scalar',
                dtype = tf.float32,
                shape = 1,
                trainable = True)

            self._init_word_embedding()
            self._init_model_body()
            self._train()

    def _train(self):
        with tf.variable_scope('train'):
            """
            lables = [batch_size, max_sentence_count]
            logits = [batch_size, max_sentence_count, 4]
            corss_entropy = [batch_size, max_sentence_count]
            """
            self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(name = 'cross_entropy',
                                                                                labels = self.labels,
                                                                                logits = self.logits)

            """
            compute regularizer
            """
            tvars = tf.trainable_variables(scope = 'current_absa')
            # print(len(tvars))

            """
            Here we are trying to calculate the average loss for the entire batch. Before doing this we try to reduce
            the class imbalance by multiplying class_weights with cross_entropy. Not sure how much helpful this is for
            reducing class imbalance.
            """
            self.loss = tf.reduce_mean(name = 'loss',
                                       input_tensor = tf.multiply(self.cross_entropy, self.label_weights,
                                                                  name = 'mul_label_weights'))
            tf.summary.scalar('train_loss', self.loss)

            """
            compute accuracy
            """
            _, self.accuracy = tf.metrics.accuracy(name = 'accuracy', labels = self.labels,
                                                   predictions = self.predictions,
                                                   weights = self.sentence_mask)
            tf.summary.scalar('train_accuracy', self.accuracy)

            """
            compute confusion matrix and f1 scores
            """
            flat_labels = tf.reshape(name = 'flat_labels', tensor = self.labels,
                                     shape = [self.batch_size * self.max_sentence_count])
            flat_predictions = tf.reshape(name = 'flat_predictions', tensor = self.predictions,
                                          shape = [self.batch_size * self.max_sentence_count])
            flat_label_weights = tf.reshape(name = 'flat_label_weights', tensor = self.sentence_mask,
                                            shape = [self.batch_size * self.max_sentence_count])
            self.confusion_matrix = tf.confusion_matrix(name = 'confusion_matrix', labels = flat_labels,
                                                        predictions = flat_predictions, weights = flat_label_weights,
                                                        num_classes = self.args.num_classes)

            self.f1_score_0 = compute_f1_score(self.confusion_matrix, class_id = 0, scope = 'f1_score_0')
            tf.summary.scalar('f1_score_0', self.f1_score_0)

            self.f1_score_1 = compute_f1_score(self.confusion_matrix, class_id = 1, scope = 'f1_score_1')
            tf.summary.scalar('f1_score_1', self.f1_score_1)

            self.f1_score_2 = compute_f1_score(self.confusion_matrix, class_id = 2, scope = 'f1_score_2')
            tf.summary.scalar('f1_score_2', self.f1_score_2)

            self.f1_score_3 = compute_f1_score(self.confusion_matrix, class_id = 3, scope = 'f1_score_3')
            tf.summary.scalar('f1_score_3', self.f1_score_3)

            self.f1_score = (self.f1_score_0 + self.f1_score_1 + self.f1_score_2 + self.f1_score_3) / 4
            tf.summary.scalar('f1_score', self.f1_score)

            """
            Get all the trainable variables. Define gradient of loss wrt to these trainable variables. Clip gradients.
            """
            grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.args.max_grad_norm)
            tf.summary.scalar('global_grad_norm', global_norm)

            """
            Apply gradients
            """
            opt = tf.train.AdamOptimizer(self.args.lr)
            self.train_op = opt.apply_gradients(name = 'train_op', grads_and_vars = zip(grads, tvars),
                                                global_step = self.global_step)

            """
            Merge all summaries
            """
            self.summary_op = tf.summary.merge_all()

    def _init_word_embedding(self):
        """
        Initialize an embedding_matrix from the scope and use this matrix to get input_embeddings for input word codes.
        Embedding matrix is learned by our model. We are not using off the shelf word embedding matrix. If we want we
        can simply initilize this matrix using some pre-trained word embeddings like GloVe.

        :param scope:
        :return:
        """
        # matrix non trainable
        with tf.variable_scope("embedding"):
            self.embedding_matrix = tf.get_variable(
                name = "embedding_matrix",
                # uncomment this if you intent to use xavier or other initializer
                # shape = [self.args.vocab_size, self.args.word_embedding_size],
                initializer = initialize_embeddings(self.args, etype = self.args.embedding_type),
                dtype = tf.float32,
                trainable = self.args.train_embeddings)
            # reviews_embedded = [batch_size, max_sentence_count, max_word_count, word_embedding_size]
            self.reviews_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.reviews)
            # aspect_embedded = [batch_size, max_aspect_words_count, aspect_embedding_size]
            self.aspect_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.aspect_words)
            # compressed_aspect_embedded = [batch_size, aspect_embedding_size]
            self.compressed_aspect_embedded = model_components.fuse_aspect_words(self.aspect_embedded,
                                                                                 self.args)

    def _init_model_body(self):
        """
        :param scope:
        :return:
        """
        # word_level
        with tf.variable_scope('word'):
            with tf.variable_scope('word_aspect_concatenate'):
                """
                aspect_embedded_reviews = [batch_size, max_sentence_count, max_word_count,
                word_embedding_size + aspect_embedding_size]. Attach aspect embedding vector to every word embedding
                vector.
                """
                repeated_aspect_embeddings = tf.tile(self.compressed_aspect_embedded,
                                                     [1, self.max_sentence_count * self.max_word_count])
                reshaped_aspect_embeddings = tf.reshape(repeated_aspect_embeddings,
                                                        [self.batch_size, self.max_sentence_count,
                                                         self.max_word_count,
                                                         self.args.aspect_embedding_size])

                """
                Masking out the aspect emebddings where actual words are padded words.
                """
                if self.args.use_word_mask:
                    reshaped_aspect_embeddings = reshaped_aspect_embeddings * self.word_mask

                aspect_embedded_reviews = tf.concat([self.reviews_embedded, reshaped_aspect_embeddings], -1)

            with tf.variable_scope('word_bilstm'):
                """
                word_level_inputs is a 3d(batch_size*max_sentence_count, max_word_count, word + aspect embedding size)
                array of word embeddings in a particular batch.
                An tf.nn.bidirectional_dynamic_rnn with time_major == False, expects input in this format
                [batch_size, max_time, ...]. Here time is modelled by number of words in a sentence i.e sentence length
                basically.
                """
                self.word_level_inputs = tf.reshape(aspect_embedded_reviews, shape = [
                    self.batch_size * self.max_sentence_count,
                    self.max_word_count,
                    self.args.word_embedding_size + self.args.aspect_embedding_size
                ], name = 'word_level_inputs')

                # TODO: lateral connection from previous layer of the previous models.
                self.previous_word_level_inputs = self.previous_word_level_inputs_scalar * self.previous_word_level_inputs
                self.previous_word_level_inputs = tf.expand_dims(self.previous_word_level_inputs, axis = 0)
                output = tf.nn.conv2d(input = self.previous_word_level_inputs,
                                      filter = self.previous_word_level_inputs_filter,
                                      strides = [1, 1, 1, 1],
                                      padding = 'SAME')
                output = tf.reduce_mean(output, axis = 0)
                self.word_level_inputs = self.word_level_inputs + output

                """
                word_lengths : [batch_size, max_sentence_count] | value at (i,j) indicates the number of words in
                jth sentence of ith doc. We reshape it to 1d array of size (batch_size*max_sentences_count).
                So every element in this array will represent original number of words (without padding) for the
                corresponding sentence. It is essentially the sentence length in terms of number of words in it.
                """
                word_level_lengths = tf.reshape(self.actual_word_count, [self.batch_size * self.max_sentence_count])

                """
                Here we are applying bi-LSTM on sequence of words. We pass an LSTM cell implementation, along with input
                data and sequence length. We get LSTM encoded word embeddings as output. Encoder output for padded words
                and sentences should be zero.
                https://www.tensorflow.org/api_docs/python/tf/nn/bidirectional_dynamic_rnn

                word_encoder_output = [batch_size*max_sentence_count, max_word_count, 2*num_units]
                """
                word_encoder_output, _ = model_components.bidirectional_rnn(self.word_cell, self.word_cell,
                                                                            self.word_level_inputs, word_level_lengths)

            with tf.variable_scope('word_attention'):
                if self.args.use_attention:
                    """
                    Concatenate aspect emebddings with word level BiLSTM hidden outputs
                    """
                    reshaped_aspect_embeddings = tf.reshape(reshaped_aspect_embeddings,
                                                            [self.batch_size * self.max_sentence_count,
                                                             self.max_word_count,
                                                             self.args.aspect_embedding_size])
                    self.aspect_embedded_encoder_output = tf.concat([word_encoder_output, reshaped_aspect_embeddings],
                                                                    -1,
                                                                    name = 'aspect_embedded_encoder_output')

                    # TODO: lateral connection from previous layer of the previous models.
                    self.previous_aspect_embedded_encoder_output = self.previous_aspect_embedded_encoder_output_scalar * self.previous_aspect_embedded_encoder_output
                    self.previous_aspect_embedded_encoder_output = tf.expand_dims(
                        self.previous_aspect_embedded_encoder_output, axis = 0)
                    output = tf.nn.conv2d(input = self.previous_aspect_embedded_encoder_output,
                                          filter = self.previous_aspect_embedded_encoder_output_filter,
                                          strides = [1, 1, 1, 1],
                                          padding = 'SAME')
                    output = tf.reduce_mean(output, axis = 0)
                    self.aspect_embedded_encoder_output = self.aspect_embedded_encoder_output + output

                    """
                    After applying word level attention we will get an initial sentence embedding
                    word_level_output = [batch_size*max_sentence_count, word_output_size]
                    """
                    word_level_output = model_components.task_specific_attention(self.aspect_embedded_encoder_output,
                                                                                 self.args.word_output_size,
                                                                                 self.args)
                else:
                    """
                    word_level_output = [batch_size*max_sentence_count, 2*num_units]
                    """
                    word_level_output = tf.reduce_mean(word_encoder_output, axis = 1)

                """
                This is a hack to handle the channing out size when toggeling between attention use or when not using
                combined input in the attention layer.
                """
                self.args.word_output_size = word_level_output.get_shape()[1]

                with tf.variable_scope('word_dropout'):
                    """
                    Here we are applying dropout to the word_level_output i.e initial sentence embeddings. But Why?
                    """
                    word_level_output = layers.dropout(word_level_output, keep_prob = self.args.dropout_keep_prob,
                                                       is_training = self.is_training)

        # sentence_level
        with tf.variable_scope('sentence'):
            """
            We will reshape word_level_output so that sentences belonging to a particular doc is considered as one data
            point. Earlier with word level processing every sentence was considered as one data point.
            sentence_inputs = [batch_size, max_sentence_count, word_output_size]
            """
            sentence_inputs = tf.reshape(word_level_output,
                                         shape = [self.batch_size, self.max_sentence_count, self.args.word_output_size])

            with tf.variable_scope('sentence_aspect_concatenate'):
                """
                concatenate aspect embeddings with sentence embeddings
                """
                repeated_aspect_embeddings = tf.tile(self.compressed_aspect_embedded,
                                                     [1, self.max_sentence_count])
                reshaped_aspect_embeddings = tf.reshape(repeated_aspect_embeddings,
                                                        [self.batch_size, self.max_sentence_count,
                                                         self.args.aspect_embedding_size])
                self.aspect_embedded_sentence_inputs = tf.concat([sentence_inputs, reshaped_aspect_embeddings], -1,
                                                                 name = 'aspect_embedded_sentence_inputs')

                # TODO: lateral connection from previous layer of the previous models.
                self.previous_aspect_embedded_sentence_inputs = self.previous_aspect_embedded_sentence_inputs_scalar * self.previous_aspect_embedded_sentence_inputs
                self.previous_aspect_embedded_sentence_inputs = tf.expand_dims(
                    self.previous_aspect_embedded_sentence_inputs, axis = 0)
                output = tf.nn.conv2d(input = self.previous_aspect_embedded_sentence_inputs,
                                      filter = self.previous_aspect_embedded_sentence_inputs_filter,
                                      strides = [1, 1, 1, 1],
                                      padding = 'SAME')
                output = tf.reduce_mean(output, axis = 0)
                self.aspect_embedded_sentence_inputs = self.aspect_embedded_sentence_inputs + output

            with tf.variable_scope('sentence_bilstm'):
                """
                sentence_encoder_output = [batch_size, max_sentence_count, 2*num_units]
                """
                self.sentence_encoder_output, _ = model_components.bidirectional_rnn(self.sentence_cell,
                                                                                     self.sentence_cell,
                                                                                     self.aspect_embedded_sentence_inputs,
                                                                                     self.actual_sentence_count)
                # TODO: lateral connection from previous layer of the previous models.
                self.previous_birnn_output = self.previous_birnn_output_scalar * self.previous_birnn_output
                self.previous_birnn_output = tf.expand_dims(self.previous_birnn_output, axis = 0)
                output = tf.nn.conv2d(input = self.previous_birnn_output,
                                      filter = self.previous_birnn_output_filter,
                                      strides = [1, 1, 1, 1],
                                      padding = 'SAME')
                output = tf.reduce_mean(output, axis = 0)
                self.sentence_encoder_output = self.sentence_encoder_output + output

        # classifier
        with tf.variable_scope('classifier'):
            """
            logits = [batch_size, max_sentence_count, 4]
            """
            self.logits = layers.fully_connected(self.sentence_encoder_output, self.args.num_classes,
                                                 activation_fn = None)

            """
            [batch_size, max_sentence_count]
            """
            self.predictions = tf.argmax(name = 'pred_argmax', input = self.logits, axis = 2,
                                         output_type = tf.int32)

            """
            [batch_size, max_sentence_count], maskout predictions of padded sentences
            """
            self.masked_predictions = tf.multiply(name = 'masked_predictions', x = self.sentence_mask,
                                                  y = self.predictions)
