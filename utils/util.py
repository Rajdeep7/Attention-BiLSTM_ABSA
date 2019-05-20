import numpy as np
import pandas as pd
import os
import tensorflow as tf
from collections import Counter
import tensorflow.contrib.layers as layers
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import h5py
from config.settings import *
from utils.data_util import read_binary, train_loader
from config.tensor_names import *
import fastText


def load_elmo_emebddings():
    h5py_file = h5py.File(ELMO_EMBEDDINGS_FILE, 'r')
    return h5py_file


def load_glove_embeddings():
    print('loading glove..')
    glove_file = datapath(GLOVE_EMBEDDINGS_FILE)
    tmp_file = get_tmpfile(TMP_GLOVE_EMBEDDINGS_FILE)
    glove2word2vec(glove_file, tmp_file)
    glove = KeyedVectors.load_word2vec_format(tmp_file)
    return glove


def load_fastText_embeddings(lang = 'en'):
    print('loading fasttext..')
    if lang == 'de':
        fasttext = KeyedVectors.load_word2vec_format(FASTTEXT_DE_EMBEDDINGS_FILE)
    else:
        fasttext = KeyedVectors.load_word2vec_format(FASTTEXT_EN_EMBEDDINGS_FILE)
    return fasttext


def load_oov_fastText_embeddings(lang = 'en', self_trained = False):
    print('loading ovv fastext model..')
    if lang == 'de':
        if self_trained:
            model_path = SELF_TRAINED_FASTTEXT_DE_EMBEDDINGS_MODEL
        else:
            model_path = FASTTEXT_DE_EMBEDDINGS_MODEL
        model = fastText.load_model(model_path)
    else:
        model = fastText.load_model(FASTTEXT_EN_EMBEDDINGS_MODEL)
    return model


def glove_embeddings(shape):
    print('using glove...')
    glove = read_binary(CODE_TO_EMBED_FILE)
    glove = glove[0:shape[0], 0:shape[1]]
    return tf.convert_to_tensor(value = glove)


def fasttext_embeddings(shape):
    print('using fasttext..')
    fasttext = read_binary(CODE_TO_EMBED_FILE)
    fasttext = fasttext[0:shape[0], 0:shape[1]]
    return tf.convert_to_tensor(value = fasttext)


def initialize_embeddings(args, etype = 'xavier'):
    if etype == 'xavier':
        return layers.xavier_initializer()
    elif etype == 'glove':
        return glove_embeddings(shape = (args.vocab_size, args.word_embedding_size))
    elif etype == 'fasttext':
        return fasttext_embeddings(shape = (args.vocab_size, args.word_embedding_size))


def batch_iterator(dataset, batch_size, max_epochs = MAX_POSSIBLE_EPOCHS):
    """
    A generator function that yields batched data
    x: One review document consisting of many sentences. Each sentence consists of
       many words and each word is represented as an integer code from the vocabulary.
    xb: Batch of review documents.

    :param dataset: Its is generator function that yields <review, stars> tuple
    :param batch_size:
    :param max_epochs: It controls maximum number of times yield can be called in this function.
    :return:
    """
    batched_aspects = []
    batched_reviews = []
    batched_lables = []
    sentence_trimmer = 50
    for i in range(max_epochs):
        for x1, x2, y in dataset:
            batched_aspects.append(x1)
            batched_reviews.append(x2[0:sentence_trimmer])
            batched_lables.append(y[0:sentence_trimmer])
            if len(batched_lables) == batch_size:
                yield batched_aspects, batched_reviews, batched_lables
                batched_aspects, batched_reviews, batched_lables = [], [], []
        # leftout handling from the current epoch
        if len(batched_aspects) != 0 and len(batched_reviews) != 0 and len(batched_lables) != 0:
            yield batched_aspects, batched_reviews, batched_lables
            batched_aspects, batched_reviews, batched_lables = [], [], []


def padded_batch(x1, x2, y, class_weights, args):
    """
    This method pads the batched input such that every doc and every sentence has uniform length.
    b: padded batched input
    document_sizes: 1d array (bacth_size) of number of sentences in every doc. Sentence count of every review doc.
    sentence_sizes_: 2d array (batch_size, document_sizes) of number of words per sentence in every doc.

    :param inputs: Input data batch
    :return:
    """
    # num of documents/reviews/datapoints in a batch
    batch_size = len(x2)

    # calculate number of sentences per document. This will be used by the dynamic rnn
    sentence_count_per_doc = np.array([len(doc) for doc in x2], dtype = np.int32)
    max_num_of_sentences_per_doc = sentence_count_per_doc.max()

    # calculate number of words per sentecne per document. This will also be used by dynamic rnn
    word_count_per_sentence_per_doc = [[len(sent) for sent in doc] for doc in x2]

    # map() function returns a list of the results after applying the given function to
    # each item of a given iterable (list, tuple etc.)
    max_num_of_words_per_sentence = max(map(max, word_count_per_sentence_per_doc))

    # calculate number of aspect words per review doc
    aspect_word_count_per_doc = np.array([len(aspect_words) for aspect_words in x1], dtype = np.int32)
    max_aspect_word_count = aspect_word_count_per_doc.max()

    # padded batch
    padded_aspects = np.zeros(shape = [batch_size, max_aspect_word_count], dtype = np.int32)
    padded_reviews = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc, max_num_of_words_per_sentence],
                              dtype = np.int32)

    # TODO: try to pad label with a different value and see how the predictions change
    padded_labels = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc], dtype = np.int32)

    # Used to differentiate between actual and padded sentences
    sentence_mask = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc], dtype = np.int32)

    # Used to differentiate between actual and padded words
    word_mask = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc, max_num_of_words_per_sentence],
                         dtype = np.float32)

    # It only gets set for the true labels. Padded labels get a weight of 0
    label_weights = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc], dtype = np.float32)

    # This will have zero word count for padded sentences
    word_count_per_sentence_per_doc_with_padding = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc],
                                                            dtype = np.int32)
    for i, document in enumerate(x2):
        for w in range(len(x1[i])):
            padded_aspects[i, w] = x1[i][w]
        for j, sentence in enumerate(document):
            word_count_per_sentence_per_doc_with_padding[i, j] = word_count_per_sentence_per_doc[i][j]
            # print(y[i][j])
            padded_labels[i, j] = y[i][j]
            sentence_mask[i, j] = 1
            if class_weights:
                label_weights[i, j] = class_weights[padded_labels[i, j]]
            for k, word in enumerate(sentence):
                padded_reviews[i, j, k] = word
                word_mask[i, j, k] = 1

    repeated_word_mask = np.repeat(word_mask, args.word_embedding_size, axis = -1)
    repeated_word_mask = np.reshape(repeated_word_mask, newshape = (
        batch_size, max_num_of_sentences_per_doc, max_num_of_words_per_sentence, args.aspect_embedding_size))

    return {
        'padded_aspects': padded_aspects,
        'padded_reviews': padded_reviews,
        'padded_labels': padded_labels,
        'actual_sentence_count': sentence_count_per_doc,
        'actual_word_count': word_count_per_sentence_per_doc_with_padding,
        'sentence_mask': sentence_mask,
        'label_weights': label_weights,
        'word_mask': repeated_word_mask
    }


def elmo_padded_batch(x1, x2, y, class_weights, args, elmo):
    """
    This method pads the batched input such that every doc and every sentence has uniform length.
    b: padded batched input
    document_sizes: 1d array (bacth_size) of number of sentences in every doc. Sentence count of every review doc.
    sentence_sizes_: 2d array (batch_size, document_sizes) of number of words per sentence in every doc.

    :param inputs: Input data batch
    :return:
    """
    # num of documents/reviews/datapoints in a batch
    batch_size = len(x2)

    # calculate number of sentences per document. This will be used by the dynamic rnn
    sentence_count_per_doc = np.array([len(doc) for doc in x2], dtype = np.int32)
    max_num_of_sentences_per_doc = sentence_count_per_doc.max()

    # calculate number of words per sentecne per document. This will also be used by dynamic rnn
    word_count_per_sentence_per_doc = [[len(sent) for sent in doc] for doc in x2]

    # map() function returns a list of the results after applying the given function to
    # each item of a given iterable (list, tuple etc.)
    max_num_of_words_per_sentence = max(map(max, word_count_per_sentence_per_doc))

    # calculate number of aspect words per review doc
    aspect_word_count_per_doc = np.array([len(aspect_words) for aspect_words in x1], dtype = np.int32)
    max_aspect_word_count = aspect_word_count_per_doc.max()

    # padded batch
    padded_aspects = np.zeros(shape = [batch_size, max_aspect_word_count, args.aspect_embedding_size],
                              dtype = np.float32)
    padded_reviews = np.zeros(
        shape = [batch_size, max_num_of_sentences_per_doc, max_num_of_words_per_sentence, args.word_embedding_size],
        dtype = np.float32)

    # TODO: try to pad label with a different value and see how the predictions change
    padded_labels = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc], dtype = np.int32)

    # Used to differentiate between actual and padded sentences
    sentence_mask = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc], dtype = np.int32)

    # Used to differentiate between actual and padded words
    word_mask = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc, max_num_of_words_per_sentence],
                         dtype = np.float32)

    # It only gets set for the true labels. Padded labels get a weight of 0
    label_weights = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc], dtype = np.float32)

    # This will have zero word count for padded sentences
    word_count_per_sentence_per_doc_with_padding = np.zeros(shape = [batch_size, max_num_of_sentences_per_doc],
                                                            dtype = np.int32)
    for i, document in enumerate(x2):
        for w in range(len(x1[i])):
            padded_aspects[i, w, :] = elmo.get(x1[i][w])
        for j, sentence in enumerate(document):
            word_count_per_sentence_per_doc_with_padding[i, j] = word_count_per_sentence_per_doc[i][j]
            padded_labels[i, j] = y[i][j]
            sentence_mask[i, j] = 1
            sent = ' '.join(sentence)
            # print(sent)
            embedded_sentence = elmo.get(sent)
            # print(embedded_sentence.shape)
            if class_weights:
                label_weights[i, j] = class_weights[padded_labels[i, j]]
            for k, word in enumerate(sentence):
                padded_reviews[i, j, k, :] = embedded_sentence[k]
                word_mask[i, j, k] = 1

    repeated_word_mask = np.repeat(word_mask, args.word_embedding_size, axis = -1)
    repeated_word_mask = np.reshape(repeated_word_mask, newshape = (
        batch_size, max_num_of_sentences_per_doc, max_num_of_words_per_sentence, args.aspect_embedding_size))

    return {
        'padded_aspects': padded_aspects,
        'padded_reviews': padded_reviews,
        'padded_labels': padded_labels,
        'actual_sentence_count': sentence_count_per_doc,
        'actual_word_count': word_count_per_sentence_per_doc_with_padding,
        'sentence_mask': sentence_mask,
        'label_weights': label_weights,
        'word_mask': repeated_word_mask
    }


def code_to_vocab(codes):
    code_to_vocab_map = read_binary(CODE_TO_VOCAB_FILE)
    words = []
    for code in codes:
        words.append(code_to_vocab_map.get(code))
    return words


def vocab_to_code(words):
    vocab_to_code_map = read_binary(VOCAB_TO_CODE_FILE)
    codes = []
    for word in words:
        codes.append(vocab_to_code_map.get(word))
    return codes


def calculate_class_weights(classes = None):
    """
    Counter - generates counts for keys, in this case star ratings are the keys.
    Series - Creates a series object from dict in this case. Keys of the dict i.e star/rating value
             are the index of the series and value of the series are the freq of corresponding ratings.
             This is calculated over the entire trainset. Then, we calculate probability distribution
             for each possible rating stars and inverse it. This will help us to analyse class imbalance becase classes
             in our case are have different possible ratings.
    :return:
    """
    # TODO: handle weights for label paddings
    label_values = []
    if classes is None:
        for _, _, labels in train_loader():
            for label in labels:
                label_values.append(label)
    else:
        for labels in classes:
            for label in labels:
                label_values.append(label)

    class_weights = pd.Series(Counter(label_values))
    # Inverse of probability distribution of weights
    class_weights = 1 / (class_weights / class_weights.mean())
    class_weights = class_weights.to_dict()
    return class_weights


def get_feed_data(x1, x2, y = None, class_weights = None, is_training = True, args = None, elmo = None):
    """
    This method formats the data to be directly used by session. It also calls the
    batch padder which pads review docs such that they are of equal lengths.

    :param x: Batched input
    :param y: Batched true lables/classes
    :param class_weights: Distribution of classes on the entire dataset
    :param is_training:
    :return:
    """

    # coz elmo processed data differently. It adds embeddings right away.
    if args.embedding_type == 'elmo':
        padded_output = elmo_padded_batch(x1, x2, y, class_weights, args, elmo)
    else:
        padded_output = padded_batch(x1, x2, y, class_weights, args)

    fd = {
        ASPECTS_TENSOR_NAME: padded_output['padded_aspects'],
        PADDED_REVIEWS_TENSOR_NAME: padded_output['padded_reviews'],
        ACTUAL_SENTENCE_COUNT_TENSOR_NAME: padded_output['actual_sentence_count'],
        ACTUAL_WORD_COUNT_TENSOR_NAME: padded_output['actual_word_count'],
        SENTENCE_MASK_TENSOR_NAME: padded_output['sentence_mask'],
        WORD_MASK_TENSOR_NAME: padded_output['word_mask']
    }
    if y is not None:
        fd[PADDED_LABELS_TENSOR_NAME] = padded_output['padded_labels']
        if class_weights is not None:
            fd[LABLE_WEIGHTS_TENSOR_NAME] = padded_output['label_weights']
        else:
            fd[LABLE_WEIGHTS_TENSOR_NAME] = np.ones_like(padded_output['padded_labels'], dtype = np.float32)
    fd[IS_TRAINING_TENSOR_NAME] = is_training
    return fd


def space_separated_token_string(tokenized_strings):
    space_separated_strings = []
    for tokens in tokenized_strings:
        space_separated_strings.append(' '.join(tokens))
    space_separated_strings = set(space_separated_strings)
    return space_separated_strings


def save_sentences_to_text(all_sentences):
    with open(ALL_SENTENCES_TEXT, 'a') as file:
        for sentence in all_sentences:
            print(sentence, file = file)


def remove_duplicate_sentences():
    unique_sentences = set()
    with open(ALL_SENTENCES_TEXT, 'r') as file:
        for line in file:
            unique_sentences.add(line.strip('\n'))

    with open(ALL_SENTENCES_TEXT, 'w') as file:
        for sentence in unique_sentences:
            print(sentence, file = file)


def get_similar_words(word, embedding):
    """
    The method returns similar words to the given word based on the provided word embedding
    :param word:
    :param embedding:
    :return:
    """
    similar_words = set()
    try:
        for similar_word in embedding.similar_by_word(word):
            similar_words.add(similar_word)
    except KeyError:
        print('key error for word : ' + word)
    return similar_words


if __name__ == '__main__':
    print(code_to_vocab([162, 581, 1723]))
    print(code_to_vocab([18, 139]))

    dataset = [[[18], [[13, 13], [123]], [[0, 0, 0, 0], [0, 0, 0, 1]]],
               [[20], [[44, 44], [24, 23]], [[0, 1, 0, 0], [1, 0, 0, 0]]]]
    print(dataset)
