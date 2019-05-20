import os
import sys

sys.path.append(os.getcwd())
import tensorflow as tf
import numpy as np
from utils.util import load_fastText_embeddings, load_glove_embeddings
from utils.data_util import read_text, read_binary
from config.settings import GLOVE_EMBEDDINGS_FILE, ALL_SENTENCES_TEXT, WORD_FREQ_FILE
from utils.visualization_util import plotConfusionMatrix, plot_multiple_confusion_matrix, plot_horizontal_bar_from_cm
from gensim.models import KeyedVectors


def concatenate_aspect_to_words():
    aspect = tf.constant([[4, 4], [1, 1]])
    batch_size = tf.constant(2)
    sentence_count = tf.constant(4)
    word_count = tf.constant(3)
    aspect = tf.tile(aspect, [1, sentence_count * word_count])
    print(aspect)
    print('-----')
    aspect = tf.reshape(aspect, [batch_size, sentence_count, word_count, 2])

    word = tf.constant(np.zeros([2, 4, 3, 5], dtype = np.int32))
    print(word)
    word = tf.concat([word, aspect], -1)
    with tf.Session() as sess:
        print(sess.run([word]))


def concatenate_aspect_to_bilstm():
    aspect = tf.constant([[4, 4], [1, 1]])
    aspect = tf.tile(aspect, tf.constant([1, 4 * 3]))
    print(aspect)
    aspect = tf.reshape(aspect, [2 * 4, 3, 2])

    bilstm = tf.constant(np.zeros([2 * 4, 3, 5], dtype = np.int32))
    print(bilstm)
    bilstm = tf.concat([bilstm, aspect], -1)
    print(bilstm)
    with tf.Session() as sess:
        print(sess.run([bilstm]))


def concatenate_aspect_to_sentence():
    aspect = tf.constant([[4, 4], [1, 1]])
    aspect = tf.tile(aspect, tf.constant([1, 4]))
    print(aspect)
    aspect = tf.reshape(aspect, [2, 4, 2])

    sent = tf.constant(np.zeros([2, 4, 5], dtype = np.int32))
    print(sent)
    sent = tf.concat([sent, aspect], -1)
    print(sent)
    with tf.Session() as sess:
        print(sess.run([sent]))


def fuse_aspect_words():
    aspect = tf.constant([[[4, 4], [1, 1], [4, 5]], [[7, 7], [8, 8], [9, 3]]], dtype = tf.float32)
    print(aspect.get_shape()[1])
    compressed_aspect = tf.reduce_mean(aspect, axis = 1)
    print(compressed_aspect)
    with tf.Session() as sess:
        print(sess.run(compressed_aspect))


def confusion_matrix():
    labels = tf.constant([[3, 3, 3, 3, 3, 3, 3], [3, 1, 3, 3, 0, 0, 0], [3, 3, 3, 3, 3, 3, 3]])
    predictions = tf.constant([[3, 3, 3, 3, 3, 3, 3], [2, 2, 2, 2, 0, 0, 0], [1, 1, 3, 3, 3, 3, 3]])
    # label_weights = tf.constant([[0.2746345, 0.2746345, 0.2746345, 0.2746345, 0.2746345, 0.2746345, 0.2746345],
    #                              [0.2746345, 9.35104, 0.2746345, 0.2746345, 0., 0., 0.],
    #                              [0.2746345, 0.2746345, 0.2746345, 0.2746345, 0.2746345, 0.2746345, 0.2746345]])
    label_weights = tf.constant([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0., 0., 0.], [1, 1, 1, 1, 1, 1, 1]])
    flat_labels = tf.reshape(labels, shape = [3 * 7])
    flat_predictions = tf.reshape(predictions, shape = [3 * 7])
    flat_label_weights = tf.reshape(label_weights, shape = [3 * 7])
    cm = tf.confusion_matrix(labels = flat_labels, predictions = flat_predictions,
                             num_classes = 4, weights = flat_label_weights)
    print(cm)
    class_id = 3
    TINY = tf.constant(0.0001)
    true_positive = tf.cast(cm[class_id, class_id], dtype = tf.float32)
    false_positive = tf.abs(tf.cast(tf.reduce_sum(cm[:, class_id]), dtype = tf.float32) - true_positive)
    false_negative = tf.abs(tf.cast(tf.reduce_sum(cm[class_id, :]), dtype = tf.float32) - true_positive)
    precision = true_positive / (true_positive + false_positive + TINY)
    recall = true_positive / (true_positive + false_negative + TINY)
    f1_score = 2 * precision * recall / (precision + recall + TINY)

    with tf.Session() as sess:
        confMatrix = sess.run(cm)
        plotConfusionMatrix(confMatrix = confMatrix, normalize = True)
        print(confMatrix)
        print(sess.run(true_positive))
        print(sess.run(precision))
        print(sess.run(recall))
        print(sess.run(f1_score))


def plot_confusion_matrix():
    matrix_1 = np.array([[124, 15, 4, 128], [10, 27, 1, 47], [7, 1, 2, 14], [51, 43, 5, 3365]])
    matrix_2 = np.array([[182, 5, 2, 74], [11, 41, 3, 32], [13, 3, 2, 5], [62, 20, 2, 1048]])
    matrix_3 = np.array([[182, 9, 3, 76], [12, 41, 2, 32], [11, 4, 4, 3], [52, 18, 8, 1103]])
    experiment_22 = np.array([[151, 11, 1, 77], [8, 72, 1, 36], [10, 5, 5, 3], [43, 40, 0, 1076]])

    plotConfusionMatrix(confMatrix = experiment_22, normalize = True,
                        classes = ['Positive', 'Negative', 'Neutral', 'Not Applicable'])


def glove_emebddings():
    for i, line in enumerate(read_text(GLOVE_EMBEDDINGS_FILE)):
        word_embedding = line.split('\n')[0].split(' ')
        word = word_embedding[0]
        embedding = word_embedding[1:]
        print(word)
        print(embedding)
        if i == 3:
            break


def check_call_by_reference(obj):
    obj.append(0)


def create_word_mask():
    # batch_size = tf.constant(2)
    # sentence_count = tf.constant(4)
    # word_count = tf.constant(3)
    # word_embedding_size = tf.constant(10)
    # review = tf.ones(shape = (batch_size, sentence_count, word_count, word_embedding_size))
    # mask = tf.constant([[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
    #                     [[0., 0., 0.], [0., 1., 0.], [0., 0., 0.], [0., 0., 0.]]])
    # repeated_mask = tf.repeat(mask, 10, axis = -1)
    #
    #
    # with tf.Session() as sess:
    #     print(sess.run([word]))

    review = np.ones(shape = (2, 4, 3, 10))
    mask = np.zeros(shape = (2, 4, 3))
    mask[1, 1, 1] = 1
    print(mask)
    print('----')
    mask = np.repeat(mask, 10, axis = -1)
    print(mask)
    print('----')
    mask = np.reshape(mask, newshape = (2, 4, 3, 10))
    print(mask)
    print('----')
    print(review * mask)


def fastText():
    fastext = load_fastText_embeddings()
    # Pick a word
    find_similar_to = 'car'

    # Finding out similar words [default= top 10]
    for similar_word in fastext.similar_by_word(find_similar_to):
        print("Word: {0}, Similarity: {1:.2f}".format(
            similar_word[0], similar_word[1]
        ))

    # Printing out the dimension of a word vector
    print("Dimension of a word vector: {}".format(
        len(fastext[find_similar_to])
    ))


def elmo():
    # from allennlp.commands.elmo import ElmoEmbedder
    # elmo = ElmoEmbedder()
    # # tokens = ["I", "ate", "an", "apple", "for", "breakfast"]
    # # vectors = elmo.embed_sentence(tokens)
    # batch = [["I", "ate", "an", "apple", "for", "breakfast"], ["how", "are", "you"]]
    # vectors = elmo.embed_batch(batch = batch)
    # print(len(vectors))
    # print(vectors[0].shape)
    # print(vectors[1].shape)
    # print(vectors[0][0].shape)
    # print('finish')
    from utils.util import load_elmo_emebddings
    elmo = load_elmo_emebddings()
    e = elmo.get('Fair menu selection .')
    print(e.shape)
    print(e[0])


def gensin():
    fastext = load_fastText_embeddings()

    glove = load_glove_embeddings()
    # Pick a word
    find_similar_to = 'food'

    # Finding out similar words [default= top 10]
    for similar_word in fastext.similar_by_word(find_similar_to):
        print("Word: {0}, Similarity: {1:.2f}".format(
            similar_word[0], similar_word[1]
        ))


def duplicate_keys():
    sentence_to_id_map = {}
    with open(ALL_SENTENCES_TEXT, 'r') as f:
        count = 0
        for line in f:
            sent = line.strip('\n')
            if sentence_to_id_map.get(sent, None) is not None:
                print(sent)
            sentence_to_id_map[sent] = count
            count += 1
            # print(sent)
        print(count)
        print(len(sentence_to_id_map))


def nltk_usage():
    import nltk
    # nltk.download('wordnet')
    from nltk.corpus import wordnet

    # Then, we're going to use the term "program" to find synsets like so:
    syns = wordnet.synsets("beautiful")
    print(len(syns))

    for syn in syns:
        # An example of a synset:
        print(syn.name())
        # Definition of that first synset:
        print(syn.definition())
        # Examples of the word in use in sentences:
        print(syn.examples())
        for l in syn.lemmas():
            print(l.name())
        print('----------------------')


def thesaurus():
    from py_thesaurus import Thesaurus
    input_word = "dream"
    new_instance = Thesaurus(input_word)
    print(new_instance.get_synonym())
    print('----------')
    print(new_instance.get_synonym(pos = 'adj'))


def read_words():
    data = read_binary(WORD_FREQ_FILE)
    for i, (w, f) in enumerate(data.items()):
        print(str(i) + '-' + w + ' : ' + str(f))


def multiple_plots():
    matrix_1 = np.array([[124, 15, 4, 128], [10, 27, 1, 47], [7, 1, 2, 14], [51, 43, 5, 3365]])
    matrix_2 = np.array([[182, 5, 2, 74], [11, 41, 3, 32], [13, 3, 2, 5], [62, 20, 2, 1048]])
    matrix_3 = np.array([[182, 9, 3, 76], [12, 41, 2, 32], [11, 4, 4, 3], [52, 18, 8, 1103]])
    matrix_4 = np.array([[151, 11, 1, 77], [8, 72, 1, 36], [10, 5, 5, 3], [43, 40, 0, 1076]])

    cm = [matrix_1, matrix_2, matrix_3, matrix_4]
    cm = np.asarray(cm)
    titles = ['aspect1', 'aspect2', 'aspect3', 'aspect4']
    classes = ['Positive', 'Negative', 'Neutral', 'N/A']
    plot_multiple_confusion_matrix(confusion_matrix = cm, count = 4, num_of_plots_in_a_row = 2, classes = classes,
                                   titles = titles, normalize = True)


def horizontal_bar():
    plot_horizontal_bar_from_cm()


def generate_rasa_vectors():
    import fastText
    from config.settings import FASTTEXT_EN_EMBEDDINGS_MODEL
    words = np.load('words.npy')
    model = fastText.load_model(FASTTEXT_EN_EMBEDDINGS_MODEL)
    embeddings = np.zeros((6906, 300), dtype = np.float32)
    for i in range(len(words)):
        embeddings[i] = model.get_word_vector(words[i])
    np.save("embeddings.npy", embeddings)


if __name__ == '__main__':
    # concatenate_aspect_to_words()
    # a = [[0, 0], [1, 1]]
    # print(len(a[0]))
    # print(a[0][1])
    # concatenate_aspect_to_bilstm()
    # concatenate_aspect_to_sentence()
    # fuse_aspect_words()
    # confusion_matrix()
    # plot_confusion_matrix()
    # glove_emebddings()
    # obj = [1, 1]
    # check_call_by_reference(obj)
    # check_call_by_reference(obj)
    # print(obj)
    # create_word_mask()
    # fastText()
    # elmo()
    # gensin()
    # duplicate_keys()
    # nltk_usage()
    # thesaurus()
    read_words()
    # multiple_plots()
    # horizontal_bar()
    # generate_rasa_vectors()
