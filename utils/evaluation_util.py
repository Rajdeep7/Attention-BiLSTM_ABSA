import numpy as np
from utils.util import code_to_vocab
import tensorflow as tf
from config.settings import TINY
from preprocessing.germEval17_data_processing import GERMEVAL_INDEX_TO_ASPECT_WORD_MAP, \
    GERMEVAL_INDEX_TO_ASPECT_SENTIMENT_WORD_MAP


def compute_f1_score(confusion_matrix, class_id, scope):
    with tf.variable_scope(scope):
        tiny = tf.constant(name = 'tiny', value = TINY)
        true_positive = tf.cast(confusion_matrix[class_id, class_id], dtype = tf.float32)
        false_positive = tf.abs(
            tf.cast(tf.reduce_sum(confusion_matrix[:, class_id]), dtype = tf.float32) - true_positive)
        false_negative = tf.abs(
            tf.cast(tf.reduce_sum(confusion_matrix[class_id, :]), dtype = tf.float32) - true_positive)
        precision = tf.truediv(name = 'precision', x = true_positive, y = true_positive + false_positive + tiny)
        recall = tf.truediv(name = 'recall', x = true_positive, y = true_positive + false_negative + tiny)
    f1_score = tf.truediv(name = scope, x = 2 * precision * recall, y = precision + recall + tiny)
    return f1_score


def compute_f1_scores(confusion_matrix, num_classes):
    true_positives = np.zeros(shape = num_classes, dtype = np.int32)
    false_positives = np.zeros(shape = num_classes, dtype = np.int32)
    false_negatives = np.zeros(shape = num_classes, dtype = np.int32)
    precisions = np.zeros(shape = num_classes, dtype = np.float32)
    recalls = np.zeros(shape = num_classes, dtype = np.float32)
    f1_scores = np.zeros(shape = num_classes, dtype = np.float32)

    for i in range(num_classes):
        true_positives[i] = confusion_matrix[i, i]
        false_positives[i] = np.abs(np.sum(confusion_matrix[:, i]) - true_positives[i])
        false_negatives[i] = np.abs(np.sum(confusion_matrix[i, :]) - true_positives[i])
        precisions[i] = true_positives[i] / (true_positives[i] + false_positives[i] + TINY)
        recalls[i] = true_positives[i] / (true_positives[i] + false_negatives[i] + TINY)
        f1_scores[i] = (2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i] + TINY)

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1_score = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall + TINY)
    return f1_scores, macro_f1_score


def compute_accuracy(confusion_matrix, num_classes):
    per_class_accuracy = np.zeros(shape = num_classes)
    total_true_positives = 0
    total_samples = 0

    for i in range(num_classes):
        true_postive_count = confusion_matrix[i, i]
        samples_count = np.sum(confusion_matrix[i, :])
        total_true_positives += true_postive_count
        total_samples += samples_count
        per_class_accuracy[i] = true_postive_count / samples_count
    total_acc = total_true_positives / total_samples
    return per_class_accuracy, total_acc


def get_aspect_index(aspect, aspect_word_index_map):
    """
    It returns the static aspect name index.

    :param aspect:
    :param aspect_word_index_map:
    :return:
    """
    aspect_words = code_to_vocab(aspect)
    aspect_words = ''.join(aspect_words)
    aspect_words = aspect_words.replace('<PAD>', '')
    # print(aspect_words)
    return aspect_word_index_map.get(aspect_words)


def evaluation_metrics(aspects, reviews, labels, preds, aspect_word_index_map, embedding = None):
    """
    This method is a wrapper method that computes different eveluation metrics and aggregates them together.

    :param aspects:
    :param reviews:
    :param labels:
    :param preds:
    :param aspect_word_index_map:
    :return:
    """
    np.set_printoptions(threshold = np.nan)

    if embedding == 'elmo':
        batch_size, n_sentences, _, _ = reviews.shape
    else:
        batch_size, n_sentences, _ = reviews.shape
    flat_lables = np.reshape(labels, [batch_size * n_sentences])
    flat_preds = np.reshape(preds, [batch_size * n_sentences])
    n_aspect = len(aspect_word_index_map) - 1
    n_sentiment_classes = 4
    n_total_sentences = n_aspect * n_sentences

    if len(flat_lables) != n_total_sentences or len(flat_preds) != n_total_sentences:
        print('ERROR~~: ')
    elif len(flat_lables) != len(flat_preds):
        print('ERROR: label-pred dimension mismatch')
    else:
        per_aspect_sentiments_cm = calculate_per_aspect_sentiment_cm(aspects, flat_lables, flat_preds, n_sentences,
                                                                     aspect_word_index_map, n_sentiment_classes)
        per_aspect_aspect_detection_cm = calculate_per_aspect_aspect_detection_cm(aspects, flat_lables, flat_preds,
                                                                                  n_sentences,
                                                                                  aspect_word_index_map)
        joint_aspect_sentiment_cm = calculate_joint_aspect_sentiment_cm(aspects, flat_lables, flat_preds,
                                                                        n_sentences,
                                                                        aspect_word_index_map, n_sentiment_classes)
        n_multilabel_success, n_multilabel_failure = calculate_absolute_joint_multilabel_evaluation(aspects,
                                                                                                    flat_lables,
                                                                                                    flat_preds,
                                                                                                    n_sentences,
                                                                                                    aspect_word_index_map)
        result = {
            'per_aspect_sentiments_cm': per_aspect_sentiments_cm,
            'per_aspect_aspect_detection_cm': per_aspect_aspect_detection_cm,
            'joint_aspect_sentiment_cm': joint_aspect_sentiment_cm,
            'n_multilabel_success': n_multilabel_success,
            'n_multilabel_failure': n_multilabel_failure,
            'count': n_sentences
        }

        # print(aspects)
        # for aspect in aspects:
        #     print(''.join(code_to_vocab(aspect)))
        # print(labels)
        # print(flat_lables)
        # print(preds)
        # print(flat_preds)
        # print(result)
        # print('----------')
        return result


def calculate_absolute_joint_multilabel_evaluation(aspects, labels, preds, n_sentences, aspect_word_index_map):
    """
    This method computes absolute multilabel accuracy. A prediction is considered as success if all its labels are
    correct. Otherwise it is marked as failure. It is stricter than the joint aspect+sentiment f1 metric where every
    label for a sentence is considered individually.

    :param aspects:
    :param labels:
    :param preds:
    :param n_sentences:
    :param aspect_word_index_map:
    :param n_sucess:
    :param n_failure:
    :return:
    """
    n_success = 0
    n_failure = 0
    aspect_with_sentiments = [0, 1, 2]

    for s in range(n_sentences):
        sentence_lables = []
        sentence_preds = []
        for a, aspect in enumerate(aspects):
            aspect_idx = get_aspect_index(aspect, aspect_word_index_map)
            # if aspect_idx is None:
            #     continue
            idx = a * n_sentences + s
            sent_label = labels[idx]
            sent_pred = preds[idx]
            if sent_label in aspect_with_sentiments:
                sentence_lables.append(aspect_idx + sent_label)
            if sent_pred in aspect_with_sentiments:
                sentence_preds.append(aspect_idx + sent_pred)

        if np.all(sentence_lables == sentence_preds):
            n_success += 1
        else:
            n_failure += 1
    return n_success, n_failure


def calculate_joint_aspect_sentiment_cm(aspects, labels, preds, n_sentences, aspect_word_index_map,
                                        n_sentiment_classes):
    """
    The goal of calculating this metric is to evaluate how well the model performs truly on the task of joint aspect
    and sentiment detection.

    Since our model by design does not classifies into the total possible joint(n_aspects*n_sentiments) classes, so we
    won't be able to truly calculate this confusion matrix. However the current computations should provide a good
    approximation.

    This is just an expanded or tiled version of per_aspect_sentiment_cm

    :param aspects:
    :param labels:
    :param preds:
    :param n_sentences:
    :param aspect_word_index_map:
    :param n_sentiment_classes:
    :return:
    """
    n_aspect = len(aspect_word_index_map) - 1
    aspect_sentiment_occurrennces = []
    # +1 is for the none sentiment class
    n_total_classes = n_aspect * (n_sentiment_classes - 1) + 1
    aspect_with_sentiments = [0, 1, 2]

    for s in range(n_sentences):
        sentence_lables = set()
        sentence_preds = set()
        for a, aspect in enumerate(aspects):
            aspect_idx = get_aspect_index(aspect, aspect_word_index_map)
            # if aspect_idx is None:
            #     continue
            idx = a * n_sentences + s
            sent_label = labels[idx]
            sent_pred = preds[idx]
            if sent_label in aspect_with_sentiments:
                sentence_lables.add(aspect_idx * (n_sentiment_classes - 1) + sent_label)
            if sent_pred in aspect_with_sentiments:
                sentence_preds.add(aspect_idx * (n_sentiment_classes - 1) + sent_pred)

        aspect_sentiment_occurrennces.extend(
            compute_occurrences(gold_labels = sentence_lables, predictions = sentence_preds,
                                none_label = n_total_classes - 1))

    cm = compute_cm(n_classes = n_total_classes, occurrences = aspect_sentiment_occurrennces)

    return cm


def calculate_per_aspect_aspect_detection_cm(aspects, labels, preds, n_sentences, aspect_word_index_map):
    """
    This metric evaluates how well the model is able to detect aspects when it is also trying to detetct sentiments. Here
    even if the model wrongly predicts sentiment for a correctly detected aspect then it considered as a success case.

    If a sentence has multiple aspect labels, this metric considers the sentence individually/independently for each
    aspect label.

    :param aspects:
    :param labels:
    :param preds:
    :param n_sentences:
    :param aspect_word_index_map:
    :return:
    """
    n_aspect = len(aspect_word_index_map)
    aspect_occurrences = []
    aspect_detection_success = [0, 1, 2]

    for s in range(n_sentences):
        sentence_lables = set()
        sentence_preds = set()
        for a, aspect in enumerate(aspects):
            aspect_idx = get_aspect_index(aspect, aspect_word_index_map)
            idx = a * n_sentences + s
            sent_label = labels[idx]
            sent_pred = preds[idx]
            if sent_label in aspect_detection_success:
                sentence_lables.add(aspect_idx)
            if sent_pred in aspect_detection_success:
                sentence_preds.add(aspect_idx)

        aspect_occurrences.extend(compute_occurrences(gold_labels = sentence_lables, predictions = sentence_preds,
                                                      none_label = aspect_word_index_map.get('none')))
    cm = compute_cm(n_classes = n_aspect, occurrences = aspect_occurrences)

    return cm


def compute_occurrences(gold_labels, predictions, none_label):
    """
    This method computes occurrence tuples <gold, pred> from gold labels and predictions.
    :param gold_labels:
    :param predictions:
    :param aspect_word_index_map:
    :return:
    """
    occurrences = []
    if len(gold_labels) > 0:
        if len(predictions) > 0:
            for gold in gold_labels:
                if gold in predictions:
                    # for handling true positive
                    occurrences.append([gold, gold])
                else:
                    # for handling false negatives
                    occurrences.append([gold, none_label])

            # for handling false positives
            for pred in predictions:
                if pred not in gold_labels:
                    occurrences.append([none_label, pred])
        else:
            for gold in gold_labels:
                occurrences.append([gold, none_label])
    else:
        pass

    return occurrences


def compute_cm(n_classes, occurrences):
    """
    This method computes confusion matrix based on the computed occurrence tuples of <gold, pred>
    :param n_classes:
    :param occurrences:
    :return:
    """
    cm = np.zeros(shape = (n_classes, 2, 2), dtype = np.int32)
    for curr_class in range(n_classes):
        for curr_occurrence in occurrences:
            gold = curr_occurrence[0]
            pred = curr_occurrence[1]
            if gold == curr_class:
                if gold == pred:
                    # true positive
                    cm[curr_class, 0, 0] += 1
                else:
                    # false negative
                    cm[curr_class, 0, 1] += 1
            else:
                if pred == curr_class:
                    # false positive
                    cm[curr_class, 1, 0] += 1
                else:
                    # true negative
                    cm[curr_class, 1, 1] += 1
    return cm


def calculate_per_aspect_sentiment_cm(aspects, labels, preds, n_sentences, aspect_word_index_map, n_sentiment_classes):
    """
    This metric does not purely evaluates ABSA in general the sence where aspect is given and you have to just predict
    the sentiment classes. Our mode does not assume that the aspect is present in the sentence, hence it classifies into
    3 polarity classses + 1 not applicable class denoting the absesnce of the aspect in the given sentence.

    This metric tells how well the model detects a given aspect and if the aspect is present then how well it is able
    to detect a sentiment corresponding to that aspect.

    :param aspects:
    :param labels:
    :param preds:
    :param n_sentences:
    :param aspect_word_index_map:
    :param n_sentiment_classes:
    :return:
    """
    n_aspect = len(aspects)
    cm = np.zeros(shape = (n_aspect, n_sentiment_classes, n_sentiment_classes), dtype = np.int32)
    for a, aspect in enumerate(aspects):
        aspect_idx = get_aspect_index(aspect, aspect_word_index_map)
        for s in range(n_sentences):
            idx = a * n_sentences + s
            sent_label = labels[idx]
            sent_pred = preds[idx]
            if sent_label == sent_pred:
                # similar predictions
                cm[aspect_idx, sent_label, sent_label] += 1
            else:
                # different predictions
                cm[aspect_idx, sent_label, sent_pred] += 1
    return cm


def calculate_per_aspect_sentiment_f1_scores(confusion_matrix):
    n_aspect, n_sentiment_classes, _ = confusion_matrix.shape
    per_aspect_sentiment_f1_scores = np.zeros(shape = (n_aspect, n_sentiment_classes), dtype = np.float32)
    per_aspect_sentiment_macro_f1_scores = np.zeros(shape = n_aspect, dtype = np.float32)
    for a in range(n_aspect):
        cm_a = confusion_matrix[a]
        f1_scores_a, macro_f1_score_a = compute_f1_scores(confusion_matrix = cm_a, num_classes = n_sentiment_classes)
        per_aspect_sentiment_f1_scores[a] = f1_scores_a
        per_aspect_sentiment_macro_f1_scores[a] = macro_f1_score_a
    return per_aspect_sentiment_f1_scores, per_aspect_sentiment_macro_f1_scores


def calculate_all_aspect_sentiment_f1_scores(confusion_matrix):
    n_aspect, n_sentiment_classes, _ = confusion_matrix.shape
    all_aspect_confusion_matrix = np.sum(confusion_matrix, axis = 0)
    f1_scores, macro_f1_score = compute_f1_scores(all_aspect_confusion_matrix, n_sentiment_classes)
    return f1_scores, macro_f1_score


def calculate_per_aspect_aspect_detection_f1_score(confusion_matrix):
    n_aspect, n_classes, _ = confusion_matrix.shape
    per_aspect_aspect_detection_f1_scores = np.zeros(shape = n_aspect, dtype = np.float32)
    for a in range(n_aspect):
        cm_a = confusion_matrix[a]
        f1_scores, _ = compute_f1_scores(confusion_matrix = cm_a, num_classes = n_classes - 1)
        per_aspect_aspect_detection_f1_scores[a] = f1_scores[0]
    return per_aspect_aspect_detection_f1_scores


def calculate_all_aspect_aspect_detection_micro_f1_score(confusion_matrix):
    n_aspect, n_classes, _ = confusion_matrix.shape
    all_aspect_confusion_matrix = np.sum(confusion_matrix, axis = 0)
    print(all_aspect_confusion_matrix)
    f1_scores, _ = compute_f1_scores(all_aspect_confusion_matrix, n_classes - 1)
    return f1_scores[0]


def calculate_joint_aspect_sentiment_f1_scores(confusion_matrix):
    n_aspect_sentiment_combinations, n_classes, _ = confusion_matrix.shape
    joint_aspect_sentiment_f1_scores = np.zeros(shape = n_aspect_sentiment_combinations, dtype = np.float32)
    for i in range(n_aspect_sentiment_combinations):
        cm_as = confusion_matrix[i]
        f1_scores, _ = compute_f1_scores(confusion_matrix = cm_as, num_classes = n_classes - 1)
        joint_aspect_sentiment_f1_scores[i] = f1_scores[0]
    return joint_aspect_sentiment_f1_scores


def calculate_joint_aspect_sentiment_micro_f1_score(confusion_matrix):
    n_aspect_sentiment_combinations, n_classes, _ = confusion_matrix.shape
    # all_aspect_confusion_matrix = np.zeros(shape = (n_classes, n_classes), dtype = np.int32)
    # for i in range(n_aspect_sentiment_combinations):
    #     if ((i + 1) % 4) != 0:
    #         all_aspect_confusion_matrix += confusion_matrix[i]
    all_aspect_confusion_matrix = np.sum(confusion_matrix, axis = 0)
    print(all_aspect_confusion_matrix)
    f1_scores, _ = compute_f1_scores(all_aspect_confusion_matrix, n_classes - 1)
    return f1_scores[0]
