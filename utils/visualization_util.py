import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import os
from config.settings import CHECKPOINT_DIR
import itertools
from utils.util import code_to_vocab
from utils.evaluation_util import compute_f1_scores, compute_accuracy, calculate_per_aspect_sentiment_f1_scores, \
    calculate_all_aspect_sentiment_f1_scores, calculate_per_aspect_aspect_detection_f1_score, \
    calculate_all_aspect_aspect_detection_micro_f1_score, calculate_joint_aspect_sentiment_f1_scores, \
    calculate_joint_aspect_sentiment_micro_f1_score


def write_experiment_parameters(args):
    dict_args = vars(args)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    file_path = os.path.join(CHECKPOINT_DIR, 'experiment_params.txt')
    with open(file_path, 'w') as f:
        for arg_key in sorted(dict_args):
            line = arg_key + '=' + str(dict_args[arg_key]) + '\n'
            f.write(line)


def print_results(info, args, msg, accuracies = None, per_class_accuracies = None, f1_scores = None,
                  per_class_f1_scores = None):
    loss = np.mean(np.asarray(info['loss']))
    acc = np.mean(np.asarray(info['accuracy']))
    f1_score = np.mean(np.asarray(info['f1_score']))
    confusion_matrix = info['confusion_matrix']

    per_class_accuracy, total_acc = compute_accuracy(confusion_matrix, args.num_classes)
    pc_f1_scores, macro_f1_score = compute_f1_scores(confusion_matrix, args.num_classes)

    if accuracies is not None and per_class_accuracies is not None and f1_scores is not None and per_class_f1_scores is not None:
        accuracies.append(total_acc)
        per_class_accuracies.append(per_class_accuracy)
        f1_scores.append(macro_f1_score)
        per_class_f1_scores.append(pc_f1_scores)

    print('-----' + msg + '-----')
    print('Total mean accuracy over batches: %.2f' % acc)
    print('Total mean loss over batches: %.2f' % loss)
    print('Total mean f1 score over batches: %.2f' % f1_score)
    print('Confusion matrix')
    print(confusion_matrix)
    print('Total accuracy using full confusion matrix: %.2f' % total_acc)
    print('Accuracy of detecting positive sentiment: %.2f' % per_class_accuracy[0])
    print('Accuracy of detecting negative sentiment: %.2f' % per_class_accuracy[1])
    print('Accuracy of detecting neutral sentiment: %.2f' % per_class_accuracy[2])
    print('Accuracy of detecting non applicability: %.2f' % per_class_accuracy[3])
    print('Total Macro f1 score: %.2f' % macro_f1_score)
    print('F1 score of detecting positive sentiment: %.2f' % pc_f1_scores[0])
    print('F1 score of detecting negative sentiment: %.2f' % pc_f1_scores[1])
    print('F1 score of detecting neutral sentiment: %.2f' % pc_f1_scores[2])
    print('F1 score of detecting non applicability: %.2f' % pc_f1_scores[3])
    print('----------------------------')


def plotConfusionMatrix(
        confMatrix: np.ndarray,
        classes = [],
        normalize = False,
        filePath = "",
        title = 'Confusion matrix',
        cmap = plt.cm.Blues,
) -> None:
    """This function plots the confusion matrix. Normalize via normalize=True."""
    if normalize:
        confMatrix = confMatrix.astype('float') / confMatrix.sum(axis = 1)[:, np.newaxis]

    plt.figure()
    plt.imshow(confMatrix, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    if not classes:
        classes = [""] * confMatrix.shape[0]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = confMatrix.max() / 2.
    for i, j in itertools.product(range(confMatrix.shape[0]), range(confMatrix.shape[1])):
        plt.text(j, i, format(confMatrix[i, j], fmt),
                 horizontalalignment = "center",
                 color = "white" if confMatrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if filePath:
        plt.savefig(filePath)
    else:
        plt.show()


def plot_accuracy(val_accuracies, train_accuracies, title):
    # line 1 points
    x1 = np.arange(len(val_accuracies))
    y1 = val_accuracies

    # plotting the line 1 points
    plt.plot(x1, y1, label = "Validation")

    # line 2 points
    x2 = np.arange(len(train_accuracies))
    y2 = train_accuracies
    # plotting the line 2 points
    plt.plot(x2, y2, label = "Train")

    # naming the x axis
    plt.xlabel('Iterations')
    # naming the y axis
    plt.ylabel('Accuracy')
    # giving a title to my graph
    plt.title(title)

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.savefig(os.path.join(CHECKPOINT_DIR, title.replace(' ', '') + '.png'), dpi = 300)
    plt.gcf().clear()
    # plt.show()


def plot_f1_score(val_f1_scores, train_f1_scores, title):
    # line 1 points
    x1 = np.arange(len(val_f1_scores))
    y1 = val_f1_scores

    # plotting the line 1 points
    plt.plot(x1, y1, label = "Validation")

    # line 2 points
    x2 = np.arange(len(train_f1_scores))
    y2 = train_f1_scores
    # plotting the line 2 points
    plt.plot(x2, y2, label = "Train")

    # naming the x axis
    plt.xlabel('Iterations')
    # naming the y axis
    plt.ylabel('F1 Score')
    # giving a title to my graph
    plt.title(title)

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.savefig(os.path.join(CHECKPOINT_DIR, title.replace(' ', '') + '.png'), dpi = 300)
    plt.gcf().clear()
    # plt.show()


def compute_aspect_detection_results(test_results, aspect_word_index_map):
    aspects = test_results['aspects']
    labels = test_results['ground_truths']
    predictions = test_results['masked_predictions']
    masks = test_results['masks']
    sentiments = [0, 1, 2]
    aspect_count = len(aspect_word_index_map) + 1
    result = np.zeros(shape = (aspect_count, aspect_count))
    for batch_id in range(len(aspects)):
        a = aspects[batch_id]
        l = labels[batch_id]
        p = predictions[batch_id]
        m = masks[batch_id]
        for review_id in range(len(a)):
            a1 = a[review_id]
            m1 = m[review_id]
            l1 = l[review_id]
            p1 = p[review_id]
            for sent_id in range(len(l1)):
                m2 = m1[sent_id]
                l2 = l1[sent_id]
                p2 = p1[sent_id]
                if m2 == 1:
                    aspect_word = ''
                    for a_word in code_to_vocab(a1):
                        aspect_word += a_word
                    idx = aspect_word_index_map[aspect_word]
                    if l2 in sentiments and p2 in sentiments:
                        result[idx, idx] += 1
                    elif l2 == 3 and p2 == 3:
                        result[-1, -1] += 1
                    else:
                        result[idx, -1] += 1
    print(result)
    for k, v in aspect_word_index_map.items():
        percent = result[v, v] / (result[v, v] + result[v, -1])
        print(k + ' : %.2f' % percent)


def plot_multiple_confusion_matrix(confusion_matrix, count = 12, num_of_plots_in_a_row = 4, titles = [], classes = [],
                                   normalize = True, cmap = plt.cm.Blues, file_path = None):
    if titles is None or len(titles) != count:
        titles = count * ['Confusion Matrix']

    if not classes:
        classes = [''] * confusion_matrix[0].shape[0]

    fmt = '.2f' if normalize else 'd'
    tick_marks = np.arange(len(classes))
    n_rows = int(count / num_of_plots_in_a_row)
    n_cols = int(num_of_plots_in_a_row)
    plt.figure()

    for i in range(count):
        print('.')
        cm = confusion_matrix[i]

        if normalize:
            cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

        plt.subplot(n_rows, n_cols, i + 1)
        # plt.subplots_adjust(hspace = 0.75, wspace = 0.75)
        plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
        plt.title(titles[i])
        plt.colorbar()
        plt.xticks(tick_marks, classes, rotation = 45)
        plt.yticks(tick_marks, classes)

        # if (i + 1) % n_cols == 1:
        plt.ylabel('True label')
        # if (i + 1) > count - n_cols:
        plt.xlabel('Predicted label')
        plt.tight_layout()

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment = "center",
                     color = "white" if cm[i, j] > thresh else "black",
                     fontsize = 'x-small')

    if file_path:
        plt.savefig(file_path)
    else:
        plt.show()


def plot_horizontal_bar_from_cm(confusion_matrix = None, classes = []):
    # plt.rcdefaults()
    # plt.subplots(figsize = (10, 30))
    width = 0.30

    # Example data
    y_lables = 12 * ['a', 'b', 'c', 'd']
    y_pos = list(range(len(y_lables)))
    print(y_pos)
    true_positives = 12 * [8.84036186, 12.94095337, 11.19919226, 10.64395389]
    false_negatives = 12 * [1, 1, 1, 1]
    false_positives = 12 * [2, 13, 13, 3]

    TP = plt.barh(y_pos, true_positives, width, color = 'green', label = 'TP')
    FN = plt.barh(y_pos, false_negatives, width, label = 'FN', left = TP)
    plt.barh(y_pos, false_positives, width, label = 'FP', left = FN)
    # ax.barh([p + width for p in y_pos], false_negatives, width, label = 'FN')
    # ax.barh([p + width * 2 for p in y_pos], false_positives, width, label = 'FP')
    plt.set_yticks([p + 1.5 * width for p in y_pos])
    plt.set_yticklabels(y_lables)
    plt.invert_yaxis()  # labels read top-to-bottom
    plt.set_xlabel('Performance')
    plt.set_title('How fast do you want to go today?')

    plt.legend(['TP', 'FN', 'FP'], loc = 'upper right')
    plt.show()


def plot_per_aspect_sentiments_cm(confusion_matrix, index_aspect_word_map):
    classes = ['Positive', 'Negative', 'Neutral', 'N/A']
    n_aspects = len(index_aspect_word_map)
    num_of_plots_in_a_row = 2
    titles = []
    for i in range(n_aspects):
        titles.append(index_aspect_word_map.get(i, None))


def print_test_evaluation_metrics(test_metrics):
    per_aspect_sentiments_cm = test_metrics['per_aspect_sentiments_cm']
    per_aspect_aspect_detection_cm = test_metrics['per_aspect_aspect_detection_cm']
    joint_aspect_sentiment_cm = test_metrics['joint_aspect_sentiment_cm']
    n_multilabel_success = test_metrics['n_multilabel_success']
    n_multilabel_failure = test_metrics['n_multilabel_failure']

    acc = n_multilabel_success / (n_multilabel_success + n_multilabel_failure)
    per_aspect_sentiment_f1_scores, per_aspect_sentiment_macro_f1_scores = calculate_per_aspect_sentiment_f1_scores(
        per_aspect_sentiments_cm)
    f1_scores, macro_f1_score = calculate_all_aspect_sentiment_f1_scores(per_aspect_sentiments_cm)
    per_aspect_aspect_detection_f1_scores = calculate_per_aspect_aspect_detection_f1_score(
        per_aspect_aspect_detection_cm)
    all_aspect_detection_micro_f1_score = calculate_all_aspect_aspect_detection_micro_f1_score(
        per_aspect_aspect_detection_cm)
    joint_aspect_sentiment_f1_scores = calculate_joint_aspect_sentiment_f1_scores(joint_aspect_sentiment_cm)
    joint_aspect_sentiment_micro_f1_score = calculate_joint_aspect_sentiment_micro_f1_score(joint_aspect_sentiment_cm)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('Total Sentences: ', test_metrics['n_sentence'])
    print('Multiclass-Multilabel accuracy: %.2f' % acc)
    print('Per aspect sentiment f1 scores')
    print(per_aspect_sentiment_f1_scores)
    print('Per aspect sentiment Macro f1 scores')
    print(per_aspect_sentiment_macro_f1_scores)
    print('All aspect sentiment f1 scores')
    print(f1_scores)
    print('All aspect sentiment Macro f1 score: %.2f' % macro_f1_score)
    print('Per aspect aspect detection f1 scores')
    print(per_aspect_aspect_detection_f1_scores)
    print('All aspect detection micro f1 score: %.5f' % all_aspect_detection_micro_f1_score)
    print('Joint aspect sentiment f1 scores')
    print(joint_aspect_sentiment_f1_scores)
    print('Joint aspect sentiment micro f1 score: %.5f' % joint_aspect_sentiment_micro_f1_score)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@')
