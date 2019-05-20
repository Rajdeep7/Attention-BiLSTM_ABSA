import argparse
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--task', default = 'organic',
                    choices = ['semeval16-restaurant', 'semeval16-laptops', 'organic'])
parser.add_argument('--mode', default = 'train', choices = ['train', 'test'])
parser.add_argument('--print-frequency', type = int, default = 50)
parser.add_argument('--eval-frequency', type = int, default = 100)
parser.add_argument('--batch-size', type = int, default = 64)
parser.add_argument('--device', default = '/cpu:0')
parser.add_argument('--max-grad-norm', type = float, default = 1.0)
parser.add_argument('--lr', type = float, default = 0.001)
parser.add_argument('--vocab-size', type = int, default = 17308, choices = [4761, 17308, 8705, 5511, 5093, 4308, 4143, 3954])
parser.add_argument('--word-embedding-size', type = int, default = 300)
parser.add_argument('--aspect-embedding-size', type = int, default = 300)
parser.add_argument('--word-cell-units', type = int, default = 64)
parser.add_argument('--word-cell-stacks', type = int, default = 1)
parser.add_argument('--sentence-cell-units', type = int, default = 64)
parser.add_argument('--sentence-cell-stacks', type = int, default = 1)
parser.add_argument('--word-output-size', type = int, default = 300)
# parser.add_argument('--sentence-output-size', type = int, default = 100)
parser.add_argument('--dropout-keep-prob', type = float, default = 0.8)
parser.add_argument('--num-classes', type = int, default = 4)
parser.add_argument('--epochs', type = int, default = 200)
parser.add_argument('--tag', default = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
parser.add_argument('--restore-dir', default = 'combined_data-2019-01-14-17-09-36')
parser.add_argument('--meta-file-path', default = 'combined_data-149900.meta')
parser.add_argument('--embedding-type', default = 'fasttext', choices = ['xavier', 'glove', 'fasttext', 'elmo'])
parser.add_argument('--aspect-fusion', default = 'mean', choices = ['mean', 'max', 'projection'])
parser.add_argument('--reg-constant', type = float, default = 0)
parser.add_argument('--load-graph', type = bool, default = True)
parser.add_argument('--hyperparam-tune', type = bool, default = False)
parser.add_argument('--train-embeddings', type = bool, default = False)
parser.add_argument('--dynamic-class-weights', type = bool, default = False)
parser.add_argument('--aspect-detector', type = bool, default = False)
parser.add_argument('--use-attention', type = bool, default = True)
parser.add_argument('--use-combined-attention-input', type = bool, default = True)
parser.add_argument('--use-word-mask', type = bool, default = False)

args = parser.parse_args()

import os
import sys

sys.path.append(os.getcwd())
import tensorflow as tf
import time
from utils.util import calculate_class_weights, batch_iterator, get_feed_data
from utils.evaluation_util import evaluation_metrics
from utils.data_util import train_loader, test_loader, val_loader
from config.settings import DATA_DIR, TFLOG_DIR, CHECKPOINT_PATH, EXPERIMENT_NAME, EXPERIMENT_PREFIX
from config.tensor_names import PADDED_REVIEWS_TENSOR_NAME, PADDED_LABELS_TENSOR_NAME, ASPECTS_TENSOR_NAME, \
    SENTENCE_MASK_TENSOR_NAME
from utils.visualization_util import write_experiment_parameters, print_results, plot_accuracy, plot_f1_score, \
    compute_aspect_detection_results, print_test_evaluation_metrics
from preprocessing.semEval16_data_processing import RESTAURANT_ASPECT_WORD_INDEX_MAP, LAPTOPS_ASPECT_WORD_INDEX_MAP
from preprocessing.organic_data_processing import ORGANIC_ASPECT_WORD_INDEX_MAP
from models import model
import numpy as np
import ray
from ray import tune


def train(config = None, reporter = None):
    """
    Main method to start training

    :param hyperparam_tune: flag for controlling hyperparameter tuning
    :param config: contains grid searched values for hyperparameters to be tuned
    :param reporter: can contain reporting values like accuracy, f1-score etc
    :return:
    """
    # set values according to hyperparamter tuner
    if args.hyperparam_tune:
        print('Data dir : ' + DATA_DIR)
        args.lr = config['learning_rate']
        args.batch_size = config['batch_size']
        args.dropout_keep_prob = config['dropout_keep_prob']

    print(args)
    write_experiment_parameters(args)

    # https://stackoverflow.com/questions/44873273/what-do-the-options-in-configproto-like-allow-soft-placement-and-log-device-plac
    config = tf.ConfigProto(allow_soft_placement = True)

    # Clears the default graph stack and resets the global default graph.
    tf.reset_default_graph()

    with tf.Session(config = config) as session:

        # get model and saver instances
        _, saver, ops = model.get_model(session = session,
                                        args = args,
                                        restore_only = False)

        # print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

        # get label weights for handling class imbalance
        class_weights = calculate_class_weights()

        # create a training summary writer
        train_writer = tf.summary.FileWriter(TFLOG_DIR, graph = session.graph)

        # initializations
        val_accuracies = []
        val_per_class_accuracies = []
        val_per_class_f1_scores = []
        val_macro_f1_scores = []
        train_accuracies = []
        train_per_class_f1_scores = []
        train_per_class_accuracies = []
        train_macro_f1_scores = []
        train_confusion_matrix = np.zeros(shape = (args.num_classes, args.num_classes), dtype = np.int32)
        best_macro_f1_score = 0
        best_step_number = 0

        # start training
        for i, (x1, x2, y) in enumerate(batch_iterator(train_loader(args.epochs), args.batch_size)):

            t0 = time.clock()

            # calculate dynamic class weights
            if args.dynamic_class_weights:
                class_weights = calculate_class_weights(classes = y)

            # get feed_dicts
            # TODO: calculate class weights for tackling class imbalance
            fd = get_feed_data(x1, x2, y, class_weights = class_weights, is_training = True, args = args)

            # run session
            step, summaries, loss, accuracy, f1_score, f1_score_0, f1_score_1, f1_score_2, f1_score3, \
            confusion_matrix, labels, predictions, label_weights, _ = session.run(
                [
                    ops['global_step'],
                    ops['summary_op'],
                    ops['loss'],
                    ops['accuracy'],
                    ops['f1_score'],
                    ops['f1_score_0'],
                    ops['f1_score_1'],
                    ops['f1_score_2'],
                    ops['f1_score_3'],
                    ops['confusion_matrix'],
                    ops['padded_labels'],
                    ops['predictions'],
                    ops['label_weights'],
                    ops['train_op']
                ], fd)

            train_writer.add_summary(summaries, global_step = step)
            td = time.clock() - t0

            if args.hyperparam_tune:
                reporter(f1_score = f1_score)

            if step % args.print_frequency == 0:
                train_confusion_matrix += confusion_matrix
                print('step %s, loss=%s, accuracy=%s, f1_score=%s, t=%s, inputs=%s' % (
                    step, loss, accuracy, f1_score, round(td, 2), fd[PADDED_REVIEWS_TENSOR_NAME].shape))
            if step != 0 and step % args.eval_frequency == 0:
                # run validation
                val_results = evaluate(session = session, ops = ops, dataset = val_loader(epochs = 1))
                print_results(val_results, args, 'VALIDATION RESULTS', val_accuracies, val_per_class_accuracies,
                              val_macro_f1_scores, val_per_class_f1_scores)
                # save a checkpoint if best f1 score
                if val_macro_f1_scores[-1] >= best_macro_f1_score:
                    best_macro_f1_score = val_macro_f1_scores[-1]
                    best_step_number = step
                    print('Best Macro F1 Score : %.2f' % best_macro_f1_score)
                    print('Best step at : ' + str(best_step_number))
                    saver.save(session, CHECKPOINT_PATH, global_step = step)
                    print('checkpoint saved')
                train_results = {'loss': loss, 'accuracy': accuracy, 'f1_score': f1_score,
                                 'confusion_matrix': train_confusion_matrix}
                print_results(train_results, args, 'TRAINING RESULTS', train_accuracies,
                              train_per_class_accuracies, train_macro_f1_scores, train_per_class_f1_scores)
                # reset train confusion matrix
                train_confusion_matrix = np.zeros(shape = (args.num_classes, args.num_classes), dtype = np.int32)

        val_per_class_accuracies = np.asarray(val_per_class_accuracies)
        train_per_class_accuracies = np.asarray(train_per_class_accuracies)
        val_per_class_f1_scores = np.asarray(val_per_class_f1_scores)
        train_per_class_f1_scores = np.asarray(train_per_class_f1_scores)

        plot_accuracy(val_accuracies, train_accuracies, title = 'Accuracy')
        plot_accuracy(val_per_class_accuracies[:, 0], train_per_class_accuracies[:, 0],
                      title = 'Accuracy Class 0 Positive Sentiment')
        plot_accuracy(val_per_class_accuracies[:, 1], train_per_class_accuracies[:, 1],
                      title = 'Accuracy Class 1 Negative Sentiment')
        plot_accuracy(val_per_class_accuracies[:, 2], train_per_class_accuracies[:, 2],
                      title = 'Accuracy Class 2 Neutral Sentiment')
        plot_accuracy(val_per_class_accuracies[:, 3], train_per_class_accuracies[:, 3],
                      title = 'Accuracy Class 3 Not Applicable Sentiment')

        plot_f1_score(val_macro_f1_scores, train_macro_f1_scores, title = 'Macro F1 Score')
        plot_f1_score(val_per_class_f1_scores[:, 0], train_per_class_f1_scores[:, 0],
                      title = 'F1 Score Class 0 Positive Sentiment')
        plot_f1_score(val_per_class_f1_scores[:, 1], train_per_class_f1_scores[:, 1],
                      title = 'F1 Score Class 1 Negative Sentiment')
        plot_f1_score(val_per_class_f1_scores[:, 2], train_per_class_f1_scores[:, 2],
                      title = 'F1 Score Class 2 Neutral Sentiment')
        plot_f1_score(val_per_class_f1_scores[:, 3], train_per_class_f1_scores[:, 3],
                      title = 'F1 Score Class 3 Not Applicable Sentiment')

        return best_step_number


def evaluate(session, ops, dataset):
    losses = []
    accuracies = []
    f1_scores = []
    masked_predictions = []
    aspects = []
    ground_truths = []
    masks = []
    cm = np.zeros(shape = (args.num_classes, args.num_classes), dtype = np.int32)
    test_metrics = {}
    if args.mode == 'test':
        if args.task == 'semeval16-restaurant':
            aspect_word_index_map = RESTAURANT_ASPECT_WORD_INDEX_MAP
        elif args.task == 'semeval16-laptops':
            aspect_word_index_map = LAPTOPS_ASPECT_WORD_INDEX_MAP
        elif args.task == 'organic':
            aspect_word_index_map = ORGANIC_ASPECT_WORD_INDEX_MAP
        n_sentiment_classes = args.num_classes
        n_aspect = len(aspect_word_index_map) - 1
        n_total_classes = n_aspect * (n_sentiment_classes - 1) + 1
        n_multilabel_success = 0
        n_multilabel_failure = 0
        n_sentence = 0
        args.batch_size = n_aspect

        per_aspect_sentiments_cm = np.zeros(shape = (n_aspect, n_sentiment_classes, n_sentiment_classes),
                                            dtype = np.int32)
        per_aspect_aspect_detection_cm = np.zeros(shape = (n_aspect + 1, 2, 2), dtype = np.int32)
        joint_aspect_sentiment_cm = np.zeros(shape = (n_total_classes, 2, 2), dtype = np.int32)

    for x1, x2, y in batch_iterator(dataset, args.batch_size, 1):
        # get feed_dicts
        fd = get_feed_data(x1, x2, y, is_training = False, args = args)

        # run evaluation
        val_accuracy, loss, f1_score, confusion_matrix, masked_prediction = session.run(
            [ops['accuracy'], ops['loss'], ops['f1_score'], ops['confusion_matrix'], ops['masked_predictions']], fd)

        losses.append(loss)
        accuracies.append(val_accuracy)
        f1_scores.append(f1_score)
        cm += confusion_matrix
        masked_predictions.append(masked_prediction)
        aspects.append(x1)
        ground_truths.append(y)
        masks.append(fd[SENTENCE_MASK_TENSOR_NAME])
        if args.mode == 'test':
            eval_results = evaluation_metrics(fd[ASPECTS_TENSOR_NAME], fd[PADDED_REVIEWS_TENSOR_NAME],
                                              fd[PADDED_LABELS_TENSOR_NAME], masked_prediction, aspect_word_index_map)
            per_aspect_sentiments_cm += eval_results['per_aspect_sentiments_cm']
            per_aspect_aspect_detection_cm += eval_results['per_aspect_aspect_detection_cm']
            joint_aspect_sentiment_cm += eval_results['joint_aspect_sentiment_cm']
            n_multilabel_success += eval_results['n_multilabel_success']
            n_multilabel_failure += eval_results['n_multilabel_failure']
            n_sentence += eval_results['count']
            test_metrics = {
                'per_aspect_sentiments_cm': per_aspect_sentiments_cm,
                'per_aspect_aspect_detection_cm': per_aspect_aspect_detection_cm,
                'joint_aspect_sentiment_cm': joint_aspect_sentiment_cm,
                'n_multilabel_success': n_multilabel_success,
                'n_multilabel_failure': n_multilabel_failure,
                'n_sentence': n_sentence
            }

    df = {'loss': losses,
          'accuracy': accuracies,
          'f1_score': f1_scores,
          'confusion_matrix': cm,
          'masked_predictions': masked_predictions,
          'aspects': aspects,
          'ground_truths': ground_truths,
          'masks': masks,
          'test_metrics': test_metrics}
    return df


def test(best_step):
    tf.reset_default_graph()
    config = tf.ConfigProto(allow_soft_placement = True)
    if args.mode == 'train':
        args.load_graph = True
        args.meta_file_path = EXPERIMENT_PREFIX + '-' + str(best_step) + '.meta'
        args.restore_dir = EXPERIMENT_NAME
    with tf.Session(config = config) as session:
        _, _, ops = model.get_model(session = session,
                                    args = args,
                                    restore_only = True)
        test_results = evaluate(session = session,
                                ops = ops,
                                dataset = test_loader(epochs = 1))

    print_results(test_results, args, msg = 'TEST RESULTS')
    print_test_evaluation_metrics(test_results['test_metrics'])
    if args.aspect_detector:
        compute_aspect_detection_results(test_results, RESTAURANT_ASPECT_WORD_INDEX_MAP)


def main():
    if args.mode == 'train':
        best_step = train()
        test(best_step)
    elif args.mode == 'test':
        test(0)


def tune_hyperparameters():
    # https://ray.readthedocs.io/en/latest/tune.html
    ray.init()
    trials = tune.run_experiments({
        'hyperparamter-tuning': {
            "run": train,
            "stop": {'f1_score': .99},
            "config": {'learning_rate': tune.grid_search([0.0001, 0.001, 0.01]),
                       'batch_size': tune.grid_search([32, 64, 128]),
                       'dropout_keep_prob': tune.grid_search([0.2, 0.3, 0.4, 0.5])
                       }
        }
    })


if __name__ == '__main__':
    if args.hyperparam_tune:
        tune_hyperparameters()
    else:
        main()
