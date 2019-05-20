import os
import sys

sys.path.append(os.getcwd())
import fastText


def unsupervised_trainer_for_germeval():
    """
    This method tries to train fasttext on db+germeval combined data
    :return:
    """
    input_file = os.path.join(*[os.path.curdir, 'dataset', 'combined_db_tweets_and_germeval_train.txt'])
    output_file = os.path.join(*[os.path.curdir, 'dataset', 'germeval_100.de.bin'])
    # with open(input_file, 'r') as f:
    #     for line in f:
    #         print(fastText.tokenize(line))
    germeval_trained = fastText.train_unsupervised(input = input_file, lr = .05, dim = 100, epoch = 100)
    germeval_trained.save_model(path = output_file)
    # print(germeval_trained.get_words(include_freq = True))


if __name__ == '__main__':
    unsupervised_trainer_for_germeval()
