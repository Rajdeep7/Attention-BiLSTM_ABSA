import os
import sys

sys.path.append(os.getcwd())
from utils.data_util import read_xml, read_text

GERMEVAL_TRAIN_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'germEval17_train_v1.4.xml'])
DB_TWEETS_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'de-crawled-tweets-db.txt'])
OUTPUT_FILE_NAME = os.path.join(*[os.path.curdir, 'dataset', 'combined_db_tweets_and_germeval_train.txt'])


def combine_db_tweets_and_germeval_train_data():
    doc = read_xml(GERMEVAL_TRAIN_FILE_PATH)
    with open(OUTPUT_FILE_NAME, 'w') as f:
        for i, review in enumerate(doc['Documents']['Document']):
            print('document-' + str(i))
            text = review['text']
            f.write(text + '\n')

        for j, line in enumerate(read_text(DB_TWEETS_FILE_PATH)):
            print('tweet-' + str(j))
            f.write(line)


if __name__ == '__main__':
    combine_db_tweets_and_germeval_train_data()
