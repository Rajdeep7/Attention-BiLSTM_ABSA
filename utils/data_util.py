import xmltodict
import ujson as json
import pandas as pd
import os
import pickle
import random
from config.settings import DATA_DIR, TRAINING_FILE, VALIDATION_FILE, TESTING_FILE


def read_text(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            yield line


def read_xml(file_path):
    doc = {}
    with open(file_path, 'rb') as f:
        doc = xmltodict.parse(f.read())
    return doc


def write_xml(file_path, data):
    with open(file_path, 'w') as f:
        xmltodict.unparse(input_dict = data, output = f, pretty = True)


def read_json(file_path):
    with open(file_path, 'rb') as f:
        for line in f:
            yield json.loads(line)


def read_line(file_path):
    with open(file_path, 'rb') as f:
        for line in f:
            yield str(line).split("\\n")[0]
            # return pd.read_csv(file_path, sep = ',')
            # return np.loadtxt(file_path, dtype=string)


def read_csv(file_path, separator = ','):
    df = pd.read_csv(file_path, sep = separator, quotechar = '"')
    # print(df.head(5))
    return df


def write_binary(data, filename):
    for dir in [DATA_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'wb') as data_f:
        pickle.dump(data, data_f)


def append_binary(data, filename):
    for dir in [DATA_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'ab') as data_f:
        pickle.dump(data, data_f)


def read_binary(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'rb') as data_f:
        data = pickle.load(data_f)
    return data


def _read_dataset(fn, epochs = 1, shuffle = True):
    """
    The yield statement suspends function’s execution and sends a value back to caller,
    but retains enough state to enable function to resume where it is left off. When resumed,
    the function continues execution immediately after the last yield run.
    This allows its code to produce a series of values over time, rather them computing
    them at once and sending them back like a list.

    x: One review document consisting of many sentences. Each sentence consists of many words and each word is
       represented as an integer code from the vocabulary.
    :param fn:
    :param epochs: This is used to specify maximum number of times yield can be called on this function.
    :return:
    """
    c = 0
    while 1:
        c += 1
        if epochs > 0 and c > epochs:
            return
        print('epoch %s' % c)
        dataset = read_binary(filename = fn)
        if shuffle:
            shuffle_data(dataset)
        else:
            print('not shuffling test data..')
        for datapoint in dataset:
            aspect_words = datapoint[0]
            reviews = datapoint[1]
            labels = datapoint[2]
            yield aspect_words, reviews, labels


def train_loader(epochs = 1):
    return _read_dataset(TRAINING_FILE, epochs = epochs, shuffle = True)


def val_loader(epochs = 1):
    return _read_dataset(VALIDATION_FILE, epochs = epochs, shuffle = True)


def test_loader(epochs = 1):
    return _read_dataset(TESTING_FILE, epochs = epochs, shuffle = False)


def shuffle_data(dataset):
    random.shuffle(dataset)
