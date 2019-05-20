import os
import sys

sys.path.append(os.getcwd())
from config.settings import WORD_FREQ_FILE, VOCAB_TO_CODE_FILE, CODE_TO_VOCAB_FILE
import numpy as np
from utils.data_util import read_binary, write_binary
from utils.util import space_separated_token_string, save_sentences_to_text, remove_duplicate_sentences
import spacy
from collections import defaultdict
import random
import re
from preprocessing.semEval16_data_processing import RESTAURANT_ASPECT_WORDS, LAPTOPS_ASPECT_WORDS

# ---SCRIPT DEPENDENCIES----
# python -m spacy download en
# --------------------------

# -----CHANGE THESE VALUES ACCORDINGLY BEFORE RUNNING THE SCRIPT-----
TYPE = 'train'
# TYPE = 'val'
# TYPE = 'test'
FILE_NAME = 'restaurant'
# FILE_NAME = 'laptops'
# -------------------------------------------------------------------
FORMATTED_FILE_NAME = 'formatted_' + FILE_NAME + '_' + TYPE + '.pickle'
PROCESSED_FILE_NAME = 'processed_' + TYPE + '.pickle'

MAX_VOCAB_SIZE = 50001
PAD = 0
NLP = spacy.load('en_core_web_sm')


def build_word_frequency_distribution():
    """
    1. Extract tokens from the review text
    2. Calculate frequency of each token
    3. Create a freq dict and store it in a file

    :return: A dict of <token, freq>
    """
    try:
        freq_dist_f = read_binary(WORD_FREQ_FILE)
        print('frequency distribution loaded')
        return freq_dist_f
    except IOError:
        pass

    print('building frequency distribution')
    freq = defaultdict(int)
    if FILE_NAME == 'restaurant':
        for aspect_word in RESTAURANT_ASPECT_WORDS:
            freq[aspect_word] += 1
    elif FILE_NAME == 'laptops':
        for aspect_word in LAPTOPS_ASPECT_WORDS:
            freq[aspect_word] += 1

    for i, review in enumerate(read_binary(FORMATTED_FILE_NAME)):
        sentences = review[1]

        for sent in sentences:
            tokens = NLP.tokenizer(sent[0])
            for token in tokens:
                freq[token.orth_] += 1
            if i % 100 == 0:
                write_binary(freq, WORD_FREQ_FILE)
                print('dump at {}'.format(i))
        write_binary(freq, WORD_FREQ_FILE)
    return freq


def build_vocabulary(lower = 1, n = MAX_VOCAB_SIZE):
    """
    1. Get word frequency distribution
    2. Sort is based on word frequencies
    3. Make a vocab dist using the most frequent words
    4. Store vocab dist in a file in format <word, identifier>

    :param lower: Identifiers below this are reserved
    :param n: Number of unique expected words
    :return: A dict of vocabulary words and an assigned identifier
    """

    try:
        vocab_to_code = read_binary(VOCAB_TO_CODE_FILE)
        code_to_vocab = read_binary(CODE_TO_VOCAB_FILE)
        print('vocabulary loaded')
        return vocab_to_code, code_to_vocab
    except IOError:
        print('building vocabulary')
    freq = build_word_frequency_distribution()

    # sorting words in ascending order based on frequency and then pick top n words
    top_words = list(sorted(freq.items(), key = lambda x: -x[1]))[:n - lower + 1]
    # create optimum vocab size
    print('Vocab count : ' + str(len(top_words)))
    # global MAX_VOCAB_SIZE
    # global UNKNOWN
    max_vocab_size = len(top_words) + 2
    unknown = max_vocab_size - 1
    vocab_to_code = {}
    code_to_vocab = {}

    vocab_to_code['<UNK>'] = unknown
    code_to_vocab[unknown] = '<UNK>'
    vocab_to_code['<PAD>'] = PAD
    code_to_vocab[PAD] = '<PAD>'

    # lower vocab indexes are reserved for padding and unknown words
    i = lower
    for w, freq in top_words:
        vocab_to_code[w] = i
        code_to_vocab[i] = w
        i += 1
    write_binary(vocab_to_code, VOCAB_TO_CODE_FILE)
    write_binary(code_to_vocab, CODE_TO_VOCAB_FILE)
    return vocab_to_code, code_to_vocab


def get_uncoded_data(code_to_vocab, datapoint):
    aspect_words = []
    review = []
    aspect_codes = datapoint[0]
    coded_sentences = datapoint[1]
    polarities = datapoint[2]

    for aspect_code in aspect_codes:
        aspect_words.append(code_to_vocab.get(aspect_code))

    for sentence in coded_sentences:
        sent_words = []
        for coded_word in sentence:
            sent_words.append(code_to_vocab.get(coded_word))
        review.append(sent_words)
    x = [aspect_words, review, polarities]
    return x


def process_data():
    vocab_to_code, code_to_vocab = build_vocabulary()
    max_vocab_size = len(vocab_to_code)
    print('Final Vocab Size : ' + str(max_vocab_size))
    try:
        tokenized_dataset = []
        all_sentences = []
        for i, review in enumerate(read_binary(FORMATTED_FILE_NAME)):
            tokenized_aspect = []
            tokenized_sentences = []

            if i == 0:
                print(review)

            sentences = review[1]
            aspect_words = review[0]
            polarities = review[2]

            for aspect_word in aspect_words:
                tokenized_aspect.append(aspect_word)
                all_sentences.append([aspect_word])

            for sent in sentences:
                tokenized_sentence = []

                # remove duplicate spaces from the sentence. This is causing problem for elmo.
                s = re.sub(' +', ' ', sent[0])

                tokens = NLP.tokenizer(s)
                for token in tokens:
                    tokenized_sentence.append(token.orth_)
                tokenized_sentences.append(tokenized_sentence)

                # all these sentences will be written to a separate txt file at the end of the process.
                all_sentences.append(tokenized_sentence)

            tokenized_review = [tokenized_aspect, tokenized_sentences, polarities]

            # dataset
            tokenized_dataset.append(tokenized_review)
            write_binary(tokenized_dataset, PROCESSED_FILE_NAME)
            print('dump at {}'.format(i))

        all_sentences = space_separated_token_string(all_sentences)
        save_sentences_to_text(all_sentences)
        # hack for elmo
        remove_duplicate_sentences()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    process_data()
