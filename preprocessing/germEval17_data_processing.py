import os
import sys

sys.path.append(os.getcwd())
from config.settings import WORD_FREQ_FILE, VOCAB_TO_CODE_FILE, CODE_TO_VOCAB_FILE, CODE_TO_EMBED_FILE
import numpy as np
from utils.data_util import read_binary, write_binary, append_binary
from utils.util import load_glove_embeddings, load_fastText_embeddings, load_oov_fastText_embeddings
import spacy
from collections import defaultdict
import random

# ---SCRIPT DEPENDENCIES----
# python -m spacy download de_core_news_sm
# --------------------------

# -----CHANGE THESE VALUES ACCORDINGLY BEFORE RUNNING THE SCRIPT-----
TYPE = 'train'
# TYPE = 'test'
# TYPE = 'val'
FILE_NAME = 'ubahn'
EMBEDDING_TYPE = 'fasttext'
CONCATENATE_EMBEDDING = True
# -------------------------------------------------------------------
FORMATTED_FILE_NAME = 'formatted_' + FILE_NAME + '_' + TYPE + '.pickle'
PROCESSED_FILE_NAME = 'processed_' + TYPE + '.pickle'

EMBEDDING_DIMENSION = 400
MAX_VOCAB_SIZE = 100000
UNKNOWN_EMBEDDING = np.random.randn(EMBEDDING_DIMENSION)
PAD = 0
PAD_EMBEDDING = np.zeros(EMBEDDING_DIMENSION)
NLP = spacy.load('de_core_news_sm')

ASPECT_WORDS = ['Allgemein', 'Atmosphäre', 'Connectivity', 'Design', 'Gastronomisches_Angebot', 'Informationen',
                'DB_App_und_Website', 'Service_und_Kundenbetreuung',
                'Komfort_und_Ausstattung', 'Gepäck', 'Auslastung_und_Platzangebot', 'Ticketkauf', 'Toiletten',
                'Zugfahrt', 'Reisen_mit_Kindern', 'Image', 'QR-Code',
                'Barrierefreiheit', 'Sicherheit', 'Sonstige_Unregelmässigkeiten', 'allgemein', 'atmosphäre',
                'connectivity',
                'design', 'gastronomisches_angebot', 'informationen',
                'db_app_und_website', 'service_und_kundenbetreuung', 'komfort_und_ausstattung',
                'gepäck',
                'auslastung_und_platzangebot',
                'ticketkauf', 'toiletten', 'zugfahrt', 'reisen_mit_kindern', 'image', 'qr-code',
                'barrierefreiheit', 'sicherheit', 'sonstige_unregelmässigkeiten',
                'db', 'app', 'website', 'service', 'kundenbetreuung', 'komfort', 'ausstattung', 'auslastung',
                'platzangebot', 'reisen', 'kindern', 'angebot', 'gastronomisches', 'sonstige', 'unregelmässigkeiten']

GERMEVAL_ASPECT_WORD_INDEX_MAP = {
    'allgemein': 0,
    'atmosphäre': 1,
    'connectivity': 2,
    'design': 3,
    'gastronomischesangebot': 4,
    'informationen': 5,
    'dbappwebsite': 6,
    'servicekundenbetreuung': 7,
    'komfortausstattung': 8,
    'gepäck': 9,
    'auslastungplatzangebot': 10,
    'ticketkauf': 11,
    'toiletten': 12,
    'zugfahrt': 13,
    'reisenmitkindern': 14,
    'image': 15,
    'qr-code': 16,
    'barrierefreiheit': 17,
    'sicherheit': 18,
    'sonstigeunregelmässigkeiten': 19,
    'none': 20
}

GERMEVAL_INDEX_TO_ASPECT_WORD_MAP = {
    0: 'Allgemein',
    1: 'Atmosphäre',
    2: 'Connectivity',
    3: 'Design',
    4: 'Gastronomisches_Angebot',
    5: 'Informationen',
    6: 'DB_App_und_Website',
    7: 'Service_und_Kundenbetreuung',
    8: 'Komfort_und_Ausstattung',
    9: 'Gepäck',
    10: 'Auslastung_und_Platzangebot',
    11: 'Ticketkauf',
    12: 'Toiletten',
    13: 'Zugfahrt',
    14: 'Reisen_mit_Kindern',
    15: 'Image',
    16: 'QR-Code',
    17: 'Barrierefreiheit',
    18: 'Sicherheit',
    19: 'Sonstige_Unregelmässigkeiten',
    20: 'none'
}

GERMEVAL_INDEX_TO_ASPECT_SENTIMENT_WORD_MAP = {
    0: 'Allgemein_positive',
    1: 'Allgemein_negative',
    2: 'Allgemein_neutral',
    3: 'Atmosphäre_positive',
    4: 'Atmosphäre_negative',
    5: 'Atmosphäre_neutral',
    6: 'Connectivity_positive',
    7: 'Connectivity_negative',
    8: 'Connectivity_neutral',
    9: 'Design_positive',
    10: 'Design_negative',
    11: 'Design_neutral',
    12: 'Gastronomisches_Angebot_positive',
    13: 'Gastronomisches_Angebot_negative',
    14: 'Gastronomisches_Angebot_neutral',
    15: 'Informationen_positive',
    16: 'Informationen_negative',
    17: 'Informationen_neutral',
    18: 'DB_App_und_Website_positive',
    19: 'DB_App_und_Website_negative',
    20: 'DB_App_und_Website_neutral',
    21: 'Service_und_Kundenbetreuung_positive',
    22: 'Service_und_Kundenbetreuung_negative',
    23: 'Service_und_Kundenbetreuung_neutral',
    24: 'Komfort_und_Ausstattung_positive',
    25: 'Komfort_und_Ausstattung_negative',
    26: 'Komfort_und_Ausstattung_neutral',
    27: 'Gepäck_positive',
    28: 'Gepäck_negative',
    29: 'Gepäck_neutral',
    30: 'Auslastung_und_Platzangebot_positive',
    31: 'Auslastung_und_Platzangebot_negative',
    32: 'Auslastung_und_Platzangebot_neutral',
    33: 'Ticketkauf_positive',
    34: 'Ticketkauf_negative',
    35: 'Ticketkauf_neutral',
    36: 'Toiletten_positive',
    37: 'Toiletten_negative',
    38: 'Toiletten_neutral',
    39: 'Zugfahrt_positive',
    40: 'Zugfahrt_negative',
    41: 'Zugfahrt_neutral',
    42: 'Reisen_mit_Kindern_positive',
    43: 'Reisen_mit_Kindern_negative',
    44: 'Reisen_mit_Kindern_neutral',
    45: 'Image_positive',
    46: 'Image_negative',
    47: 'Image_neutral',
    48: 'QR-Code_positive',
    49: 'QR-Code_negative',
    50: 'QR-Code_neutral',
    51: 'Barrierefreiheit_positive',
    52: 'Barrierefreiheit_negative',
    53: 'Barrierefreiheit_neutral',
    54: 'Sicherheit_positive',
    55: 'Sicherheit_negative',
    56: 'Sicherheit_neutral',
    57: 'Sonstige_Unregelmässigkeiten_positive',
    58: 'Sonstige_Unregelmässigkeiten_negative',
    59: 'Sonstige_Unregelmässigkeiten_neutral',
    60: 'none'
}


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
    for aspect_word in ASPECT_WORDS:
        freq[aspect_word] += 1

    files = [FORMATTED_FILE_NAME]
    if EMBEDDING_TYPE == 'fasttext':
        files.append(FORMATTED_FILE_NAME.replace('train', 'test'))
        files.append(FORMATTED_FILE_NAME.replace('train', 'val'))

    for file_path in files:
        print('building vocab from file - ' + file_path)
        for i, review in enumerate(read_binary(file_path)):
            tokenized_text = review[1]

            for token in tokenized_text:
                freq[token] += 1
                if i % 100 == 0:
                    write_binary(freq, WORD_FREQ_FILE)
                    # print('dump at {}'.format(i))
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

    # get glove embeddings
    print('loading embeddings')
    if EMBEDDING_TYPE == 'glove':
        word_to_embeddings = load_glove_embeddings()
    elif EMBEDDING_TYPE == 'fasttext':
        # word_to_embeddings = load_fastText_embeddings(lang = 'de')
        word_to_embeddings = load_oov_fastText_embeddings(lang = 'de')
        if CONCATENATE_EMBEDDING:
            word_to_embeddings_self_trained = load_oov_fastText_embeddings(lang = 'de', self_trained = True)
    else:
        word_to_embeddings = {}

    # sorting words in ascending order based on frequency and then pick top n words
    top_words = list(sorted(freq.items(), key = lambda x: -x[1]))[:n - lower + 1]
    # create optimum vocab size
    print('Vocab count : ' + str(len(top_words)))
    MAX_VOCAB_SIZE = len(top_words) + 2
    UNKNOWN = MAX_VOCAB_SIZE - 1
    vocab_to_code = {}
    code_to_vocab = {}

    # an array of embeddings with index referring to the vocab code. First and last index is
    # reserved for padding and unknown words respectively.
    code_to_embed = np.zeros(shape = (MAX_VOCAB_SIZE, EMBEDDING_DIMENSION), dtype = np.float32)
    code_to_embed[PAD] = PAD_EMBEDDING
    code_to_embed[UNKNOWN] = UNKNOWN_EMBEDDING
    vocab_to_code['<UNK>'] = UNKNOWN
    code_to_vocab[UNKNOWN] = '<UNK>'
    vocab_to_code['<PAD>'] = PAD
    code_to_vocab[PAD] = '<PAD>'

    # lower vocab indexes are reserved for padding and unknown words
    i = lower
    unknow_count = 0
    for w, freq in top_words:
        vocab_to_code[w] = i
        code_to_vocab[i] = w
        try:
            if EMBEDDING_TYPE == 'glove':
                embedding = word_to_embeddings.word_vec(w)
            elif EMBEDDING_TYPE == 'fasttext':
                embedding = word_to_embeddings.get_word_vector(w)
                if CONCATENATE_EMBEDDING:
                    self_trained_emebdding = word_to_embeddings_self_trained.get_word_vector(w)
                    embedding = np.concatenate((embedding, self_trained_emebdding), axis = 0)

        except KeyError:
            embedding = UNKNOWN_EMBEDDING
            unknow_count += 1
            print('unknown embeddings ', w)
        code_to_embed[i] = embedding
        i += 1
    print('unknown_embedding_count ', str(unknow_count))
    write_binary(vocab_to_code, VOCAB_TO_CODE_FILE)
    write_binary(code_to_vocab, CODE_TO_VOCAB_FILE)
    write_binary(code_to_embed, CODE_TO_EMBED_FILE)
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
    vocab_size = len(vocab_to_code)
    unknown = vocab_size - 1
    print('Final Vocab Size : ' + str(vocab_size))
    coded_dataset = []
    for i, review in enumerate(read_binary(FORMATTED_FILE_NAME)):
        coded_aspect = []
        coded_text = []

        if i == 0:
            print(review)

        text = review[1]
        aspect_words = review[0]
        polarity = review[2]

        for aspect_word in aspect_words:
            a = vocab_to_code.get(aspect_word, unknown)
            if a == unknown:
                print('STOP')
                print(aspect_word)
            coded_aspect.append(a)

        for word in text:
            word_code = vocab_to_code.get(word, unknown)
            coded_text.append(word_code)

        coded_review = [coded_aspect, [coded_text], [polarity]]
        coded_dataset.append(coded_review)
        write_binary(coded_dataset, PROCESSED_FILE_NAME)
        print('dump at {}'.format(i))

    datapoint = coded_dataset[0]
    print(datapoint)
    print(get_uncoded_data(code_to_vocab, datapoint))


if __name__ == '__main__':
    process_data()
