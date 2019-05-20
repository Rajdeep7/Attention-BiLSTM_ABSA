import os
import datetime

EXPERIMENT_TIME = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
# EXPERIMENT_PREFIX = 'semeval16-restaurant'
# EXPERIMENT_PREFIX = 'semeval16-laptops'
# EXPERIMENT_PREFIX = 'combined_data'
# EXPERIMENT_PREFIX = 'all_combined_data'
# EXPERIMENT_PREFIX = 'all_combined_organic_reduced_data'
# EXPERIMENT_PREFIX = 'germeval17'
# EXPERIMENT_PREFIX = 'organic'
EXPERIMENT_PREFIX = 'organic_reduced'
EXPERIMENT_NAME = EXPERIMENT_PREFIX + '-' + str(EXPERIMENT_TIME)

PROJECT_ROOT_DIR = os.getcwd()
BASE_DATA_DIR = os.path.join(PROJECT_ROOT_DIR, 'processed_data')
BASE_CHECKPOINT_DIR = os.path.join(PROJECT_ROOT_DIR, 'checkpoints')
BASE_TFLOG_DIR = os.path.join(PROJECT_ROOT_DIR, 'tflog')

DATA_DIR = os.path.join(BASE_DATA_DIR, EXPERIMENT_PREFIX)
CHECKPOINT_DIR = os.path.join(BASE_CHECKPOINT_DIR, EXPERIMENT_NAME)
TFLOG_DIR = os.path.join(BASE_TFLOG_DIR, EXPERIMENT_NAME)

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, EXPERIMENT_PREFIX)
TRAINING_FILE = 'processed_train.pickle'
VALIDATION_FILE = 'processed_val.pickle'
TESTING_FILE = 'processed_test.pickle'
MAX_POSSIBLE_EPOCHS = 300
VOCAB_TO_CODE_FILE = 'vocab_to_code.pickle'
CODE_TO_VOCAB_FILE = 'code_to_vocab.pickle'
CODE_TO_EMBED_FILE = 'code_to_embed.pickle'
WORD_FREQ_FILE = 'word_freq.pickle'
TINY = 0.0001
GLOVE_EMBEDDINGS_FILE = os.path.join(*[PROJECT_ROOT_DIR, 'dataset', 'glove.840B.300d.txt'])
TMP_GLOVE_EMBEDDINGS_FILE = os.path.join(*[PROJECT_ROOT_DIR, 'dataset', 'test_word2vec.txt'])
FASTTEXT_EN_EMBEDDINGS_FILE = os.path.join(*[PROJECT_ROOT_DIR, 'dataset', 'crawl-300d-2M-subword.vec'])
FASTTEXT_EN_EMBEDDINGS_MODEL = os.path.join(*[PROJECT_ROOT_DIR, 'dataset', 'cc.en.300.bin'])
FASTTEXT_DE_EMBEDDINGS_FILE = os.path.join(*[PROJECT_ROOT_DIR, 'dataset', 'wiki.de.vec'])
FASTTEXT_DE_EMBEDDINGS_MODEL = os.path.join(*[PROJECT_ROOT_DIR, 'dataset', 'wiki.de.bin'])
SELF_TRAINED_FASTTEXT_DE_EMBEDDINGS_MODEL = os.path.join(*[PROJECT_ROOT_DIR, 'dataset', 'germeval_100.de.bin'])
ALL_SENTENCES_TEXT = os.path.join(DATA_DIR, 'all_sentences.txt')
ELMO_EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'elmo_embeddings.hdf5')