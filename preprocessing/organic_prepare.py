import sys
import os

sys.path.append(os.getcwd())
from utils.data_util import read_xml, write_binary
from utils.util import load_glove_embeddings, get_similar_words
import spacy
import nltk
import numpy as np
import random

nltk.download('wordnet')
from nltk.corpus import wordnet

# -----CHANGE THESE VALUES ACCORDINGLY BEFORE RUNNING THE SCRIPT----
DATA_TYPE = 'organic_reduced'

TYPE = 'train'
INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'organic_train_coarse.xml'])
# TYPE = 'val'
# INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'organic_val_coarse.xml'])
# TYPE = 'test'
# INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'organic_test_coarse.xml'])

# DATA_TYPE = 'organic'

# TYPE = 'train'
# INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'organic_train_fine.xml'])
# TYPE = 'val'
# INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'organic_val_fine.xml'])
# TYPE = 'test'
# INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'organic_test_fine.xml'])

OVERSAMPLING = False
PARAPHRASING = False
REDUCED = False
MATCH_TYPE = 'similar'
# MATCH_TYPE = 'synonym'
MAX_MATCHED_WORDS = 3
MINORITY_CLASS = [2]
if PARAPHRASING and MATCH_TYPE == 'similar':
    embedding = load_glove_embeddings()
# -------------------------------------------------------------------

TOTAL_SENTENCE_COUNT = 0
TOTAL_REVIEW_COUNT = 0
TOTAL_AUGMENTED_REVIEW_COUNT = 0
TOTAL_POSITIVE_LABEL_COUNT = 0
TOTAL_NEGATIVE_LABEL_COUNT = 0
TOTAL_NEUTRAL_LABEL_COUNT = 0
TOTAL_NOT_APPLICABLE_LABEL_COUNT = 0
ASPECT_TO_SENTENCE_FREQUENCY = {}

NLP = spacy.load('en_core_web_sm')


def get_one_hot_encoded_sentiment(sentiment):
    p = [0, 0, 0, 0]
    global TOTAL_POSITIVE_LABEL_COUNT
    global TOTAL_NEGATIVE_LABEL_COUNT
    global TOTAL_NEUTRAL_LABEL_COUNT

    if sentiment == 'positive':
        p[0] = 1
        TOTAL_POSITIVE_LABEL_COUNT += 1
    elif sentiment == 'negative':
        p[1] = 1
        TOTAL_NEGATIVE_LABEL_COUNT += 1
    elif sentiment == 'neutral':
        p[2] = 1
        TOTAL_NEUTRAL_LABEL_COUNT += 1
    else:
        p[3] = 1
    return p


def get_categorical_sentiment(sentiment):
    return np.argmax(get_one_hot_encoded_sentiment(sentiment))


def update_aspect_to_sentence_frequency(aspect):
    global ASPECT_TO_SENTENCE_FREQUENCY
    count = ASPECT_TO_SENTENCE_FREQUENCY.get(aspect)
    if count:
        ASPECT_TO_SENTENCE_FREQUENCY[aspect] = count + 1
    else:
        ASPECT_TO_SENTENCE_FREQUENCY[aspect] = 1


def make_flatten_restaurant_data_sentence_level(reviews, mode = 'train'):
    # we have 9 entities, 14 attributes so total 126 possible aspects
    # but in training data we have only 111 aspects present. In total we selected 111 aspects based our understanding of
    # which entity-attribute pair makes sense.
    organic_possible_aspects = ['organic_general#general',
                                'organic_general#price',
                                'organic_general#taste',
                                'organic_general#nutritional_quality_freshness_appearance',
                                'organic_general#safety',
                                'organic_general#healthiness',
                                'organic_general#chemicals_pesticides',
                                'organic_general#label',
                                'organic_general#origin_source',
                                'organic_general#local',
                                'organic_general#availability',
                                'organic_general#environment',
                                'organic_general#animal_welfare',
                                'organic_general#productivity',
                                'organic_products#general',
                                'organic_products#price',
                                'organic_products#taste',
                                'organic_products#nutritional_quality_freshness_appearance',
                                'organic_products#safety',
                                'organic_products#healthiness',
                                'organic_products#chemicals_pesticides',
                                'organic_products#label',
                                'organic_products#origin_source',
                                'organic_products#local',
                                'organic_products#availability',
                                'organic_products#environment',
                                'organic_products#animal_welfare',
                                'organic_products#productivity',
                                'organic_farming#general',
                                'organic_farming#price',
                                'organic_farming#taste',
                                'organic_farming#nutritional_quality_freshness_appearance',
                                'organic_farming#safety',
                                'organic_farming#healthiness',
                                'organic_farming#chemicals_pesticides',
                                'organic_farming#label',
                                'organic_farming#origin_source',
                                'organic_farming#local',
                                'organic_farming#availability',
                                'organic_farming#environment',
                                'organic_farming#animal_welfare',
                                'organic_farming#productivity',
                                'organic_companies#general',
                                'organic_companies#price',
                                'organic_companies#taste',
                                'organic_companies#nutritional_quality_freshness_appearance',
                                'organic_companies#safety',
                                'organic_companies#healthiness',
                                'organic_companies#chemicals_pesticides',
                                'organic_companies#label',
                                'organic_companies#origin_source',
                                'organic_companies#local',
                                'organic_companies#availability',
                                'organic_companies#environment',
                                'organic_companies#animal_welfare',
                                'organic_companies#productivity',
                                'conventional_general#general',
                                'conventional_general#price',
                                'conventional_general#nutritional_quality_freshness_appearance',
                                'conventional_general#safety',
                                'conventional_general#healthiness',
                                'conventional_general#chemicals_pesticides',
                                'conventional_general#label',
                                'conventional_general#origin_source',
                                'conventional_general#productivity',
                                'conventional_products#general',
                                'conventional_products#price',
                                'conventional_products#taste',
                                'conventional_products#nutritional_quality_freshness_appearance',
                                'conventional_products#safety',
                                'conventional_products#healthiness',
                                'conventional_products#chemicals_pesticides',
                                'conventional_products#label',
                                'conventional_products#origin_source',
                                'conventional_products#local',
                                'conventional_products#availability',
                                'conventional_products#environment',
                                'conventional_products#animal_welfare',
                                'conventional_products#productivity',
                                'conventional_farming#general',
                                'conventional_farming#price',
                                'conventional_farming#taste',
                                'conventional_farming#nutritional_quality_freshness_appearance',
                                'conventional_farming#safety',
                                'conventional_farming#healthiness',
                                'conventional_farming#chemicals_pesticides',
                                'conventional_farming#label',
                                'conventional_farming#origin_source',
                                'conventional_farming#environment',
                                'conventional_farming#animal_welfare',
                                'conventional_farming#productivity',
                                'conventional_companies#general',
                                'conventional_companies#taste',
                                'conventional_companies#safety',
                                'conventional_companies#chemicals_pesticides',
                                'conventional_companies#label',
                                'conventional_companies#availability',
                                'conventional_companies#environment',
                                'conventional_companies#animal_welfare',
                                'conventional_companies#productivity',
                                'gmo_genetic_engineering#general',
                                'gmo_genetic_engineering#price',
                                'gmo_genetic_engineering#taste',
                                'gmo_genetic_engineering#nutritional_quality_freshness_appearance',
                                'gmo_genetic_engineering#safety',
                                'gmo_genetic_engineering#healthiness',
                                'gmo_genetic_engineering#chemicals_pesticides',
                                'gmo_genetic_engineering#label',
                                'gmo_genetic_engineering#origin_source',
                                'gmo_genetic_engineering#environment',
                                'gmo_genetic_engineering#productivity'
                                ]

    reduced_organic_possible_aspects = ['organic#general',
                                        'organic#price',
                                        'organic#quality',
                                        'organic#safety_healthiness',
                                        'organic#trustworthy_sources',
                                        'organic#environment',
                                        'conventional#general',
                                        'conventional#price',
                                        'conventional#quality',
                                        'conventional#safety_healthiness',
                                        'conventional#trustworthy_sources',
                                        'conventional#environment',
                                        'gmo_genetic_engineering#general',
                                        'gmo_genetic_engineering#price',
                                        'gmo_genetic_engineering#quality',
                                        'gmo_genetic_engineering#safety_healthiness',
                                        'gmo_genetic_engineering#trustworthy_sources',
                                        'gmo_genetic_engineering#environment']

    global TOTAL_SENTENCE_COUNT
    global TOTAL_REVIEW_COUNT
    global TOTAL_AUGMENTED_REVIEW_COUNT
    global TOTAL_POSITIVE_LABEL_COUNT
    global TOTAL_NEGATIVE_LABEL_COUNT
    global TOTAL_NEUTRAL_LABEL_COUNT
    global TOTAL_NOT_APPLICABLE_LABEL_COUNT
    global ASPECT_TO_SENTENCE_FREQUENCY
    global DATA_TYPE

    if DATA_TYPE == 'organic':
        possible_aspects = organic_possible_aspects
    elif DATA_TYPE == 'organic_reduced':
        possible_aspects = reduced_organic_possible_aspects

    TOTAL_SENTENCE_COUNT = 0
    TOTAL_REVIEW_COUNT = 0
    TOTAL_AUGMENTED_REVIEW_COUNT = 0
    TOTAL_POSITIVE_LABEL_COUNT = 0
    TOTAL_NEGATIVE_LABEL_COUNT = 0
    TOTAL_NEUTRAL_LABEL_COUNT = 0
    TOTAL_NOT_APPLICABLE_LABEL_COUNT = 0
    ASPECT_TO_SENTENCE_FREQUENCY = {}

    dataset = []
    for i, review in enumerate(reviews):
        TOTAL_REVIEW_COUNT += 1
        print('review-' + str(i))

        review_text = []
        aspect_sentence_polarity_map = {}
        sentences = review['sentences']['sentence']
        if isinstance(sentences, dict):
            sentences = [sentences]
        for j, sentence in enumerate(sentences):
            TOTAL_SENTENCE_COUNT += 1
            sentence_text = []
            sentence_text.append(sentence['text'])
            if 'Opinions' in sentence.keys():
                opinions = sentence['Opinions']['Opinion']
                if isinstance(opinions, dict):
                    opinions = [opinions]

                for opinion in opinions:
                    aspect_category = opinion['@category'].lower()
                    update_aspect_to_sentence_frequency(aspect_category)
                    polarity = get_categorical_sentiment(opinion['@polarity'])

                    # Here we are trying to create a map of sentences and aspects. Basicly, for the current review which
                    # sentence is related to which aspect.
                    sentence_polarity = aspect_sentence_polarity_map.get(aspect_category, [])
                    sentence_polarity.append([j, polarity])
                    aspect_sentence_polarity_map[aspect_category] = sentence_polarity
            # else:
            #     # no aspect, contains no sentiment, either out of domain or just some fact
            #     sentence_polarity = aspect_sentence_polarity_map.get('relevance', [])
            #     sentence_polarity.append([j, 3])
            #     aspect_sentence_polarity_map['relevance'] = sentence_polarity

            review_text.append(sentence_text)

        # It could be that a particular review has no sentence for some aspects. Here we are just adding an empty
        # sentence list for such aspects.
        if not REDUCED:
            for aspect in possible_aspects:
                if aspect not in aspect_sentence_polarity_map.keys():
                    aspect_sentence_polarity_map[aspect] = []

        # Now for every possible aspect we will create a datapoint using this particular review.
        for a, sent_polarities in aspect_sentence_polarity_map.items():
            TOTAL_AUGMENTED_REVIEW_COUNT += 1
            aspect_words = []
            aspects = a.split('#')
            aspect_words.extend(aspects[0].split('_'))
            if len(aspects) > 1:
                aspect_words.extend(aspects[1].split('_'))
            augmented_review = []
            augmented_polarity = []
            # check which sentences from the current review are related to this aspect 'a' and has some polarity.
            # Iterate over each sentence from the review and check in the aspect's map whether it is present there
            # or not. If yes, mark the sentence's sentiment polarity accordinly or otherwise mark it N/A(3)
            for j, s in enumerate(review_text):
                updated_polarity = 3
                for sent_polarity in sent_polarities:
                    if j == sent_polarity[0]:
                        # sentence j contains current aspect
                        updated_polarity = sent_polarity[1]
                        break
                if updated_polarity == 3:
                    TOTAL_NOT_APPLICABLE_LABEL_COUNT += 1
                augmented_polarity.append(updated_polarity)
                augmented_review.append(s)
            augmented_datapoint = [aspect_words, augmented_review, augmented_polarity]
            dataset.append(augmented_datapoint)

            if OVERSAMPLING:
                oversampled_datapoints = oversampling(augmented_datapoint)
                if oversampled_datapoints is not None:
                    for oversampled_datapoint in oversampled_datapoints:
                        TOTAL_NEUTRAL_LABEL_COUNT += 1
                        TOTAL_AUGMENTED_REVIEW_COUNT += 1
                        dataset.append(oversampled_datapoint)

        print('---------')
    print(dataset[0])
    print(dataset[1])
    print(dataset[2])
    print(dataset[3])
    print(dataset[4])
    print(dataset[5])
    print(dataset[6])
    print(dataset[7])
    print(dataset[8])
    print(dataset[9])
    print(dataset[10])
    print(dataset[11])
    print(dataset[12])
    print(len(dataset))
    output_file_name = 'formatted_' + DATA_TYPE + '_' + mode + '.pickle'
    write_binary(dataset, filename = output_file_name)
    print('---', mode, '---')
    print('TOTAL_REVIEW_COUNT: ', TOTAL_REVIEW_COUNT)
    print('TOTAL_SENTENCE_COUNT: ', TOTAL_SENTENCE_COUNT)
    print('TOTAL_AUGMENTED_REVIEW_COUNT: ', TOTAL_AUGMENTED_REVIEW_COUNT)
    print('TOTAL_POSITIVE_LABEL_COUNT: ', TOTAL_POSITIVE_LABEL_COUNT)
    print('TOTAL_NEGATIVE_LABEL_COUNT: ', TOTAL_NEGATIVE_LABEL_COUNT)
    print('TOTAL_NEUTRAL_LABEL_COUNT: ', TOTAL_NEUTRAL_LABEL_COUNT)
    print('TOTAL_NOT_APPLICABLE_LABEL_COUNT: ', TOTAL_NOT_APPLICABLE_LABEL_COUNT)
    total_label_count = TOTAL_POSITIVE_LABEL_COUNT + TOTAL_NEGATIVE_LABEL_COUNT + TOTAL_NEUTRAL_LABEL_COUNT + TOTAL_NOT_APPLICABLE_LABEL_COUNT
    print('TOTAL_LABELS: ', total_label_count)
    print('CLASS 0: ', (TOTAL_POSITIVE_LABEL_COUNT / total_label_count) * 100)
    print('CLASS 1: ', (TOTAL_NEGATIVE_LABEL_COUNT / total_label_count) * 100)
    print('CLASS 2: ', (TOTAL_NEUTRAL_LABEL_COUNT / total_label_count) * 100)
    print('CLASS 3: ', (TOTAL_NOT_APPLICABLE_LABEL_COUNT / total_label_count) * 100)
    print('ASPECT_TO_SENTENCE_FREQUENCY:')
    for k, v in ASPECT_TO_SENTENCE_FREQUENCY.items():
        print(k + ": " + str(v))


def oversampling(datapoint):
    augmented = False
    oversampled_datapoints = []
    sent_paraphrases_map = {}
    aspect_words = datapoint[0]
    review = datapoint[1]
    polarity = datapoint[2]
    augmented_review, augmented_polarity = [], []
    for i, sent_polarity in enumerate(polarity):
        if sent_polarity in MINORITY_CLASS:
            sent = review[i]
            augmented_review.append(sent)
            augmented_polarity.append(polarity[i])
            augmented = True
            if PARAPHRASING:
                sent_paraphrases_map[sent[0]] = paraphrase(sent)
    oversampled_datapoints.append([aspect_words, augmented_review, augmented_polarity])

    # permute paraphrased sentences to create new data points
    if PARAPHRASING:
        for i, sent in enumerate(augmented_review):
            paraphrases = sent_paraphrases_map.get(sent[0])
            for p_sent in paraphrases:
                new_review = augmented_review.copy()
                new_review[i] = [p_sent]
                oversampled_datapoints.append([aspect_words, new_review, augmented_polarity])

    if augmented:
        # print('##########################')
        # for dp in oversampled_datapoints:
        #     print(dp)
        # print('##########################')
        return oversampled_datapoints
    else:
        return None


def paraphrase(text):
    """
    This method paraphrases a sentence by replacing nouns and adjectives with synonym words.

    1) find POS tags.
    2) find synonyms for nouns and adjectives.
    3) Make sure to use noun synonyms for nouns and adjective synonyms for adjectives.
    4) permute synonyms to create multiple paraphrases.
    :param text:
    :return:
    """

    word_to_synonym_map = {}
    paraphrases = []
    sentence = text[0]

    doc = NLP(sentence)
    for token in doc:
        word = token.text
        word_pos = token.pos_
        if word_pos == 'NOUN':
            synonyms = get_matched_words(word, word_pos = 'n')
            word_to_synonym_map[word] = synonyms
        elif word_pos == 'ADJ':
            synonyms = get_matched_words(word, word_pos = 'a')
            word_to_synonym_map[word] = synonyms

    for word, synonyms in word_to_synonym_map.items():
        for synonym in synonyms:
            paraphrases.append(sentence.replace(word, synonym))

    print('##########################')
    for dp in paraphrases:
        print(dp)
    print('##########################')
    return paraphrases


def limit_matched_words(words):
    matched_words = set()
    count = MAX_MATCHED_WORDS
    for word in words:
        if count > 0:
            matched_words.add(word)
            count -= 1
        else:
            break
    return matched_words


def get_matched_words(word, word_pos):
    matched_words = set()
    if MATCH_TYPE == 'synonym':
        matched_words = limit_matched_words(get_synonyms(word, word_pos))
    elif MATCH_TYPE == 'similar':
        matched_words = limit_matched_words(get_similar_words_using_embedding(word))
    return matched_words


def get_similar_words_using_embedding(word):
    similar_words = get_similar_words(word, embedding = embedding)
    reduced_similar_words = set()
    for similar_word in similar_words:
        if word not in similar_word[0]:
            reduced_similar_words.add(similar_word[0])
    return reduced_similar_words


def get_synonyms(word, word_pos):
    synonyms = set()
    syns = wordnet.synsets(word)
    for syn in syns:
        syn_pos = syn.name().split('.')[1]
        if syn_pos == word_pos:
            for l in syn.lemmas():
                synonyms.add(' '.join(l.name().split('_')))
    return synonyms


def make_split(doc):
    test_reviews = []
    val_reviews = []
    reviews = doc['Reviews']['Review']
    for i, review in enumerate(reviews):
        r = random.random()
        if r < 0.5:
            test_reviews.append(review)
        else:
            val_reviews.append(review)
    return test_reviews, val_reviews


if __name__ == '__main__':
    doc = read_xml(INPUT_FILE_PATH)
    reviews = doc['Reviews']['Review']
    make_flatten_restaurant_data_sentence_level(reviews, mode = TYPE)
