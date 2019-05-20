import sys
import os

sys.path.append(os.getcwd())
from utils.data_util import read_xml, write_binary
import spacy
import numpy as np
import random

# -----CHANGE THESE VALUES ACCORDINGLY BEFORE RUNNING THE SCRIPT-----
INCLUDE_NOT_APPLICABLE = True
INCLUDE_PERCENTAGE = 0.3
INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'germEval17_train_v1.4.xml'])
OUTPUT_FILE_NAME = 'formatted_ubahn_train.pickle'
# INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'germEval17_dev_v1.4.xml'])
# OUTPUT_FILE_NAME = 'formatted_ubahn_val.pickle'
# INPUT_FILE_PATH = os.path.join(*[os.path.curdir, 'dataset', 'germEval17_test_TIMESTAMP1.xml'])
# OUTPUT_FILE_NAME = 'formatted_ubahn_test.pickle'

# -------------------------------------------------------------------

TOTAL_REVIEW_COUNT = 0
TOTAL_AUGMENTED_REVIEW_COUNT = 0
TOTAL_POSITIVE_LABEL_COUNT = 0
TOTAL_NEGATIVE_LABEL_COUNT = 0
TOTAL_NEUTRAL_LABEL_COUNT = 0
TOTAL_NOT_APPLICABLE_LABEL_COUNT = 0
ASPECT_TO_TEXT_FREQUENCY = {}

NLP = spacy.load('de_core_news_sm')


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


def update_aspect_to_text_frequency(aspect):
    global ASPECT_TO_TEXT_FREQUENCY
    count = ASPECT_TO_TEXT_FREQUENCY.get(aspect)
    if count:
        ASPECT_TO_TEXT_FREQUENCY[aspect] = count + 1
    else:
        ASPECT_TO_TEXT_FREQUENCY[aspect] = 1


def make_flat_data():
    """
    [
    [[aspect1], [review1], [polarity]],
    [[aspect2], [review1], [polarity]]
    ]

    [['food', 'quality'], [[['Judging from previous posts this used to be a good place, but not any longer.'], [0, 0, 0, 1]],
                      ,[['We, there were four of us, arrived at noon - the place was empty - and the staff acted like we
                        were imposing on them and they were very rude.'], [0, 0, 0, 1]],
                      [['They never brought us complimentary noodles, ignored repeated requests for sugar,
                        and threw our dishes on the table.'], [0, 0, 0, 1]],
                      [['The food was lousy - too sweet or too salty and the portions tiny.'], [0, 1, 0, 0]],
                      [['After all that, they complained to me about the small tip.'], [0, 0, 0, 1]],
                      [['Avoid this place!'], [0, 0, 0, 1]]
                      ]
    ]

    This method reads data from the original xml file and formats it in the way shown above. If N is the number of
    possible aspects in this dataset then we repeat or augment each review N times once for each aspect. A review can
    consist of any number of sentences. Each sentence in a review has a label. Labels represent sentiment polarity or
    non applicability of a sentence corresponding to an aspect. For instance, in the above example labels for each
    sentence are generated for the aspect food#quality. Sentences which either do dont talk about this particular aspect
    or any of the possible aspects are labeled as N/A in this datapoint. For instance, the last sentence "Avoid this place"
    is maked as N/A in this datapoint. Although this same sentence will be labelled as NEGATIVE in another datapoint of
    the same review for another aspect restaurant#general.
    :return:
    """

    possible_categories = ['allgemein',
                           'atmosphäre',
                           'connectivity',
                           'design',
                           'gastronomisches_angebot',
                           'informationen',
                           'db_app_und_website',
                           'service_und_kundenbetreuung',
                           'komfort_und_ausstattung',
                           'gepäck',
                           'auslastung_und_platzangebot',
                           'ticketkauf',
                           'toiletten',
                           'zugfahrt',
                           'reisen_mit_kindern',
                           'image',
                           'qr-code',
                           'barrierefreiheit',
                           'sicherheit',
                           'sonstige_unregelmässigkeiten']

    global TOTAL_REVIEW_COUNT
    global TOTAL_AUGMENTED_REVIEW_COUNT
    global TOTAL_POSITIVE_LABEL_COUNT
    global TOTAL_NEGATIVE_LABEL_COUNT
    global TOTAL_NEUTRAL_LABEL_COUNT
    global TOTAL_NOT_APPLICABLE_LABEL_COUNT
    global INCLUDE_NOT_APPLICABLE
    global INCLUDE_PERCENTAGE

    doc = read_xml(INPUT_FILE_PATH)
    dataset = []
    for i, review in enumerate(doc['Documents']['Document']):
        TOTAL_REVIEW_COUNT += 1
        print('document-' + str(i))

        tokenized_review_text = []
        category_polarity_map = {}
        text = review['text']
        tokens = NLP(text)

        if 'Opinions' in review.keys():
            opinions = review['Opinions']['Opinion']
            if isinstance(opinions, dict):
                opinions = [opinions]
            for opinion in opinions:
                category = opinion['@category'].lower().split('#')[0]
                update_aspect_to_text_frequency(category)
                polarity = get_categorical_sentiment(opinion['@polarity'])
                category_polarity_map[category] = polarity

        for token in tokens:
            tokenized_review_text.append(token.text)

        if INCLUDE_NOT_APPLICABLE:
            for possible_category in possible_categories:
                sentiment = category_polarity_map.get(possible_category, None)
                if sentiment is None:
                    ran = random.random()
                    if ran <= INCLUDE_PERCENTAGE:
                        sentiment = 3
                        TOTAL_NOT_APPLICABLE_LABEL_COUNT += 1
                    else:
                        continue
                category_tokens = possible_category.split('_')
                if 'und' in category_tokens:
                    category_tokens.remove('und')
                datapoint = [category_tokens, tokenized_review_text, sentiment]
                # print(datapoint)
                dataset.append(datapoint)
                TOTAL_AUGMENTED_REVIEW_COUNT += 1
        else:
            for category, polarity in category_polarity_map.items():
                category_tokens = category.split('_')
                if 'und' in category_tokens:
                    category_tokens.remove('und')
                datapoint = [category_tokens, tokenized_review_text, polarity]
                print(datapoint)
                dataset.append(datapoint)
        print('---------')
    print(dataset[0])
    print(len(dataset))
    write_binary(dataset, filename = OUTPUT_FILE_NAME)
    print('TOTAL_REVIEW_COUNT: ', TOTAL_REVIEW_COUNT)
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
    print('ASPECT_TO_TEXT_FREQUENCY:')
    for k, v in ASPECT_TO_TEXT_FREQUENCY.items():
        print(k + ": " + str(v))


if __name__ == '__main__':
    make_flat_data()
