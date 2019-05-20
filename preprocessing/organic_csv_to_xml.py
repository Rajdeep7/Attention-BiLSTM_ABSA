import sys
import os

sys.path.append(os.getcwd())
from utils.data_util import write_xml, read_line

REMOVE_IRRELEVANT = False
COARSER_ASPECTS = True
VERSION = 'v2'

def get_sentiment_value(polarity):
    """
    :param polarity:
    :return:
    """
    if polarity.strip() == 'p':
        return 'positive'
    elif polarity.strip() == 'n':
        return 'negative'
    elif polarity.strip() == '0':
        return 'neutral'
    else:
        return None


def get_aspect_value(line):
    entity = get_entity_value(line[6+1])
    attribute = get_attribute_value(line[7+1])
    aspect = entity + '#' + attribute
    return aspect


def get_entity_value(entity):
    if COARSER_ASPECTS:
        if entity.strip() in ['g', 'p', 'f', 'c']:
            return 'organic'
        elif entity.strip() in ['cg', 'cp', 'cf', 'cc']:
            return 'conventional'
        elif entity.strip() in ['gg']:
            return 'gmo_genetic_engineering'
        else:
            return ''
    else:
        if entity.strip() == 'g':
            return 'organic_general'
        elif entity.strip() == 'p':
            return 'organic_products'
        elif entity.strip() == 'f':
            return 'organic_farming'
        elif entity.strip() == 'c':
            return 'organic_companies'
        elif entity.strip() == 'cg':
            return 'conventional_general'
        elif entity.strip() == 'cp':
            return 'conventional_products'
        elif entity.strip() == 'cf':
            return 'conventional_farming'
        elif entity.strip() == 'cc':
            return 'conventional_companies'
        elif entity.strip() == 'gg':
            return 'gmo_genetic_engineering'
        else:
            return ''


def get_attribute_value(attribute):
    if COARSER_ASPECTS:
        if attribute.strip() in ['g']:
            return 'general'
        elif attribute.strip() in ['p']:
            return 'price'
        elif attribute.strip() in ['t', 'q']:
            return 'quality'
        elif attribute.strip() in ['s', 'h', 'c']:
            return 'safety_healthiness'
        elif attribute.strip() in ['ll', 'or', 'l', 'av']:
            return 'trustworthy_sources'
        elif attribute.strip() in ['e', 'a', 'pp']:
            return 'environment'
        else:
            return ''
    else:
        if attribute.strip() == 'g':
            return 'general'
        elif attribute.strip() == 'p':
            return 'price'
        elif attribute.strip() == 't':
            return 'taste'
        elif attribute.strip() == 'q':
            return 'nutritional_quality_freshness_appearance'
        elif attribute.strip() == 's':
            return 'safety'
        elif attribute.strip() == 'h':
            return 'healthiness'
        elif attribute.strip() == 'c':
            return 'chemicals_pesticides'
        elif attribute.strip() == 'll':
            return 'label'
        elif attribute.strip() == 'or':
            return 'origin_source'
        elif attribute.strip() == 'l':
            return 'local'
        elif attribute.strip() == 'av':
            return 'availability'
        elif attribute.strip() == 'e':
            return 'environment'
        elif attribute.strip() == 'a':
            return 'animal_welfare'
        elif attribute.strip() == 'pp':
            return 'productivity'
        else:
            return ''


def remove_irrelevant_review(output_doc):
    """
    Removes irrelevant comments i.e comments with 0 relevant sentences
    :param output_doc:
    :return:
    """
    if len(output_doc['Reviews']['Review']) > 0 and REMOVE_IRRELEVANT:
        last_review = output_doc['Reviews']['Review'][-1]
        irrelevant = True
        for sentence in last_review['sentences']['sentence']:
            if 'Opinions' in sentence.keys():
                irrelevant = False
                break
        if irrelevant:
            output_doc['Reviews']['Review'].pop(-1)


def csv_to_xml(input_csv, output_xml):
    """

    :param input_csv:
    :param output_xml:
    :return:
    """
    input_doc = read_line(input_csv)
    review_list = []
    reviews = {
        'Review': review_list
    }
    output_doc = {'Reviews': reviews}
    current_review_id = -1
    for i, l in enumerate(input_doc):
        # print(i)
        if i == 0:
            # skipping header line
            continue
        line = l.split('|')

        # sentenceId == 1 marks the start of a new review
        # if line[3+1] == '1.0':
        if current_review_id != int(float(line[2 + 1])):
            sent_ids = []
            current_review_id = int(float(line[2+1]))
            review = {
                '@rid': current_review_id
            }
            remove_irrelevant_review(output_doc = output_doc)
            output_doc['Reviews']['Review'].append(review)
            sentence_list = []
            sentences = {
                'sentence': sentence_list
            }
            output_doc['Reviews']['Review'][-1]['sentences'] = sentences

        # check for duplicate sentence
        duplicate_sentence = True if int(float(line[3+1])) in sent_ids else False

        # if current sentence id alread exists it means multiple opinions.
        if duplicate_sentence:
            # print(int(float(line[2+1])))
            # sentence = output_doc['Reviews']['Review'][-1]['sentences']['sentence'][int(float(line[3+1]))-1]
            sentence = output_doc['Reviews']['Review'][-1]['sentences']['sentence'][- 1]
            # print(sentence)
            if 'Opinions' in sentence.keys():
                opinion_list = sentence['Opinions']['Opinion']
        else:
            # create a new opinion list
            sent_ids.append(int(float(line[3+1])))
            opinion_list = []

            # create a sentence
            sentence = {
                '@id': int(float(line[3+1])),
                'text': line[-3],
            }

        opinion = {
            '@category': get_aspect_value(line),
            '@polarity': get_sentiment_value(line[5+1])
        }
        if not opinion['@polarity'] or opinion['@category'] == '#':
            pass
        else:
            opinion_list.append(opinion)
        opinions = {
            'Opinion': opinion_list
        }

        # if sentence is relevant then only add its opinions.
        if line[4+1] == '9' and opinion['@polarity'] and opinion['@category'] and opinion['@category'] != '#':
            sentence['Opinions'] = opinions

        # extract lastest review and add sentence
        if not duplicate_sentence:
            output_doc['Reviews']['Review'][-1]['sentences']['sentence'].append(sentence)

    # write to the xml file
    print(len(output_doc['Reviews']['Review']))
    write_xml(file_path = output_xml, data = output_doc)


if __name__ == '__main__':
    input_file = os.path.join(*[os.path.curdir, 'dataset', 'organic_train_'+VERSION+'.csv'])
    output_file = os.path.join(*[os.path.curdir, 'dataset', 'organic_train_'+VERSION+'.xml'])
    csv_to_xml(input_file, output_file)
    input_file = os.path.join(*[os.path.curdir, 'dataset', 'organic_val_'+VERSION+'.csv'])
    output_file = os.path.join(*[os.path.curdir, 'dataset', 'organic_val_'+VERSION+'.xml'])
    csv_to_xml(input_file, output_file)
    input_file = os.path.join(*[os.path.curdir, 'dataset', 'organic_test_'+VERSION+'.csv'])
    output_file = os.path.join(*[os.path.curdir, 'dataset', 'organic_test_'+VERSION+'.xml'])
    csv_to_xml(input_file, output_file)
