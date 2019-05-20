from utils.data_util import read_xml, write_xml
from preprocessing.germEval17_data_processing import GERMEVAL_ASPECT_WORD_INDEX_MAP, GERMEVAL_INDEX_TO_ASPECT_WORD_MAP


def get_sentiment_value(polarity):
    """

    :param polarity:
    :return:
    """
    if polarity == 0:
        return 'positive'
    elif polarity == 1:
        return 'negative'
    elif polarity == 2:
        return 'neutral'
    else:
        return None


def get_formatted_predictions(raw_predictions):
    """

    :param raw_predictions:
    :return:
    """
    predictions = []
    for i, pred in enumerate(raw_predictions):
        categories = []
        for j, aspect_preds in enumerate(pred):
            if aspect_preds != 3:
                # if prediction is other than not applicable add it to categories list.
                category_tuple = [GERMEVAL_INDEX_TO_ASPECT_WORD_MAP.get(j), get_sentiment_value(aspect_preds)]
                categories.append(category_tuple)
        predictions.append(categories)
    return predictions


def write_prediction_to_xml(gold_standard_file, prediction_file, raw_predictions):
    """

    :param gold_standard_file:
    :param prediction_file:
    :param raw_predictions:
    :return:
    """
    gold = read_xml(gold_standard_file)
    preds = gold.copy()
    print(len(preds['Documents']['Document']))

    predictions = get_formatted_predictions(raw_predictions)
    print(len(predictions))

    if len(predictions) == len(preds['Documents']['Document']):
        for i, pred_doc in enumerate(preds['Documents']['Document']):
            # cleaning prediction doc
            pred_doc.pop('relevance', None)
            pred_doc.pop('Opinions', None)

            # categories = [['SSSumit', 'positive'], ['DDugar', 'neutral']]
            categories = predictions[i]
            # fill predicted values
            if len(categories) > 0:
                pred_doc['relevance'] = 'true'
                opinion_list = []
                for category in categories:
                    opinion = {
                        '@category': category[0],
                        '@polarity': category[1]
                    }
                    opinion_list.append(opinion)
                opinions = {
                    'Opinion': opinion_list
                }
                pred_doc['Opinions'] = opinions
            else:
                pred_doc['relevance'] = 'false'
    else:
        print('Number of elements in gold and predictions dont match!')
    write_xml(file_path = prediction_file, data = preds)


if __name__ == '__main__':
    gold_standard_file = '/home/sumit/Documents/repo/GermEval2017/gold1.xml'
    prediction_file = '/home/sumit/Documents/repo/GermEval2017/predictions2.xml'
    write_prediction_to_xml(gold_standard_file, prediction_file, None)
