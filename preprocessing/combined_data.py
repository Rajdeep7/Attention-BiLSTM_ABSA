import os
import sys

sys.path.append(os.getcwd())
from utils.data_util import read_binary, write_binary

# -----CHANGE THESE VALUES ACCORDINGLY BEFORE RUNNING THE SCRIPT-----
TYPE = 'train'
# TYPE = 'val'
# -------------------------------------------------------------------
PROCESSED_RESTAURANT_FILE_NAME = 'processed_' + TYPE + '.pickle.restaurant'
PROCESSED_LAPTOPS_FILE_NAME = 'processed_' + TYPE + '.pickle.laptops'
PROCESSED_ORGANIC_FILE_NAME = 'processed_' + TYPE + '.pickle.organic'
OUTPUT_FILE_NAME = 'processed_' + TYPE + '.pickle'


def combine_processed_data():
    combined_dataset = []

    restaurant = read_binary(filename = PROCESSED_RESTAURANT_FILE_NAME)
    print('Restaurant-'+str(len(restaurant)))
    combined_dataset.extend(restaurant)
    print(len(combined_dataset))

    laptops = read_binary(filename = PROCESSED_LAPTOPS_FILE_NAME)
    print('Laptops-' + str(len(laptops)))
    combined_dataset.extend(laptops)
    print(len(combined_dataset))

    # organic = read_binary(filename = PROCESSED_ORGANIC_FILE_NAME)
    # print('Organic-' + str(len(organic)))
    # combined_dataset.extend(organic)
    # print(len(combined_dataset))

    write_binary(combined_dataset, OUTPUT_FILE_NAME)

if __name__ == '__main__':
    combine_processed_data()
