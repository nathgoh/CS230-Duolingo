"""
Duolingo SLAM Shared Task - LSTM Model


"""

import argparse
from collections import defaultdict, namedtuple
from io import open
import math
import os
from random import shuffle, uniform


from get_data import load_data, extract_features
import numpy as np
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--train', help='Training file name', required=True)
    parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    parser.add_argument('--pred', help='Output file name for predictions, defaults to test_name.pred')
    args = parser.parse_args()

    if not args.pred:
        args.pred = args.test + '.pred'

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.test)

    # Assert that the train course matches the test course
    assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5]

    training_data, training_labels = load_data(args.train)
    test_data = load_data(args.test)

    print("Formatting train data...")    
    training_feature_dict = extract_features(training_data)

    print("Formatting test data...")
if __name__ == '__main__':
    main()