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

    
    training_feature_dict = extract_features(training_data)
    # print(training_labels)

    # training_instances = [LSTM_Instance(features=instance_data.to_features(),
    #                                                  label=training_labels[instance_data.instance_id],
    #                                                  name=instance_data.instance_id
    #                                                  ) for instance_data in training_data]

    # test_instances = [LSTM_Instance(features=instance_data.to_features(),
    #                                              label=None,
    #                                              name=instance_data.instance_id
    #                                              ) for instance_data in test_data]

    # LSTM_model = LSTM()
    # LSTM_model.train(training_instances, iterations=10)

    # predictions = LSTM_model.predict_test_set(test_instances)

    # with open(args.pred, 'wt') as f:
    #     for instance_id, prediction in iteritems(predictions):
    #         f.write(instance_id + ' ' + str(prediction) + '\n')

# class LSTM_Instance(namedtuple('Instance', ['features', 'label', 'name'])):
#     """
#     A named tuple for packaging together the instance features, label, and name.
#     """
#     def __new__(cls, features, label, name):
#         if label:
#             if not isinstance(label, (int, float)):
#                 raise TypeError('LSTM_Instance label must be a number.')
#             label = float(label)
#         if not isinstance(features, dict):
#             raise TypeError('LSTM_Instance features must be a dict.')
#         return super(LSTM_Instance, cls).__new__(cls, features, label, name)



if __name__ == '__main__':
    main()
