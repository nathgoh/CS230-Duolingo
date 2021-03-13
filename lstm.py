"""
Duolingo SLAM Shared Task - LSTM Model

"""
import argparse
from io import open
import math
import os
from random import shuffle, uniform

from get_data import load_data, extract_features
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

def main():
    parser = argparse.ArgumentParser(description='Duolingo shared task baseline model')
    parser.add_argument('--train', help='Training file name', required=True)
    parser.add_argument('--test', help='Test file name, to make predictions on', required=True)
    # parser.add_argument('--pred', help='Output file name for predictions, defaults to test_name.pred')
    args = parser.parse_args()

    # if not args.pred:
    #     args.pred = args.test + '.pred'

    assert os.path.isfile(args.train)
    assert os.path.isfile(args.test)

    # Assert that the train course matches the test course
    assert os.path.basename(args.train)[:5] == os.path.basename(args.test)[:5]

    training_data, training_labels = load_data(args.train)
    test_data = load_data(args.test)

    print("Formatting train data...")    
    format_data(training_data, training_labels)

    print("Formatting test data...")

def create_embeddings(data):
    """
    Mapping of each distinct feature to an unique index in a dictionary is used to create an embedding matrix for 
    each said distinct feature. Each individual embedding matrix will be concatenated
    together to create one large embedding matrix.

    Parameters:
        data: a list of InstanceData objects from that data type and track.
    Return:
        feature_maxtrix_concat: concatenated embedding matrix
    """     
    feature_dict = extract_features(data)  

    # Creating embedding matrices for each feature
    print("Building embedding matrix...")
    users, formats, tokens = [], [], []
    for key in feature_dict:
        if "user:" in key:
            users.append(feature_dict[key])
        if "format:" in key:
            formats.append(feature_dict[key])
        if "token:" in key:
            tokens.append(feature_dict[key])

    # Embedding layers
    user_tensor, format_tensor, token_tensor = Input(shape = (None, ), name = "users"), Input(shape = (None, ), name = "formats"), Input(shape = (None, ), name = "tokens")
    user_embed = layers.Embedding(len(users), LARGE_EMBED_SIZE) (user_tensor)
    format_embed = layers.Embedding(len(formats), SMALL_EMBED_SIZE) (format_tensor)
    token_embed = layers.Embedding(len(tokens), LARGE_EMBED_SIZE) (token_tensor)

    feature_matrix = [user_embed, format_embed, token_embed]
    feature_matrix_concat = layers.Concatenate()(feature_matrix)

    print("Embedding matrix: {}".format(feature_matrix_concat))

    # Help hopefully make network train faster
    feature_matrix_concat = layers.BatchNormalization() (feature_matrix_concat)
    input_tensor = [user_tensor, format_tensor, token_tensor]

    return feature_matrix_concat, input_tensor

if __name__ == '__main__':
    main()