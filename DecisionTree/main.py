#!/usr/bin/env python3
# Written by Alan Felt for CS6350 Machine Learning

import numpy as np
import pandas as pd
import MLib as ml

def DecisionTree():
    # initialization data for car dataset
    # attrib_labels = ['buying', 'maint', 'doors',
    #                  'persons', 'lug_boot', 'safety']
    # attribs = {
    #     'buying': {'vhigh', 'high', 'med', 'low'},
    #     'maint': {'vhigh', 'high', 'med', 'low'},
    #     'doors': {'2', '3', '4', '5more'},
    #     'persons': {'2', '4', 'more'},
    #     'lug_boot': {'small', 'med', 'big'},
    #     'safety': {'low', 'med', 'high'}
    # }
    # data_labels = {'unacc', 'acc', 'good', 'vgood'}
    # train_filepath = "car/train.csv"
    # test_filepath = "car/test.csv"
    # numeric_data = None
    # index_col = 6

    # initilalization data for bank dataset
    attrib_labels = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    attribs = {
        'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
        'job':{'admin.','unemployed','management','housemaid','entrepreneur','student',
            'blue-collar','self-employed','retired','technician','services'}, # 'unknown'
        'marital':{"married","divorced","single"}, 
        'education':{"secondary","primary","tertiary"},  # "unknown"
        'default':{'yes', 'no'}, 
        'balance':{'low', 'high'}, # this is a numeric value which will be converted to categorical
        'housing':{'yes', 'no'}, 
        'loan':{'yes', 'no'}, 
        'contact':{'telephone', 'cellular'}, #'unknown'
        'day':{'early', 'late'}, # this is a numeric value which will be converted to categorical
        'month':{"jan", "feb", "mar", 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', "nov", "dec"}, 
        'duration':{'short', 'long'}, # this is a numeric value which will be converted to categorical
        'campaign':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'pdays':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'previous':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'poutcome':{'other', 'failure', 'success'} # 'unknown'
        }
    data_labels = {'yes', 'no'}
    train_filepath = 'bank/train.csv'
    test_filepath = 'bank/test.csv'
    numeric_data = {
        'age':['young', 'old'], # this is a numeric value which will be converted to categorical
        'balance':['low', 'high'], # this is a numeric value which will be converted to categorical
        'day':['early', 'late'], # this is a numeric value which will be converted to categorical
        'duration':['short', 'long'], # this is a numeric value which will be converted to categorical
        'campaign':['few', 'many'], # this is a numeric value which will be converted to categorical
        'pdays':['few', 'many'], # this is a numeric value which will be converted to categorical
        'previous':['few', 'many'], # this is a numeric value which will be converted to categorical
        }
    index_col = 16 
    
    training_data = ml.importData(train_filepath, attribs, attrib_labels, data_labels, numeric_data=numeric_data, index_col=index_col)
    test_data = ml.importData(test_filepath, attribs, attrib_labels, data_labels, numeric_data=numeric_data, index_col=index_col)
    train_error = np.zeros([16,1])
    test_error = np.zeros([16,1])
    for max_depth in np.arange(start=1, stop=17):
        tree = ml.ID3(training_data, attribs, None, 'entropy', max_depth) # build the tree
        print("Depth: " + str(max_depth)) # print out current max_depth value
        # printTree(tree) # crude print of tree
        train_error[max_depth - 1 ] = treeError(tree, training_data) # test the error for the given dataset
        test_error[max_depth - 1] = treeError(tree, test_data) # test the error for the given dataset
    print('Avg Error Training Dataset = ' + str(np.average(train_error)))
    print('Avg Error Test Dataset = ' + str(np.average(test_error)))

    return

if __name__ == "__main__":
    DecisionTree()
