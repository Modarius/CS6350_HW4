from re import T
from time import time
from copy import deepcopy
import MLib as ml
import numpy as np

def main():
    # initilalization data for bank dataset
    attrib_labels = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    attribs = {
        'age':{'young', 'old'}, # this is a numeric value which will be converted to categorical
        'job':{'admin.','unemployed','management','housemaid','entrepreneur','student',
            'blue-collar','self-employed','retired','technician','services', 'unknown'}, # 'unknown'
        'marital':{'married','divorced','single'},
        'education':{'secondary','primary','tertiary', 'unknown'},  # 'unknown'
        'default':{'yes', 'no'}, 
        'balance':{'low', 'high'}, # this is a numeric value which will be converted to categorical
        'housing':{'yes', 'no'}, 
        'loan':{'yes', 'no'}, 
        'contact':{'telephone', 'cellular', 'unknown'}, #'unknown'
        'day':{'early', 'late'}, # this is a numeric value which will be converted to categorical
        'month':{'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'}, 
        'duration':{'short', 'long'}, # this is a numeric value which will be converted to categorical
        'campaign':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'pdays':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'previous':{'few', 'many'}, # this is a numeric value which will be converted to categorical
        'poutcome':{'other', 'failure', 'success', 'unknown'}, # 'unknown'
        'label':{-1, 1} # {'no', 'yes'}
        }
    train_filepath = 'bank/train.csv'
    tests_filepath = 'bank/test.csv'
    numeric_data = {
        'age':['young', 'old'], # this is a numeric value which will be converted to categorical
        'balance':['low', 'high'], # this is a numeric value which will be converted to categorical
        'day':['early', 'late'], # this is a numeric value which will be converted to categorical
        'duration':['short', 'long'], # this is a numeric value which will be converted to categorical
        'campaign':['few', 'many'], # this is a numeric value which will be converted to categorical
        'pdays':['few', 'many'], # this is a numeric value which will be converted to categorical
        'previous':['few', 'many'], # this is a numeric value which will be converted to categorical
        }
    new_labels = {'no': -1, 'yes': 1}
    
    train_data = ml.importData(train_filepath, attribs, attrib_labels, numeric_data=numeric_data, change_label=new_labels)
    tests_data = ml.importData(tests_filepath, attribs, attrib_labels, numeric_data=numeric_data, change_label=new_labels)
    del attribs['label'] # remove the label as a valid attribute

    if(train_data is not None and tests_data is not None):
        # bT = np.zeros(500, dtype=ml.Ensemble)
        # bT_train_error = np.zeros(500)
        # bT_tests_error = np.zeros(500)
        # for i in np.arange(100):
        #     tick = time()
        #     examples = train_data.sample(1000, replace=False)
        #     bT[i] = ml.baggedDecisionTree(examples, attribs=attribs, T=500, m=100, prev_ensemble=None)
        #     print(str(i) + " " + str((time() - tick)) + "s",flush=True)

        train_truth = train_data['label'].to_numpy()
        num_train = len(train_truth)
        tests_truth = tests_data['label'].to_numpy()
        num_tests = len(tests_truth)
        aB_train_data = deepcopy(train_data)
        aB = None
        bT = None
        rT = None
        aB_train_error = np.zeros(500)
        aB_tests_error = np.zeros(500)
        aB_train_tree_error = np.zeros(500)
        aB_tests_tree_error = np.zeros(500)
        bT_train_error = np.zeros(500)
        bT_tests_error = np.zeros(500)
        rT_train_error = np.zeros(500)
        rT_tests_error = np.zeros(500)
        for i in np.arange(0, 5, 1): # 5 for testing purposes
            tick = time()
            aB, aB_train_data = ml.adaBoost(aB_train_data, attribs=attribs, T=1, prev_ensemble=aB)
            aB_train_error[i] = sum(train_truth != aB.HFinal(train_data)) / num_train
            aB_tests_error[i] = sum(tests_truth != aB.HFinal(tests_data)) / num_tests            
            aB_train_tree_error[i] = sum(train_truth != aB.HFinal(train_data, idxs=i)) / num_train
            aB_tests_tree_error[i] = sum(tests_truth != aB.HFinal(tests_data, idxs=i)) / num_tests

            bT = ml.baggedDecisionTree(train_data, attribs=attribs, T=1, m=500, prev_ensemble=bT)
            bT_train_error[i] = sum(train_truth != bT.HFinal(train_data)) / num_train
            bT_tests_error[i] = sum(tests_truth != bT.HFinal(tests_data)) / num_tests

            rT = ml.randomTree(train_data, attribs=attribs, T=1, m=500, prev_ensemble=rT)
            rT_train_error[i] = sum(train_truth != rT.HFinal(train_data)) / num_train
            rT_tests_error[i] = sum(tests_truth != rT.HFinal(tests_data)) / num_tests

            print(str(i) + " " + str((time() - tick)) + "s",flush=True)
        np.savetxt('train_error.txt', np.array([rT.getWeights()]).T, fmt='%s', delimiter=',')
        np.savetxt('train_error.txt', np.array([rT_train_error]).T, fmt='%s', delimiter=',')
        np.savetxt('tests_error.txt', np.array([rT_tests_error]).T, fmt='%s', delimiter=',')
    return

if __name__ == "__main__":
    main()