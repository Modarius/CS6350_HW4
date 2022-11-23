from copy import deepcopy
import os
import pandas as pd
import numpy as np
import MLib as ml

def importData(filename, column_labels=None, index_col=None, header=None):
    # index_col is a number indicating which column, if any, is the index into the data
    # header is the line of the data if any that where column labels are indicated
    terms = pd.read_csv(filename, sep=',', names=column_labels, index_col=index_col, header=header) # read in the csv file into a DataFrame object , index_col=index_col
    # if needed any processing can be done here
    terms['label'].where(terms['label'] != 0, -1, inplace=True) # change labels to be {-1, 1}
    return terms

def gamma_a(gamma_0, a, t):
    return gamma_0/(1+gamma_0/a * t)

def gamma_b(gamma_0, a, t):
    return gamma_0/(1+t)

def trainStochasticSVM(S, t=0, C=(100/873), gamma_func=gamma_b, gamma_0=1, gamma_a=0, ww=None):
    gamma_t = gamma_func(gamma_0=gamma_0, a=gamma_a, t=t)
    yy = S.pop('label').to_numpy()
    xx0 = S.to_numpy()
    xx = np.vstack((np.ones(xx0.shape[0]), xx0.T)).T
    N = xx.shape[0]
    if ww is None: ww = np.zeros(xx.shape[1]) # ww is shape [w0,0]

    for i in range(N):
        xx_fake = np.array(xx[i])
        yy_fake = np.array(yy[i])
        if (yy_fake*np.dot(ww,xx_fake.T)<=1):
            ww[-1] = 0 # zero out b, ww is augmented with b
            ww = ww - gamma_t * ww + gamma_t * C * N * yy_fake * xx_fake
        else:
            ww[0:-2] = (1-gamma_t) * ww[0:-2] # ww[0:-2] is w0
    return ww

def testStochasticSVM(S, ww):
    S = deepcopy(S)
    yy = S.pop('label').to_numpy()
    xx0 = S.to_numpy()
    xx = np.vstack((np.ones(xx0.shape[0]), xx0.T)).T
    prediction = np.where(np.dot(ww,xx.T) < 0, -1, 1)
    compare = np.where(prediction*yy > 0, 1, 0)
    error = 1 - compare.sum()/len(compare)
    return error

def main(runs):
    ww_overall = np.empty(runs, dtype=object)
    train_error_overall = np.empty(runs, dtype=object)
    test_error_overall = np.empty(runs, dtype=object)
    for test_run in range(runs):
        EPOCHS = 100
        dir_path = os.path.dirname(os.path.realpath(__file__)) # https://stackoverflow.com/a/5137509
        train = importData(dir_path + '/bank-note/train.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
        test = importData(dir_path + '/bank-note/test.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])

        rand = np.random.default_rng()
        train_indices = np.arange(train.shape[0])
        test_indices = np.arange(test.shape[0])
        ww_curr = None
        train_error = np.zeros(EPOCHS)
        test_error = np.zeros(EPOCHS)
        for i in range(0,EPOCHS):
            rand.shuffle(train_indices) # get a new ordering of data
            rand.shuffle(test_indices) # get a new ordering of data
            train_rand = train.iloc[train_indices] # randomize the data but the same for x and y so they match up
            test_rand = test.iloc[test_indices]
            ww_curr = trainStochasticSVM(train_rand, t=i, gamma_func=gamma_b, gamma_0=.3, gamma_a=0.01, ww=ww_curr)
            train_error[i] = testStochasticSVM(train, ww_curr)
            test_error[i] = testStochasticSVM(test, ww_curr)
        ww_overall[test_run] = ww_curr
        train_error_overall[test_run] = train_error
        test_error_overall[test_run] = test_error
        #print(str(test_run) + ": " + str(ww_curr))
    ww_overall = np.vstack(ww_overall)
    train_error_overall = np.vstack(train_error_overall)
    test_error_overall = np.vstack(test_error_overall)
    np.savetxt("ww_overall.csv", ww_overall, delimiter=",")
    np.savetxt("train_error_overall.csv", train_error_overall, delimiter=",")
    np.savetxt("test_error_overall.csv", test_error_overall, delimiter=",")
    return

if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth = 1000) # this is just formatting for numpy print, https://www.reddit.com/r/vscode/comments/s2xjgz/how_do_i_increase_the_width_of_the_output_of_the/?ref=share&ref_source=link
    main(50)