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

def main(runs):
    EPOCHS = 10
    errors = np.zeros((3, EPOCHS))
    for test_run in range(runs):

        dir_path = os.path.dirname(os.path.realpath(__file__)) # https://stackoverflow.com/a/5137509
        train = importData(dir_path + '/bank-note/train.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
        train_Y = train.pop('label').to_numpy()
        train_X = train.to_numpy()

        test = importData(dir_path + '/bank-note/test.csv', ['variance', 'skewness', 'curtosis', 'entropy', 'label'])
        test_Y = test.pop('label').to_numpy()
        test_X = test.to_numpy()

        standard_model = ml.Perceptron(length=(train.shape[1]))
        voted_model = ml.VotedPerceptron(length=(train.shape[1]))
        average_model = ml.AveragePerceptron(length=(train.shape[1]))
        error = np.zeros((3, EPOCHS))
        rand = np.random.default_rng()
        indices = np.arange(train.shape[0])
        for i in range(0,EPOCHS):
            rand.shuffle(indices) # get a new ordering of data
            train_X = train_X[indices] # randomize the data but the same for x and y so they match up
            train_Y = train_Y[indices]
            standard_model.train(train_X, train_Y)
            voted_model.train(train_X, train_Y)
            average_model.train(train_X, train_Y)
            error[0,i] = standard_model.test(test_X, test_Y)
            error[1,i] = voted_model.test(test_X, test_Y)
            error[2,i] = average_model.test(test_X, test_Y)
        errors[0] += error[0,:]
        errors[1] += error[1,:]
        errors[2] += error[2,:]
    errors /= runs
    avg_errors = {"standard":errors[0,:],
                "voted":errors[1,:],
                "average":errors[2,:]}
    print("Epochs 1 ----> 10")
    for key in avg_errors.keys():
        print(key + " perceptron:\t" + str(avg_errors[key]))
    e = pd.DataFrame(avg_errors)
    e.to_csv("perceptron.csv")
    return

if __name__ == "__main__":
    np.set_printoptions(edgeitems=30, linewidth = 1000) # this is just formatting for numpy print, https://www.reddit.com/r/vscode/comments/s2xjgz/how_do_i_increase_the_width_of_the_output_of_the/?ref=share&ref_source=link
    main(1)