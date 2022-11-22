from copy import deepcopy
import pandas as pd
import numpy as np

def LMSRegression(S, r=.001, threshold=10**-6, label_name=''):
    S = deepcopy(S)
    Y = S.pop(label_name).to_numpy() # seperate off output column
    m = S.shape[0] # number of examples in X
    d = S.shape[1] + 1 # number of dimensions to X per example
    W = np.zeros(d) # init weights to zero (+1 because first slot in x is a 1)
    X = np.vstack((np.ones(m), S.to_numpy().T))
    nwvd = np.inf
    dj_dwj = np.empty(d)
    w_star = np.matmul(np.linalg.inv(np.matmul(X,X.T)), np.matmul(X,Y)) # analytic method of finding weights
    costs = np.empty(0)
    while (nwvd > threshold):
        me = (Y - np.matmul(W, X))
        dj_dwj = np.sum(-(me * X), axis=1)
        W_1 = W - r * dj_dwj
        nwvd = np.linalg.norm(W_1 - W)
        W = W_1
        cost = 1/2 * (me**2).sum()
        costs = np.append(costs, [cost])
    return W, w_star, costs

# I was unable to get Stochastic Grad. descent working despite quite a bit of work on it.
# I tried it with a loop in the while loop to iterate over all examples in X and the way it is 
# currently where each iteration just picks a random sample of X and updates the weights. The 
# cost never converges. I've also tried r values from .5 -> 10**-12 and nothing works there
# either
def SGD(S, r=.001, label_name=''):
    S = deepcopy(S)
    Y = S.pop(label_name).to_numpy() # seperate off output column
    m = S.shape[0] # number of examples in X
    d = S.shape[1] + 1 # number of dimensions to X per example
    W = np.zeros(d) # init weights to zero (+1 because first slot in x is a 1)
    X = np.vstack((np.ones(m), S.to_numpy().T))
    idxs = np.arange(0,m)
    costs = np.empty(0)
    rng = np.random.default_rng()
    iterations = 0
    while (iterations < 5000): 
        # i = rng.integers(0,m)
        # me = (Y[i] - np.matmul(W, X[:,i])) # get the error (actual - prediction)
        # W = W + r * me * X[:,i] # update the weights based on that error, the rate, and the vector of the sample data point X[i]
        # iterations += 1
        rng.shuffle(idxs) # get a random ordering of indexes to X[i]
        for i in np.arange(m):
            me = (Y[i] - np.matmul(W, X[:,i]))
            W = W + r * me * X[:,i]
        cost = 1/2 * (me**2).sum()
        costs = np.append(costs, [cost])
        iterations += 1
    return W, costs

def evalLMS(S, W, label_name=''):
    S = deepcopy(S)
    m = S.shape[0] # number of examples in X
    Y = S.pop(label_name).to_numpy() # seperate off output column
    X = np.vstack((np.ones(m), S.to_numpy().T))
    cost = 1/2 * ((Y - np.matmul(W, X))**2).sum()
    return cost

def main():
    train_filepath = "concrete/train.csv"
    test_filepath = "concrete/test.csv"
    train = pd.read_csv(train_filepath, sep=',', names=["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "SLUMP"]) # read in the csv file into a DataFrame object , index_col=index_col
    test = pd.read_csv(test_filepath, sep=',', names=["Cement", "Slag", "Fly ash", "Water", "SP", "Coarse Aggr", "Fine Aggr", "SLUMP"]) # read in the csv file into a DataFrame object , index_col=index_col
    
    print("Beginning LMS Batch Alg.")
    W, W_star, costs = LMSRegression(train, r=.0125, label_name='SLUMP')
    np.savetxt('train_costs_LMS.csv', costs, delimiter=',')
    print("Cost of LMS Test is: " + str(evalLMS(test, W, label_name='SLUMP')))
    print("W: " + str(W) + "\nW_star: " + str(W_star))

    print("Beginning Stochastic Grad. Descent Alg.")
    W, costs = SGD(train, r=.0001, label_name='SLUMP') # this is implemented but doesnt work for some reason
    np.savetxt('train_costs_SGD.csv', costs, delimiter=',')
    print("Cost of SGD Test is: " + str(evalLMS(test, W, label_name='SLUMP')))
    print("W: " + str(W))
    return

if __name__ == "__main__":
    main()