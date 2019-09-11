import scipy.io
import numpy as np
import multiprocessing as mp
import time
from functions import before_neural_network as BNN
from functions import during_neural_network as DNN
from functions import after_neural_network as ANN
from functions import par_help

def train():
    pass
def run():
    # declare variables for load the data
    RxPw = -5
    spans = 1
    datapath = "../MATLAB/raw_data/"
    string_name = "Data_{}dBm_{}spans.mat".format(RxPw,spans)

    # read signals mat file
    file = scipy.io.loadmat(datapath+string_name)
    # get transmitted signal
    Stx = (file.get('Stx'))
    Stx = np.array(Stx[0][0])
    # get received signal
    Srx = (file.get('Srx'))
    Srx = np.array(Srx[0][0])
    # read IQmap mat file
    file = scipy.io.loadmat(datapath+"IQmap.mat")
    # get IQmap
    IQmap = np.array(file.get('IQmap'))

    # define some parameters
    Mqam = IQmap.shape[0]   # modulation format
    num_iter_train = 8000   # number of iterations
    pdata = 0.15            # percentage of data that will be used


    # definition of data percentage for train and validation
    p_train = 0.6           # train data percentage
    p_val = 0.2             # validation data percentage

    # extract only one polarization from Tx and Rx signal
    Stx = Stx[0, :]
    Srx = Srx[0, :]

    # Check if the gradient is being well calculated
    BNN.checkgrad(Srx, Stx, 4, p_train, p_val)

    # Define the search space
    nSamples = np.arange(1, 16, 2)
    lambda_r = np.array([0, 0.1, 1, 10])
    nodes = np.array([5, 10, 15, 20, 35, 50])

    # Shortcut for variables size
    nSamples_size = len(nSamples)
    lambda_size = len(lambda_r)
    nodes_size = len(nodes)

    # Cut data vectors to the pretended size
    m = Stx.shape[0]
    Stx = Stx[0:round(pdata*m)]
    Srx = Srx[0:round(pdata*m)]

    pool = mp.Pool(processes=32)
    start = time.time()

    # Unravel search space
    s = par_help.unravel(Stx, Srx, nSamples, nodes, lambda_r, num_iter_train, p_train, p_val)
    # Search for the best ANN architecture
    num_iter_total = len(nSamples) * len(lambda_r) * len(nodes)
    res = [pool.apply(DNN.trainANN, args=(s[x])) for x in range(0, num_iter_total, 1)]

    end = time.time()
    pool.close()

    t_elapsed = end-start;

    MSE_train, MSE_val = par_help.result(nSamples_size, lambda_size, nodes_size, res)

    # Get the best values
    idx = np.unravel_index(np.argmin(MSE_val), MSE_val.shape)
    nSamplesBest = nSamples[idx[0]]
    lambdaBest = lambda_r[idx[1]]
    nodesBest = nodes[idx[2]]

    # Prepare the data for final train
    X, Y = BNN.dataConverterRegression(Srx, Stx, nSamplesBest)
    # Divide data in train, validation and test
    X_train, X_val, X_test, Y_train, Y_val, Y_test = BNN.divideData(X, Y, p_train, p_val)
    # Train ANN with best values
    num_iter = 1e4
    # Train final ANN for the first time
    Theta1, Theta2 = DNN.NN_regression(X_train, Y_train, nSamplesBest, nodesBest,
                                       lambdaBest, num_iter)
    # Join validation data with the train one
    X2 = np.concatenate((X_train, X_val), 0)
    Y2 = np.concatenate((Y_train, Y_val), 0)
    # Train final ANN a second time
    Theta1F, Theta2F = DNN.NN_regression(X2, Y2, nSamplesBest, nodesBest, lambdaBest, num_iter,
                                         1, Theta1, Theta2)
    # Predict test data
    Y_test_pred = ANN.predictReg(Theta1F, Theta2F, X_test)
    # Calculate test error
    MSE_test, trash = ANN.errCalculator(Y_test_pred, Y_test)

    #Save data
    string_res = "Results_{}dBm_{}spans_par.mat".format(RxPw, spans)

    ANN.saveVar(string_res, MSE_test, MSE_val, MSE_train, nSamples, lambda_r, nodes, num_iter_train, nSamplesBest, lambdaBest,
                nodesBest, num_iter, Theta1F, Theta2F, X_train.shape[0], X_val.shape[0], X_test.shape[0],t_elapsed)



if __name__ == '__main__':
    mp.freeze_support()
    run()
