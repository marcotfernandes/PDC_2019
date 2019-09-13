import numpy as np
import scipy.io


def errCalculator(Y1, Y2):
    m = Y1.shape[0]

    real_tmp = np.power((Y1[:, 0] - Y2[:, 0]), 2)
    imag_tmp = np.power((Y1[:, 1] - Y2[:, 1]), 2)

    J = np.sum(real_tmp + imag_tmp) / (2 * m)
    MSE_Real = np.sum(real_tmp) / m
    MSE_Imag = np.sum(imag_tmp) / m
    MSE = np.mean([MSE_Real, MSE_Imag])
    return MSE, J


def predictReg(Theta1, Theta2, X):
    m = X.shape[0]
    Y = np.zeros((m, 1))

    bias1 = np.ones((m, 1))
    a1 = np.concatenate((bias1, X), 1)
    z2 = np.matmul(a1, np.transpose(Theta1))
    a2 = 1.0 / (1.0 + np.exp(-z2))

    bias2 = np.ones((m, 1))
    a2 = np.concatenate((bias2, a2), 1)

    z3 = np.matmul(a2, np.transpose(Theta2))
    return z3


def saveVar(string, MSE_test, MSE_val, MSE_train, nSamples, lambda_r, nodes,
            num_iter_train, nSamplesBest, lambdaBest, nodesBest, num_iter,
            Theta1F, Theta2F, train_size, val_size, test_size, time_t):
    # Error struct
    Error = {
        'Test': MSE_test,
        'Val': MSE_val,
        'Train': MSE_train,
    }

    SearchParam = {
        'nSamples': nSamples,
        'lambda': lambda_r,
        'nodes': nodes,
        'iterations': num_iter_train,
    }

    BestParam = {
        'nSamples': nSamplesBest,
        'lambda': lambdaBest,
        'nodes': nodesBest,
        'iterations': num_iter,
    }

    Param = {
        'Search': SearchParam,
        'Best': BestParam,
    }

    Weights = {
        'Theta1': Theta1F,
        'Theta2': Theta2F,
    }

    DataSetSize = {
        'Train': train_size,
        'Validation': val_size,
        'Test': test_size,
    }

    scipy.io.savemat(
        string, {
            'MSE': Error,
            'Param': Param,
            'Weights': Weights,
            'DataSetSize': DataSetSize,
            'TimeSearch': time_t
        })
