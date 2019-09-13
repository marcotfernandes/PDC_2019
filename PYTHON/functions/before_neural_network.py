import numpy as np
import math
from functions import during_neural_network as DNN


def dataConverterRegression(Srx, Tx, nSamples):
    m = Srx.shape[0] - 2 * nSamples
    X = np.zeros((m, 2 * nSamples))
    tmp = np.zeros((2, Srx.shape[0]))
    tmp[0:] = np.real(Srx)
    tmp[1:] = np.imag(Srx)
    Y = np.zeros((m, 2))
    Y[:, 0] = np.real(Tx[0:m])
    Y[:, 1] = np.imag(Tx[0:m])

    flag = 1

    for i in range(0, m):
        cnt = i
        for j in range(0, 2 * nSamples):
            if flag:
                X[i, j] = tmp[0, cnt]
            else:
                X[i, j] = tmp[1, cnt]
                cnt += 1
            flag = not flag

    return X, Y


def divideData(X, Y, p_train, p_val):
    m = X.shape[0]

    train_length = int(math.floor(p_train * m))
    val_length = int(math.floor(p_val * m))
    test_length = int(math.floor((1 - p_val - p_train) * m))
    #
    idx_start = 0
    idx_end = train_length
    X_train = X[idx_start:idx_end, :]
    Y_train = Y[idx_start:idx_end, :]

    idx_start = idx_end
    idx_end += val_length
    X_val = X[idx_start:idx_end, :]
    Y_val = Y[idx_start:idx_end, :]

    idx_start = idx_end
    idx_end += test_length
    X_test = X[idx_start:idx_end, :]
    Y_test = Y[idx_start:idx_end, :]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def computeNumericalGradient(J, theta):
    theta = np.ravel(theta)
    numgrad = np.zeros(theta.shape)

    eps = 0.0001

    for i in range(0, 10):
        e = np.zeros(theta.shape)
        e[i] = 1
        tmp1 = J(theta + eps * e)
        tmp2 = J(theta - eps * e)
        numgrad[i] = (tmp1[0] - tmp2[0]) / (2 * eps)
    return numgrad


def checkgrad(Srx, Stx, nSamples, p_train, p_val):
    X, Y = dataConverterRegression(Srx, Stx, nSamples)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = \
        divideData(X, Y, p_train, p_val)

    lambda_r = 0
    hidden_layer_size = 5
    input_layer_size = X_train.shape[1]
    output_layer_size = Y_train.shape[1]

    Theta1 = DNN.randInitializeWeights(input_layer_size, hidden_layer_size)
    Theta2 = DNN.randInitializeWeights(hidden_layer_size, output_layer_size)

    Theta1 = np.ravel(Theta1)
    Theta2 = np.ravel(Theta2)
    theta = np.concatenate((Theta1, Theta2))

    trash, grad = DNN.nnCostFunction(theta, input_layer_size,
                                     hidden_layer_size, output_layer_size,
                                     X_train, Y_train, lambda_r)

    function_handler = lambda x: DNN.nnCostFunction(
        x, input_layer_size, hidden_layer_size, output_layer_size, X_train,
        Y_train, lambda_r)

    numGrad = computeNumericalGradient(function_handler, theta)

    diff = np.linalg.norm(numGrad[0:10] -
                          grad[0:10]) / np.linalg.norm(numGrad[0:10] +
                                                       grad[0:10])

    if diff > 1e-8:
        print("Gradient error \n")
        return
