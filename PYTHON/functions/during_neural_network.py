import numpy as np
from functions import after_neural_network as ANN
from scipy.optimize import minimize


def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))

    epsilom_init = 0.12
    W = np.random.random((L_out, L_in + 1)) * 2 * epsilom_init - epsilom_init
    return W


def sigmoidGradient(z):
    g = np.zeros(z.shape)
    h = 1 / (1 + np.exp(-z))
    g = h * (1 - h)
    return g


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lambda_r):
    # Reshape nn_params back into Theta1 e Theta2
    Theta1 = np.reshape(nn_params[0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))
    # Number of examples
    m = x.shape[0]
    # Inicialize the output variables
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # Part 1 -  Feedforward the neural network and return the cost in the variable J.
    bias11 = np.ones((m, 1))
    a11 = np.concatenate((bias11, x), 1)
    a11 = np.transpose(a11)
    z11 = np.matmul(Theta1, a11)
    a22 = 1.0 / (1.0 + np.exp(-z11))
    bias22 = np.ones((1, m))
    a22 = np.concatenate((bias22, a22), 0)
    z22 = np.matmul(Theta2, a22)
    a33 = z22
    delta33 = a33 - np.transpose(y)
    delta22_aux1 = np.matmul(np.transpose(Theta2), delta33)
    delta22_aux2 = np.concatenate((bias22, sigmoidGradient(z11)), 0)
    delta22 = delta22_aux1 * delta22_aux2
    Theta1_grad = Theta1_grad + np.matmul(delta22[1:, :], np.transpose(a11))
    Theta2_grad = Theta2_grad + np.matmul(delta33, np.transpose(a22))
    h = np.transpose(a33)

    trash, J = ANN.errCalculator(y, h)
    Theta1_grad = Theta1_grad / m
    Theta2_grad = Theta2_grad / m

    # Regularization
    theta1 = Theta1[:, 1:]
    theta2 = Theta2[:, 1:]
    reg = (lambda_r / (2 * m)) * (np.sum(np.power(np.ravel(theta1), 2)) + np.sum(np.power(np.ravel(theta2), 2)))
    J += reg

    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_r / m) * theta1
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_r / m) * theta2

    grad = np.concatenate((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))

    return J, grad


def NN_regression(X_train, Y_train, nSamples, nodes, lambda_r, num_iter, no=0, th1=0, th2=0):
    output_layer_size = 2
    input_layer_size = 2 * nSamples
    hidden_layer_size = nodes

    if no:
        initial_Theta1 = th1
        initial_Theta2 = th2
    else:
        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size)

    initial_nn_params = np.concatenate((np.ravel(initial_Theta1), np.ravel(initial_Theta2)), 0)

    cost_fucntion_handler = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size,
                                                     output_layer_size, X_train, Y_train, lambda_r)[0]
    res = minimize(cost_fucntion_handler, initial_nn_params, method='nelder-mead',
                   options={'xtol': -1, 'disp': True, 'maxiter': num_iter,
                            'maxfev' : 5000000})

    Theta1 = np.reshape(res.x[0:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(res.x[(hidden_layer_size * (input_layer_size + 1)):],
                        (output_layer_size, (hidden_layer_size + 1)))

    return Theta1, Theta2

def trainANN(X_train, Y_train, nSamples, nodes, lambda_r, num_iter, X_val, Y_val):
    Theta1, Theta2 = NN_regression(X_train, Y_train, nSamples, nodes, lambda_r, num_iter)

    Y_train_pred = ANN.predictReg(Theta1, Theta2, X_train)
    Y_val_pred = ANN.predictReg(Theta1, Theta2, X_val)

    MSE_train, trash = ANN.errCalculator(Y_train_pred,Y_train)
    MSE_val, trash = ANN.errCalculator(Y_val_pred,Y_val)

    return MSE_train, MSE_val
