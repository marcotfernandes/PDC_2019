import numpy as np


def unravel(Stx, Srx, nSamples, nodes, lambda_r, num_iter_train, p_train,
            p_val):
    # Shortcut for variables size
    nSamples_size = len(nSamples)
    lambda_size = len(lambda_r)
    nodes_size = len(nodes)

    s = []
    s1 = list([
        Stx, Srx, nSamples[0], 4, lambda_r[0], num_iter_train, p_train, p_val
    ])
    s2 = list([
        Stx, Srx, nSamples[0], 4, lambda_r[0], num_iter_train, p_train, p_val
    ])

    for i in range(0, nSamples_size):
        for j in range(0, lambda_size):
            for k in range(0, nodes_size):
                nSamples_aux = nSamples[i]
                lambda_aux = lambda_r[j]
                nodes_aux = nodes[k]
                s1 = list([
                    Stx, Srx, nSamples_aux, nodes_aux, lambda_aux,
                    num_iter_train, p_train, p_val
                ])
                s.append(s1)
    return s


def result(nSamples_size, lambda_size, nodes_size, res):
    cnt = 0
    MSE_train = np.zeros((nSamples_size, lambda_size, nodes_size))
    MSE_val = np.zeros((nSamples_size, lambda_size, nodes_size))
    for i in range(0, nSamples_size):
        for j in range(0, lambda_size):
            for k in range(0, nodes_size):
                MSE_train[i, j, k] = res[cnt][0]
                MSE_val[i, j, k] = res[cnt][1]

    return MSE_train, MSE_val
