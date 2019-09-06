import scipy.io
import numpy as np
from functions import before_neural_network as BNN
from functions import during_neural_network as DNN
from functions import after_neural_network as ANN
from datetime import date

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
pdata = 1               # percentage of data that will be used


# definition of data percentage for train and validation
p_train = 0.6           # train data percentage
p_val = 0.2             # validation data percentage

# extract only one polarization from Tx and Rx signal
Stx = Stx[0, :]
Srx = Srx[0, :]

# Check if the gradient is being well calculated
BNN.checkgrad(Srx, Stx, 4, p_train, p_val)

# Define the search space
nSamples = np.arange(1, 16, 5)
lambda_r = np.array([0, 0.1, 1])
nodes = np.arange(5, 21, 5)

# Shortcut for variables size
nSamples_size = len(nSamples)
lambda_size = len(lambda_r)
nodes_size = len(nodes)

# Initialize error variables
MSE_train = np.zeros((nSamples_size, lambda_size, nodes_size))
MSE_val = np.zeros((nSamples_size, lambda_size, nodes_size))

# Cut data vectors to the pretended size
m = Stx.shape[0]
Stx = Stx[0:round(pdata*m)]
Srx = Srx[0:round(pdata*m)]

# Search for the best ANN architecture
for i in range(0,nSamples_size) :
    # Convert the data in ANN format
    X, Y = BNN.dataConverterRegression(Srx, Stx, nSamples[i])
    # Divide the data in train and validation
    X_train, X_val, X_trash, Y_train, Y_val, Y_trash = BNN.divideData(X,Y,p_train,p_val)

    for j in range(0, lambda_size) :
        for k in range(0, nodes_size) :
            nSamples_aux = nSamples[i]
            lambda_aux = lambda_r[j]
            nodes_aux = nodes[k]
            MSE_train[i,j,k], MSE_val[i,j,k] = DNN.trainANN(X_train, Y_train, nSamples_aux, nodes_aux, lambda_aux,
                         num_iter_train, X_val, Y_val)

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
today = date.today()
str_date = today.strftime("%d%m%Y")
string_res = "Results_{}dBm_{}spans_{}.mat".format(RxPw, spans, str_date)

ANN.saveVar(string_res, MSE_test, MSE_val, MSE_train, nSamples, lambda_r, nodes, num_iter_train, nSamplesBest, lambdaBest,
            nodesBest, num_iter, Theta1F, Theta2F, X_train.shape[0], X_val.shape[0], X_test.shape[0])





