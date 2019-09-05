import scipy.io
import numpy as np
from functions import before_neural_network as BNN
from functions import during_neural_network as DNN

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
num_iter = 800          # number of iterations
pdata = 0.15            # percentage of data that will be used


# definition of data percentage for train and validation
p_train = 0.6           # train data percentage
p_val = 0.2             # validation data percentage

# extract only one polarization from Tx and Rx signal
Stx = Stx[0, :]
Srx = Srx[0, :]

# Check if the gradient is being well calculated
BNN.checkgrad(Srx, Stx,4,p_train,p_val)

# Define the search space
nSamples = np.arange(1, 10, 5)
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

# Begin the train
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
                         num_iter, X_val, Y_val)

# Get the best values
