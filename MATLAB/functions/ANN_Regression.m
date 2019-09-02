 function [Theta1 Theta2] = ANN_Regression(X_Train,Y_Train,nSamples,nodes,lambda,iter,varargin)
% [Theta1 Theta2] = ANN_Regression(X_Train,Y_Train,varargin)
% This function trains a Artificial Neural Network with X_Train being the
% inputs and Y_Train being the known values of the corresponding outputs.
% The input can be with the values of:
% - theta (initial thetas) - default : random
% - nSamples (number of samples) - default : 4
% - nodes (number of nodes in the hidden layer) - default : 5
% - iter (number of iterations) - default : 10


%% Default values
global input_layer_size;
global hidden_layer_size;

output_layer_size = 2;

flag = false;

%% Parse arguments

%fprintf('Initializing Neural Network Parameters ...\n')

if(nargin > 6)
    if(strcmp(varargin{1},'theta'))
        initial_Theta1 = varargin{2};
        initial_Theta2 = varargin{3};
        flag = true;
    end
end



%% Setup the parameters of the NN

input_layer_size  = 2*nSamples;   % Size of the input layer
hidden_layer_size = nodes;        % Number of nodes of the hidden layer

if(~flag) 
    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    initial_Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);
end

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

%% Some few prints

% fprintf('\nValues of ANN:\n')
% fprintf('Number of Samples: %i \n', nSamples)
% fprintf('Size of input layer: %i \n',input_layer_size)
% fprintf('Size of hidden layer: %i \n', hidden_layer_size)
% fprintf('Value of regularization variable: %i \n',lambda)
% fprintf('Number of iterations: %i \n', iter)
% if(flag)
%     fprintf('Initial thetas were passed\n')
% else
%     fprintf('Initial thetas are random\n')
% end

%% Train the Neural Network

% options.Algorithm = 'levenberg-marquardt';
options.MaxFunEvals = 500000;
options.TolFun = 1e-10;
options.TolX   = -1;
% options.MaxIter = iter;
options.Display = 'off';
% options.OutputFcn = @outfun;
% options.Display = 'iter';

% Create "short hand" for the cost function to be minimized
% costFunction = @(p) nnCostFunction(p, ...
%                                    input_layer_size, ...
%                                    hidden_layer_size, ...
%                                    output_layer_size, X_Train, Y_Train, lambda);
%cost = zeros(1,iter);
% nn_params = lsqnonlin(costFunction,initial_nn_params,[],[],options);


options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: 
                          % the cost value and the gradient,
                          % sparseAutoencoderCost.m satisfies this.
options.Method = 'levenberg-marquardt';
options.maxIter = iter;	  % Maximum number of iterations of L-BFGS to run 
% options.display = 'iter';

[nn_params, ~] = minFunc(@(p) nnCostFunction (p, input_layer_size, ...
  hidden_layer_size, output_layer_size, X_Train, Y_Train, lambda), initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 2, (hidden_layer_size + 1));


end