function  errPerIter(nn_params,iter)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
global input_layer_size;
global hidden_layer_size;
global X_Train;
global X_Val;
global Y_Train;
global Y_Val;
global Error_train;
global Error_val;

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 2, (hidden_layer_size + 1));

Y_train_pred = predictReg(Theta1,Theta2,X_Train);
Y_val_pred   = predictReg(Theta1,Theta2,X_Val);

% ========== Calculate the errors ========== %
[Error_train(iter),~] = ...
    errCalculator(Y_train_pred, Y_Train);

[Error_val(iter),~] = ...
    errCalculator(Y_val_pred, Y_Val);
end

