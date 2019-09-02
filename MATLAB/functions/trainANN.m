function [MSE_TRAIN,MSE_VAL] = trainANN(X_Train,Y_Train,nSamples,...
                nodes,lambda,iter,X_Val,Y_Val)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [Theta1, Theta2] = ANN_Regression(X_Train,Y_Train,nSamples,...
                    nodes,lambda,iter);

    % == Predict train and validation outputs == %
    Y_train_pred = predictReg(Theta1,Theta2,X_Train);
    Y_val_pred   = predictReg(Theta1,Theta2,X_Val);

    % ========== Calculate the errors ========== %
    [MSE_TRAIN,~] = errCalculator(Y_train_pred, Y_Train);

    [MSE_VAL,~] = errCalculator(Y_val_pred, Y_Val);
end

