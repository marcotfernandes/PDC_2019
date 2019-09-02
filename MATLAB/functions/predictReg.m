function Y = predictReg(Theta1, Theta2, X)
%   Y = predictionReg(Theta1, Theta2, X)
%The point of this function is to predict the output of a regression ANN
%with the weights Theta1 and Theta2, the prevision is given by Y. X is the
%input vector.

% Length of X vector
m = size(X,1);

% Definition of the output vector
Y = zeros(m,1);

% Output of hidden layer
a1 = [ones(m, 1) X];
z2=a1*Theta1';
a2 = 1.0 ./ (1.0 + exp(-z2));

% Input of output layer
a2= [ones(m, 1) a2];

% Output of output layer
z3=a2*Theta2';
Y = z3;


end

