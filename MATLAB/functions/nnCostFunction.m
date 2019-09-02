function [J, grad] = nnCostFunction(nn_params,input_layer_size, ...
                                   hidden_layer_size,num_labels, X, y, lambda)
                                   
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of examples
m = size(X, 1);
         
% Inicialize the variables that the function will output
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Part 1: Feedforward the neural network and return the cost in the variable J. 

a11 = [ones(m,1) X]';
z22 = Theta1*a11;
a22 = 1.0 ./ (1.0 + exp(-z22));
a22 = [ones(1,m); a22];
z33 = Theta2*a22;
a33 = z33;
delta33 =(a33 -y');
delta22=Theta2'*delta33.*[ones(1,m); sigmoidGradient(z22)];
Theta1_grad=Theta1_grad+delta22(2:end,:)*a11';
Theta2_grad=Theta2_grad +delta33*(a22)';
h = a33';

[~,J] = errCalculator(y,h);

Theta1_grad=Theta1_grad./m;
Theta2_grad=Theta2_grad./m;

%Regularization
% Part 3: Implement regularization with the cost function and gradients.
%The gradients for  the regularization are computed separately and then 
%added Theta1_grad  and Theta2_grad from Part 2.

theta1=Theta1(:,2:end);
theta2=Theta2(:,2:end);
reg=(lambda/(2*m))*(sum(theta1(:).^2)+sum(theta2(:).^2));
J=J+reg;

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m).*theta1;
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m).*theta2;

% =========================================================================
% Unroll gradients
%   The returned variable grad is an "unrolled" vector of the
%   partial derivatives of the neural network.

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
