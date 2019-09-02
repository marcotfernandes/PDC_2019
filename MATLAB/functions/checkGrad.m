function checkGrad(Srx,Stx,nSamples,percentage_train, ...
    percentage_validation)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% =========== Convert the data ============== %
[X, Y] = dataConverterRegression(Srx,Stx,nSamples);
% ============ Divide the data ============== %
[X_Train, ~, ~, Y_Train, ~, ~] = ...
    divideData(X, Y, percentage_train, percentage_validation);    
lambda=0;
hidden_layer_size=5;
input_layer_size=size(X_Train,2);
output_layer_size=size(Y_Train,2);

Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);

% Now Output layer is 2, I and Q values
Theta2 = randInitializeWeights(hidden_layer_size, output_layer_size);
theta = [Theta1(:);Theta2(:)];

[~, grad] = nnCostFunction(theta, input_layer_size,...
    hidden_layer_size, output_layer_size, X_Train, Y_Train,lambda);

numGrad = computeNumericalGradient( @(x) nnCostFunction(x, ...
    input_layer_size,  hidden_layer_size, output_layer_size, X_Train,...
    Y_Train,lambda), theta);

% Use this to visually compare the gradients side by side
% disp([numGrad(1:10) grad(1:10)]); 

% Compare numerically computed gradients with those computed analytically
diff = norm(numGrad(1:10)-grad(1:10))/norm(numGrad(1:10)+grad(1:10));
% s = ...
% sprintf('Difference between calculated gradient and theoretical %.2d',diff);
% disp(s);   % The difference should be small

if(diff > 1e-8)
    fprintF('Gradient error \n')
    return
end

end

