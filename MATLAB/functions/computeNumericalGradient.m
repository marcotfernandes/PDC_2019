function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: vector of parameters
% J: function that outputs the cost (a real-number). 
%Calling y = J(theta) will return the cost at theta. 
 
% Initialize numgrad with zeros
theta=theta(:);
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Implement numerical gradient checking, and return the result in numgrad.  
% Write a code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: Since theta is a large vector, check the gradients of only the first 
%10 elements of theta.  
eps=0.0001;

for i=1:10
    e=zeros(size(theta));
    e(i)=1;
    numgrad(i)=( J(theta+eps*e)-J(theta-eps*e) )./(2*eps);
end
