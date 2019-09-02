function [MSE_Mean, J] = errCalculator(Y1, Y2)
% function [MSE_Real,MSE_Img,MSE_Mean J] = errCalculator(Y1, Y2)
%   [MSE_Mean,J] = errCalculator(Y1, Y2)
% This function calculate the MSE (Minumum Square Erros) of the two Input
% Vectors (Y1 and Y2), as in this case one column of Y* correspond to a
% real part and the other to an imaginary part the function rteurn two
% values of MSE, the real and the imaginary as their mean value, returns 
% also the value of the correlation between the two matrixes (R), a value
% close to 1 shows a good correlation and close to 0 a bad correlation.

m = size(Y1,1);
% Real Error
% MSE_Real = (1/m *sum(((Y1(:,1) - Y2(:,1)).^2)));
% Imaginary Error
% MSE_Img  = (1/m *sum(((Y1(:,2) - Y2(:,2)).^2)));

% valor para a cost function
J = sum(( (Y1(:,1) - Y2(:,1)).^2 + (Y1(:,2) - Y2(:,2)).^2 ))/(2*m);

%MSE
% MSE_Mean1 = sum(sqrt( (Y1(:,1) - Y2(:,1)).^2 + (Y1(:,2) - Y2(:,2)).^2 ))/(m);
MSE_Real = sum( (Y1(:,1) - Y2(:,1)).^2 )./m;
MSE_Imag = sum( (Y1(:,2) - Y2(:,2)).^2 )./m;
MSE_Mean = mean([MSE_Real MSE_Imag]);
end

