function stop = outfun(x,optimValues,state)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% global costResidual;
% global costResNorm;
% stop = 0;
% costResidual(optimValues.iteration+1) = optimValues.residual;
% costResNorm(optimValues.iteration+1) = optimValues.resnorm;
global Cost;


Cost(optimValues.iteration+1) = optimValues.fval;

if(optimValues.iteration > 0)
    errPerIter(x,optimValues.iteration);
end

end

