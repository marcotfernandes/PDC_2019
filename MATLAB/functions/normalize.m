function [out, mX, stdX] = normalize(X,mX,stdX)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

if(isempty(mX))
    mX = mean(X);
    stdX = std(X);
end
out = (X-mX)./stdX;
end

