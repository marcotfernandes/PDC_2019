function X = dataConversionHelper(m,sizeSamp,tmp)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    for i=1:m            
        flag = 1;
        cnt = i;
        for j=1:2*sizeSamp
            if(flag)
                X(i,j) = tmp(1,cnt);
            else
                X(i,j) = tmp(2,cnt);
                cnt = cnt + 1;
            end
            flag = ~flag;
        end
    end
end

