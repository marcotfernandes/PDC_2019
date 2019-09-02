function [X Y] = dataConverterRegressionMiddle(Srx, Tx, nSamples)

    m = size(Srx,2) - 2*nSamples;
    X = zeros(m,2*nSamples);
    tmp(1,:) = real(Srx);
    tmp(2,:) = imag(Srx);

    ind = (nSamples+1)/2;
    Y = zeros(m-ind-1,2);
    Y(:,1) = real(Tx(ind:m));
    Y(:,2) = imag(Tx(ind:m));

    flag = 1;

    for i=1:m
        cnt = i;
        for j=1:2*nSamples
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