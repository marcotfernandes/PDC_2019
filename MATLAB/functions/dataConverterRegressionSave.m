function dataConverterRegressionSave(Srx, Tx, nSamples)
% [X Y] = dataConverterRegression(Srx,Tx,nSamples)
% In this function the inputs must be a vector with complex numbers
% corresponding to the received signal (Srx) and a vector with the
% corresponding class value (Tx), nSamples is the number of samples to be
% used.
    
   nSamplesSize = length(nSamples);
   
   for k =1:nSamplesSize

        m = size(Srx,2) - 2*nSamples(k);
        X = zeros(m,2*nSamples(k));
        tmp(1,:) = real(Srx);
        tmp(2,:) = imag(Srx);

        Y = zeros(m,2);
        Y(:,1) = real(Tx(1:m));
        Y(:,2) = imag(Tx(1:m));

        flag = 1;

        for i=1:m
            cnt = i;
            for j=1:2*nSamples(k)
                if(flag)
                    X(i,j) = tmp(1,cnt);
                else
                    X(i,j) = tmp(2,cnt);
                    cnt = cnt + 1;
                end
                flag = ~flag;
            end
        end
        s = sprintf('./input_data/Data%iSamples',2*nSamples(k));
        save(s,'X','Y');
   end
end
