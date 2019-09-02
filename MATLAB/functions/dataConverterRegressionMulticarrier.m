function [X, Y] = dataConverterRegressionMulticarrier(Srx, Tx, nSamples,Subcarrier)
% [X Y] = dataConverterRegression(Srx,Tx,nSamples)
% In this function the inputs must be a vector with complex numbers
% corresponding to the received signal (Srx) and a vector with the
% corresponding class value (Tx), nSamples is the number of samples to be
% used.
  
   carrier = size(Srx,2);
   

        m = size(Srx{1},2) - 2*nSamples;
        X = zeros(m,2*nSamples*carrier);
        
        Y = zeros(m,2);
        Y(:,1) = real(Tx{Subcarrier}(1:m));
        Y(:,2) = imag(Tx{Subcarrier}(1:m));
        
        
        for l = 1:carrier
            
            tmp(1,:) = real(Srx{l}(1,:));
            tmp(2,:) = imag(Srx{l}(1,:));
            
            
            sizeSamp = nSamples;
            tmpI = 2*l - 1;
            tmpF = sizeSamp*2 + tmpI - 1;
            X(:,tmpI:tmpF) = dataConversionHelper(m,sizeSamp,tmp);
        end
       
   end
