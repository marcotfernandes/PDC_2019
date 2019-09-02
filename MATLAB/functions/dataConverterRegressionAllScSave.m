function dataConverterRegressionAllScSave(Srx, Tx, nSamples)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

for k = 1:length(nSamples)
    [X1, Y] = dataConverterRegression(Srx{1}, Tx{1}, nSamples(k));
    [X2, ~] = dataConverterRegression(Srx{2}, Tx{2}, nSamples(k));
    [X3, ~] = dataConverterRegression(Srx{3}, Tx{3}, nSamples(k));
    [X4, ~] = dataConverterRegression(Srx{4}, Tx{4}, nSamples(k));
    [X5, ~] = dataConverterRegression(Srx{5}, Tx{5}, nSamples(k));
    [X6, ~] = dataConverterRegression(Srx{6}, Tx{6}, nSamples(k));
    [X7, ~] = dataConverterRegression(Srx{7}, Tx{7}, nSamples(k));
    [X8, ~] = dataConverterRegression(Srx{8}, Tx{8}, nSamples(k));

    X = [X1 X2 X3 X4 X5 X6 X7 X8];

    s = sprintf('./dataAllSc/Data%iSamples',2*nSamples(k));
    save(s,'X','Y');
end

end

