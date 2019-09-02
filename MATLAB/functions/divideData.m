function [X_train, X_val, X_test, ...
    Y_train, Y_val, Y_test] = divideData(X, Y, p_train, p_val)
% Divide the X and Y data in the p_train for training data, p_val for
% validation data and 1 - (p_train + p_val) for testing

try
m = size(X,1);
%m2 = size(X,2);

index_start = round(m.*rand(1));

train_length = round(p_train * m);
val_length   = round(p_val * m);
test_length  = round((1-p_train-p_val) * m);

% X_train = zeros(train_length,m2);
% X_val   = zeros(val_length,m2);
% X_test  = zeros(test_length,m2);
% 
% Y_train = zeros(train_length,m2);
% Y_val   = zeros(val_length,m2);
% Y_test  = zeros(test_length,m2);

% ===== TRAIN ====

tmp = (index_start + train_length);
if (tmp > m)
    tmp1 = tmp - m;
    X_train = [X(index_start:m,:); X(1:tmp1-1,:)];
    Y_train = [Y(index_start:m,:); Y(1:tmp1-1,:)];
    index_start = tmp1;
else
    X_train = X(index_start:tmp-1,:);
    Y_train = Y(index_start:tmp-1,:);
    index_start = tmp;
end

% ===== VAL ====

tmp = (index_start + val_length);
if (tmp > m)
    tmp1 = tmp - m;
    X_val = [X(index_start:m,:); X(1:tmp1-1,:)];
    Y_val = [Y(index_start:m,:); Y(1:tmp1-1,:)];
    index_start = tmp1;
else
    X_val = X(index_start:tmp-1,:);
    Y_val = Y(index_start:tmp-1,:);
    index_start = tmp;
end

% ===== TEST ===

tmp = (index_start + test_length);
if (tmp > m)
    tmp1 = tmp - m;
    X_test = [X(index_start:m,:); X(1:tmp1-1,:)];
    Y_test = [Y(index_start:m,:); Y(1:tmp1-1,:)];
    index_start = tmp1;
else
    X_test = X(index_start:tmp-1,:);
    Y_test = Y(index_start:tmp-1,:);
    index_start = tmp;
end

catch e
    [X_train, X_val, X_test, ...
        Y_train, Y_val, Y_test] = divideData(X, Y, p_train, p_val);
end

end

