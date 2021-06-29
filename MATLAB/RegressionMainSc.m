%% ====================== Start parallel pool ========================== %%
% a = gcp; if(~a.Connected),parpool,end
%% Main program to call the training of a Regression Neural Network
clear all;
close all;
restoredefaultpath;
clc;
%% Load the needed values
addpath(genpath('./functions/'))
addpath('./input_data/')
addpath('./tools/')
datapath = 'raw_data/';
RxPw = -5;
Spans = 1;
nSc = 1;
string_name =  [num2str(RxPw) 'dBm_' num2str(Spans) 'spans'];
filename = ['Data_' string_name];
string_res   = sprintf('./results/results_%s.mat',string_name); 
load(['./' datapath filename]);
load(['./' datapath 'IQmap.mat']);

%% Select SubCarrier and M-QAM Format
Stx1 = Stx;
Srx1 = Srx;
for sc = 1:nSc

nSc = sc;
Srx = Srx1{nSc}(1,:);
Stx = Stx1{nSc}(1,:);
sSc = ['Sc' num2str(nSc)];
%% Adjust for precision
IQmap = single(IQmap);
Stx   = single(Stx);
%% Parameters 
Mqam  = length(IQmap);      % Modulation format 
iter  = 100;                % # of Iterations  
pdata = 0.15;               % Data Percentage
%% ===  Definition of the data percentage for train and validation ===== %%
p_train = 0.60; % Train data percentage
p_val = 0.20;   % Train data percentage
%% ========== Check if the gradients are being well calculated ========= %%
checkGrad(Srx,Stx,4,p_train, p_val); 
%% ==================== Define the search space ======================== %%
nSamples = 1:2:15;         % # of Samples to evaluate
lambda = [0 0.1 1 10];       % Regulatization Constant
nodes = [ 5 10 15 20 35 50]; % # of Neurons to evaluate
%% =================== Size of variables shortcut ====================== %%
nSamples_size = length(nSamples);
lambda_size   = length(lambda);
nodes_size    = length(nodes);
%% ==================== Initialize error variables ===================== %%
MSE_train = zeros(nSamples_size,lambda_size,nodes_size);
MSE_val   = zeros(nSamples_size,lambda_size,nodes_size);

%% ==== Cut data vectors and and convert the data in ANN format! ======= %%
% This has to be done just once, uncomment if you change the samples search
% space or the percentage of data or the input data

% Cut vectors to pretended size
Stx = Stx(1:round(pdata*end));
Srx = Srx(1:round(pdata*end));

dataConverterRegressionSave(Srx,Stx,nSamples);

%% =============== Analize the best Architecture  ====================== %%

timeET = tic; 
numIter = nSamples_size * lambda_size * nodes_size;

parfor i = 1:nSamples_size
    % ========== Name of data variable ========== %
    s = sprintf('Data%iSamples.mat',2*nSamples(i));
    DATA{i}=load(s);
    X = DATA{i}.X;
    Y = DATA{i}.Y;
    % ============ Divide the data ============== %
    [X_Train, X_Val, ~, Y_Train, Y_Val, ~] = ...
        divideData(X, Y, p_train, p_val);
    
    for j = 1:lambda_size
        nSamples_aux = nSamples(i);
        lambda_aux = lambda(j);
        for k = 1:nodes_size
            % ========= Train the ANN and calculate the errors ========== %
            [MSE_train(i,j,k),MSE_val(i,j,k)] = trainANN(X_Train,...
                Y_Train,nSamples_aux,nodes(k),lambda_aux,iter,X_Val,Y_Val);
        end
    end
end

train_time = toc(timeET);

%% ==================== Get the best values ============================ %%
[~, idxSamp] = min(MSE_val(:));
[r,c,p] = ind2sub(size(MSE_val),idxSamp);
nSamplesBest = nSamples(r);
lambdaBest   = lambda(c);
nodesBest    = nodes(p);
%% ================= Prepare data for final train ====================== %%

% ============ Load the data for the best value of Samples ============== %
s = sprintf('Data%iSamples.mat',2*nSamplesBest);
load(s);    
    
[X_Train, X_Val, X_Test, Y_Train, Y_Val, Y_Test] = ...
    divideData(X, Y, p_train, p_val);

%% ============== Train the ANN with the best values =================== %%

iter = 4000;
timeEF = tic;
% ====================== Train the ANN a first time ===================== %
[Theta1, Theta2] = ANN_Regression(X_Train,Y_Train,nSamplesBest,...
    nodesBest,lambdaBest,iter);
         
% ============= Join the validation data with the train data ============ %         
X_2 = [X_Train ; X_Val];
Y_2 = [Y_Train ; Y_Val];

% ================= Train the ANN a second time ========================= %
[Theta1F, Theta2F] = ANN_Regression(X_2,Y_2,nSamplesBest,nodesBest,...
    lambdaBest,iter,'theta',Theta1,Theta2);

sCost = ['Samples' num2str(nSamplesBest)  ...
        'lambda' num2str(lambdaBest)  ...
        'nodes' num2str(nodesBest) ];
sCost = strrep(sCost,'.','_');
% CostFinalTrain.(sCost) = costResidual;

 time_final_train = toc(timeEF);   
% ================== Predict the test data ============================== %
Y_test_pred   = predictReg(Theta1F,Theta2F,X_Test);

% ==================== Calculate the error ============================== %
[MSE_test_mean,~] = errCalculator(Y_test_pred, Y_Test);

% =============== Save the variables of interest ======================== %

[time_string_train, ~]=timeConverter(train_time);
[time_string_final_train, ~]=timeConverter(time_final_train);

res.(sSc).Pesos.Theta1 = Theta1F;
res.(sSc).Pesos.Theta2 = Theta2F;
res.(sSc).Erros.Train = MSE_train;
res.(sSc).Erros.Val   = MSE_val;
res.(sSc).Erros.Test  = MSE_test_mean;
res.(sSc).Arch.Samples = nSamplesBest;
res.(sSc).Arch.lambda  = lambdaBest;
res.(sSc).Arch.nodes   = nodesBest;
res.(sSc).HyperParam.Samples = nSamples;
res.(sSc).HyperParam.lambda = lambda;
res.(sSc).HyperParam.nodes = nodes;
res.(sSc).TimeMin.SearchTrain = time_string_train;
res.(sSc).TimeMin.FinalTrain  = time_string_final_train;
res.(sSc).DataSetSize.Train = size(X_Train,1);
res.(sSc).DataSetSize.Validation = size(X_Val,1);
res.(sSc).DataSetSize.Test = size(X_Test,1);



end

res.nSC = nSc;
save(string_res,'res');






