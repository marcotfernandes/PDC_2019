clear
close all
clc

addpath('../');
Nodes = [10 14 18 22];

for i = 1:numel(Nodes)
    file_name = sprintf('Results_-5dBm_1spans_par_%i.mat',Nodes(i));
    DATA{i} = load(file_name);
end

hFig = figure;
hold on

for i=1:numel(Nodes)
    Tprog(i) = DATA{i}.Time.Prog;
    Tpar(i) = DATA{i}.Time.Par;
    Tpar(i) = DATA{i}.Time.Gather;
end

hPlot(1) = plot(Nodes,Tpar);