clear
close all
clc

Nodes = [1 2 4 6 8 10 14 18 22];

for i = 1:numel(Nodes)
    file_name = sprintf('Results_-5dBm_1spans_par_%i.mat',Nodes(i));
    DATA{i} = load(file_name);
end

hFig = figure;
hold on

for i=1:numel(Nodes)
    Tprog(i) = DATA{i}.Time.Prog;
    Tpar(i) = DATA{i}.Time.Par;
end

hPlot(1) = plot(Nodes,Tpar./Tprog);
