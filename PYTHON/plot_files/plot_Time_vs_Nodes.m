clear all;
clear global;
close all;
clc;

%% 

addpath('../functions-plot/');
RGB = fancyColors();
color = {RGB.itblue, RGB.itred, RGB.green, RGB.violet, RGB.orange, ...
    RGB.black, RGB.gray, RGB.cyan, RGB.pink, RGB.yellow};
marker = {'o','square','^','pentagram','hexagram','+','v','x','>','.'};

%% load DATA

Nodes = [1 2 4 6 8 10 14 18 22];
path = ["../second_results/"];

for j = 1:length(path)
    for i = 1:numel(Nodes)
        file_name = sprintf('%sResults_-5dBm_1spans_par_%i.mat',...
            path(j),Nodes(i));
        DATA{j}{i} = load(file_name);
    end
end

%% plot DATA

hFig = figure;
hold on

for j = 1:length(path)
    for i=1:numel(Nodes)
        Tprog(j,i) = DATA{j}{i}.Time.Prog;
        Tpar(j,i) = DATA{j}{i}.Time.Par;
        Tscatter(i) = DATA{j}{i}.Time.Scatter(1);
        Tgather(i) = DATA{j}{i}.Time.Gather(1);
    end
end

Tamdahl = Amdahl_Law(Tprog(1),Tpar(1),Tprog(1)-Tpar(1),Nodes);
y_eval = Tprog(1)./(Tprog-Tscatter-Tgather);

hPlot(1) = plot(Nodes,Tamdahl);
hPlot(2) = plot(Nodes,y_eval);
hPlot(3) = plot(Nodes,ones(1,length(Nodes)));

set(hPlot(1),'color',color{2},'linestyle','-',...
        'linewidth',1.5,'marker','none',...
        'markersize',7,'markerfacecolor','w');
set(hPlot(2),'color',color{1},'linestyle','--',...
    'linewidth',1.9,'marker',marker{1},...
    'markersize',7,'markerfacecolor','w');
set(hPlot(3),'color',RGB.black,'linestyle','--',...
        'linewidth',1,'marker','none',...
        'markersize',7,'markerfacecolor','w');



%%
xLim = [min(Nodes) max(Nodes)];
yLim = [0 1.01*max(Tamdahl)];

xlabel('Number of CPU cores','Interpreter','latex','FontSize',11);
ylabel('SpeedUp','Interpreter','latex','FontSize',11);
xAxis = get(gca,'xaxis');
set(xAxis,'TickLabelInterpreter','latex','FontSize',12,...
    'TickValues',1:5:25);
yAxis = get(gca,'yaxis');
set(yAxis,'TickLabelInterpreter','latex','FontSize',12);
axis([xLim yLim]);

set(gca,'YScale','linear','YMinorGrid','off','Box','on',...
    'PlotBoxAspectRatio',[1 0.8 1]);

grid on;
set(gca,'GridLineStyle','--','XMinorTick','off','XMinorGrid','on');

str_leg = {'Theoretical','Experimental'};

hLeg = legend(str_leg,'Location','NorthWest');
set(hLeg,'Interpreter','latex','fontsize',9,'color','w',...
    'edgecolor',RGB.black);


