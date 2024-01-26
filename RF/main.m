%%  Clear environment variables
warning off           
close all               
clear                
clc                     

%% add path
addpath('ClassRF\')

%%  Read data
res = xlsread('data2.xlsx');

%%  Analyze data
num_class = length(unique(res(:, end)));  % Number of categories 
num_res = size(res, 1);                   % Number of samples 
num_size = 0.8;                           % Proportion of training set to data set
flag_conusion = 1;                      

%%  Set variables to store data
P_train = []; P_test = [];
T_train = []; T_test = [];

%% Partition the data set
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           
    mid_size = size(mid_res, 1);                    
    mid_tiran = round(num_size * mid_size);         

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % Training set input
    T_train = [T_train; mid_res(1: mid_tiran, end)];              %Training set output

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  %Test set input
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % Test set output
end

%% Data transposition
P_train = P_train'; P_test = P_test';
T_train = T_train'; T_test = T_test';

%%  Get the number of training set and test samples
M = size(P_train, 2);
P = size(P_test , 2);

%% Data normalization
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);

t_train = T_train;
t_test  = T_test;

%%  Data transposition
p_train = p_train'; p_test = p_test';
t_train = t_train'; t_test = t_test';

% %%  parameter settings
ntree=10;
mtry=1;

%%Modeling

model = classRF_train(p_train, t_train, ntree, mtry);

%%  Feature importance
importance = model.importance';

%%  Simulation test
[T_sim1, Vote1] = classRF_predict(p_train, model);
[T_sim2, Vote2] = classRF_predict(p_test , model);
%% prob1 prob2
prob1 = Vote1./ntree;
prob2 = Vote2./ntree;

%% Test set ROC curve
[tpr,fpr,thresholds] = roc(full(ind2vec(T_test)),prob2');
AUC =  trapz([0 fpr{1} 1],[0 tpr{1} 1]);
x_dig=0:0.1:1;
y_dig=x_dig;
h=figure;
set(h,'units','normalized','position',[0.1 0.1 0.48 0.8]);
set(h,'color','w');%设
plot(([0, fpr{1},1]),([0,tpr{1},1]),'LineWidth',3,'MarkerSize',3);hold on;
plot(x_dig,y_dig,'--','LineWidth',1.5);
xlabel('False Positive Ratio (1-specificity)','fontsize',2,'FontWeight','bold');%x轴
ylabel('True Positive Ratio (Sensitivity)','fontsize',2,'FontWeight','bold');%y轴
set(gca,'YLim',[0,1.02]);
set(gca,'XLim',[-0.01,1.01]);
set(gca,'FontSize',24,'LineWidth',1.6)
set(get(gca,'YLabel'),'FontSize',24);
set(get(gca,'XLabel'),'FontSize',24);
set(gca,'YTick',0:0.2:1)
grid off
ROCtitle={['AUC = ',num2str(AUC,'%.4f')]};
hh=legend(ROCtitle,'Location','southeast')%,'Location','southeast');
set(hh,'edgecolor','white');
print('figure1', '-dpng', '-r600');

%% Training set ROC curve
[tpr,fpr,thresholds] = roc(full(ind2vec(T_train)),prob1');
AUC =  trapz([0 fpr{1} 1],[0 tpr{1} 1]);
x_dig=0:0.1:1;
y_dig=x_dig;
h=figure;
set(h,'units','normalized','position',[0.1 0.1 0.48 0.8]);
set(h,'color','w');
plot(([0, fpr{1},1]),([0,tpr{1},1]),'LineWidth',3,'MarkerSize',3);hold on;
plot(x_dig,y_dig,'--','LineWidth',1.5);
xlabel('False Positive Ratio (1-specificity)','fontsize',2,'FontWeight','bold');%x轴
ylabel('True Positive Ratio (Sensitivity)','fontsize',2,'FontWeight','bold');%y轴
set(gca,'YLim',[0,1.02]);
set(gca,'XLim',[-0.01,1.01]);
set(gca,'FontSize',24,'LineWidth',1.6)
set(get(gca,'YLabel'),'FontSize',24);
set(get(gca,'XLabel'),'FontSize',24);
set(gca,'YTick',0:0.2:1)
grid off
ROCtitle={['AUC = ',num2str(AUC,'%.4f')]};
hh=legend(ROCtitle,'Location','southeast')%,'Location','southeast');
set(hh,'edgecolor','white');
print('figure1', '-dpng', '-r600');
%%  性能评价
error1 = sum((T_sim1' == T_train)) / M * 100 ;
error2 = sum((T_sim2' == T_test )) / P * 100 ;

%% Plot feature importance
figure
bar(importance)
legend('importance')
xlabel('feature')
ylabel('importance')

figure
plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
legend('Actual value', 'Predictive value')
xlabel('Prediction sample')
ylabel('Prediction result')
string = {'Comparison of training set prediction results'; ['Accuracy=' num2str(error1) '%']};
title(string)
grid

figure
plot(1: P, T_test, 'r-*', 1: P, T_sim2, 'b-o', 'LineWidth', 1)
legend('Actual value', 'Predictive value')
xlabel('Prediction sample')
ylabel('Prediction result')
string = {'Comparison of test set prediction results'; ['Accuracy=' num2str(error2) '%']};
title(string)
grid

%% Confusion matrix
if flag_conusion == 1

    figure
    cm = confusionchart(T_train, T_sim1);
    cm.Title = 'Confusion Matrix for Train Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
    
    figure
    cm = confusionchart(T_test, T_sim2);
    cm.Title = 'Confusion Matrix for Test Data';
    cm.ColumnSummary = 'column-normalized';
    cm.RowSummary = 'row-normalized';
end
disp('-------------------------------------------------------------')
disp('RF Training set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(T_train, T_sim1);
fprintf('\n')
disp('-------------------------------------------------------------')
disp('RF Test set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(T_test, T_sim2);
fprintf('\n')

