%% Clear variables
clc;clear                               
close all                                 

%% Read data
data=xlsread("data.xlsx");
input0=data(:,2:end);                     
output=data(:,1);                       
Num=length(output);                       

%% 切分数据集
c = cvpartition(output,"HoldOut",0.2);    % Data set split ratio setting
trainingIndices = training(c);            % training set index
testIndices = test(c);                    % Test set index
XTrain = input0(trainingIndices,:);       
YTrain = output(trainingIndices);         
Num_trian=length(YTrain);                 
XTest = input0(testIndices,:);            
YTest = output(testIndices);              

%% Check whether there is a categorical variable in the feature (independent variable)
T = input('Enter the classification feature position [ ], if there is no classification feature, enter 0: ');      

%% Check the cross-validation fold
K = 5;                                    

%% Train individual machine learning models

% SVM
[trainedClassifier1, validationAccuracy1,validationPredictions1,validationScores1] = Train_SVM(XTrain, YTrain,K,T,6,680);
[Tsim1,Tsimscore1]=trainedClassifier1.predictFcn(XTest);
% KNN
[trainedClassifier3, validationAccuracy3,validationPredictions3,validationScores3] = Train_KNN(XTrain, YTrain,K,T,1);
[Tsim3,Tsimscore3]=trainedClassifier3.predictFcn(XTest);
% BP
[trainedClassifier4, validationAccuracy4,validationPredictions4,validationScores4] = Train_BP(XTrain, YTrain,K,T,14,0.01);
[Tsim4,Tsimscore4]=trainedClassifier4.predictFcn(XTest);

%% Performance indicator output of model training
%% Model training set performance indicators
C1 = confusionmat(YTrain, validationPredictions1) ;    
A=statsOfMeasure(C1);          
C3 = confusionmat(YTrain, validationPredictions3) ;    
C=statsOfMeasure(C3);  
C4 = confusionmat(YTrain, validationPredictions4) ;       
D=statsOfMeasure(C4);  
% Write to excel table
writetable(A,'Model performance indicator output (validation).xls','Sheet','SVM');
writetable(C,'Model performance indicator output (validation).xls','Sheet','KNN');
writetable(D,'Model performance indicator output (validation).xls','Sheet','BPNN');

%% Model test set performance indicators
C1 = confusionmat(YTest, Tsim1) ;       
A=statsOfMeasure(C1);           
C3 = confusionmat(YTest, Tsim3) ;       
C=statsOfMeasure(C3);  
C4 = confusionmat(YTest, Tsim4) ;       
D=statsOfMeasure(C4);  
% 写入excel表
writetable(A,'Model performance indicator output (test).xls','Sheet','SVM');
writetable(C,'Model performance indicator output (test).xls','Sheet','KNN');
writetable(D,'Model performance indicator output (test).xls','Sheet','BPNN');


%% Confusion matrix (validation)
figure
subplot(2,2,1);
cm1 = confusionchart(YTrain, validationPredictions1);
cm1.Title = 'SVM model cross validation confusion matrix';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';

subplot(2,2,3);
cm1 = confusionchart(YTrain, validationPredictions3);
cm1.Title = 'KNN model cross validation confusion matrix';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';

subplot(2,2,4);
cm1 = confusionchart(YTrain, validationPredictions4);
cm1.Title = 'BPNN model cross validation confusion matrix';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';

%% ROC curve (validation)
P_auc = 1;       
[x1,y1,~,auc1] = perfcurve(YTrain,validationScores1(:,P_auc),P_auc);
[x3,y3,~,auc3] = perfcurve(YTrain,validationScores3(:,P_auc),P_auc);
[x4,y4,~,auc4] = perfcurve(YTrain,validationScores4(:,P_auc),P_auc);
figure
plot(x1,y1,'LineWidth',1.5);  
hold on
plot(x3,y3,'LineWidth',1.5);  
hold on
plot(x4,y4,'LineWidth',1.5);  
hold on
plot(x1,x1,'--','Color','k','linewidth',1.5)
xlabel('False Positive Rate','FontSize',14,'Fontname','Times New Roman');  
ylabel('True Positive Rate','FontSize',14,'Fontname','Times New Roman');  
title('Model training cross-validation ROC curve comparison'); 
legend(['SVM-AUC=',num2str(auc1)],['KNN-AUC=',num2str(auc3)],['BPNN-AUC=',num2str(auc4)],Location="southeast")
set(gca,'Box','off','FontSize',12,'LineWidth',1.5);
axis([-0.02,1,-0.02,1]) 


%% Confusion matrix (test)
figure
subplot(2,2,1);
cm1 = confusionchart(YTest, Tsim1);
cm1.Title = 'SVM model test set confusion matrix';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized'

subplot(2,2,3);
cm1 = confusionchart(YTest, Tsim3);
cm1.Title = 'KNN model test set confusion matrix';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';

subplot(2,2,4);
cm1 = confusionchart(YTest, Tsim4);
cm1.Title = 'BPNN model test set confusion matrix';
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';

%% ROC curve (test)
[x1,y1,~,auc1] = perfcurve(YTest,Tsimscore1(:,P_auc),P_auc);
[x3,y3,~,auc3] = perfcurve(YTest,Tsimscore3(:,P_auc),P_auc);
[x4,y4,~,auc4] = perfcurve(YTest,Tsimscore4(:,P_auc),P_auc);
figure
plot(x1,y1,'LineWidth',1.5);  
hold on
plot(x3,y3,'LineWidth',1.5);  
hold on
plot(x4,y4,'LineWidth',1.5);  
hold on
plot(x1,x1,'--','Color','k','linewidth',1.5)
xlabel('False Positive Rate','FontSize',14,'Fontname','Times New Roman');  
ylabel('True Positive Rate','FontSize',14,'Fontname','Times New Roman');  
title('Model test set ROC curve comparison'); 
legend(['SVM-AUC=',num2str(auc1)],['KNN-AUC=',num2str(auc3)],['BPNN-AUC=',num2str(auc4)],Location="southeast")
set(gca,'Box','off','FontSize',12,'LineWidth',1.5);
axis([-0.02,1,-0.02,1]) 

disp('-------------------------------------------------------------')
disp('SVM training set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(YTrain,validationPredictions1 );
fprintf('\n')
disp('-------------------------------------------------------------')
disp('SVM test set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(YTest, Tsim1);
fprintf('\n')
disp('-------------------------------------------------------------')
disp('KNN training set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(YTrain,validationPredictions3 );
fprintf('\n')
disp('-------------------------------------------------------------')
disp('KNN test set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(YTest, Tsim3);
fprintf('\n')
disp('-------------------------------------------------------------')
disp('BPNN training set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(YTrain,validationPredictions4);
fprintf('\n')
disp('-------------------------------------------------------------')
disp('BPNN test set error index')
[confmat,Accuracy,Precision,Recall,F1_score]=calc_error(YTest, Tsim4);
fprintf('\n')

disp('-----------------------The code has finished running--------------------------')