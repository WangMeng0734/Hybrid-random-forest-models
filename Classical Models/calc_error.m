function [confmat,Accuracy,Precision,Recall,F1_score]=calc_error(x1,x2)

 % 混淆矩阵Confusion matrix
confmat = confusionmat(x1,x2);

disp(' ')
disp('1.混淆矩阵........')
disp(confmat)

% 准确率Accuracy
disp(' ')
Accuracy= sum((x1 == x2))  / length(x1)  ;
disp(['2.准确率Accuracy=',num2str(Accuracy)])

% 精确率Precision
disp(' ')
Precision=confmat(1,1)/sum(confmat(:,1));
disp(['3.精确率Precision=',num2str(Precision)])

% 召回率Recall
disp(' ')
Recall=confmat(1,1)/sum(confmat(1,:));
disp(['4.召回率Recall=',num2str(Recall)])

% 调和平均数F1_score
disp(' ')
F1_score=2*(Precision*Recall)/(Precision+Recall);
disp(['5.调和平均数F1_score=',num2str(F1_score)])
end

