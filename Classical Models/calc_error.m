function [confmat,Accuracy,Precision,Recall,F1_score]=calc_error(x1,x2)

 % ��������Confusion matrix
confmat = confusionmat(x1,x2);

disp(' ')
disp('1.��������........')
disp(confmat)

% ׼ȷ��Accuracy
disp(' ')
Accuracy= sum((x1 == x2))  / length(x1)  ;
disp(['2.׼ȷ��Accuracy=',num2str(Accuracy)])

% ��ȷ��Precision
disp(' ')
Precision=confmat(1,1)/sum(confmat(:,1));
disp(['3.��ȷ��Precision=',num2str(Precision)])

% �ٻ���Recall
disp(' ')
Recall=confmat(1,1)/sum(confmat(1,:));
disp(['4.�ٻ���Recall=',num2str(Recall)])

% ����ƽ����F1_score
disp(' ')
F1_score=2*(Precision*Recall)/(Precision+Recall);
disp(['5.����ƽ����F1_score=',num2str(F1_score)])
end

