function [confmat,Accuracy,Precision,Recall,F1_score]=calc_error(x1,x2)

 % Confusion matrix
confmat = confusionmat(x1,x2);

disp(' ')
disp('1.Confusion matrix........')
disp(confmat)

% Accuracy
disp(' ')
Accuracy= sum((x1 == x2'))  / length(x1)  ;
disp(['2.Accuracy=',num2str(Accuracy)])

% Precision
disp(' ')
Precision=confmat(1,1)/sum(confmat(:,1));
disp(['3.Precision=',num2str(Precision)])

% Recall
disp(' ')
Recall=confmat(1,1)/sum(confmat(1,:));
disp(['4.Recall=',num2str(Recall)])

% F1_score
disp(' ')
F1_score=2*(Precision*Recall)/(Precision+Recall);
disp(['5.F1_score=',num2str(F1_score)])
end

