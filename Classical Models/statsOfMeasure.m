function [stats] = statsOfMeasure(confusion)
confusion = confusion';
% confusion: 3x3 confusion matrix
tp = [];
fp = [];
fn = [];
tn = [];
len = size(confusion, 1);
for k = 1:len
    % True positives           % | x o o |
    tp_value = confusion(k,k); % | o o o |
    tp = [tp, tp_value];       % | o o o |
    
    % False positives                          % | o x x |
    fp_value = sum(confusion(k,:)) - tp_value; % | o o o |
    fp = [fp, fp_value];                       % | o o o |
    
    % False negatives                          % | o o o |
    fn_value = sum(confusion(:,k)) - tp_value; % | x o o |
    fn = [fn, fn_value];                       % | x o o |
    
    % True negatives (all the rest)                                    % | o o o |
    tn_value = sum(sum(confusion)) - (tp_value + fp_value + fn_value); % | o x x |
    tn = [tn, tn_value];                                               % | o x x |
end

% Statistics of interest for confusion matrix
prec = tp ./ (tp + fp); % precision
sens = tp ./ (tp + fn); % sensitivity, recall
spec = tn ./ (tn + fp); % specificity
acc = sum(tp) ./ sum(sum(confusion));
f1 = (2 .* prec .* sens) ./ (prec + sens);

% For micro-average
microprec = sum(tp) ./ (sum(tp) + sum(fp)); % precision
microsens = sum(tp) ./ (sum(tp) + sum(fn)); % sensitivity, recall
microspec = sum(tn) ./ (sum(tn) + sum(fp)); % specificity
microacc = acc;
microf1 = (2 .* microprec .* microsens) ./ (microprec + microsens);

% Names of the rows
name = ["true_positive"; "false_positive"; "false_negative"; "true_negative"; ...
    "precision"; "sensitivity"; "specificity"; "accuracy"; "F-measure"];

% Names of the columns
varNames = ["name"; "classes"; "macroAVG"; "microAVG"];

% Values of the columns for each class
values = [tp; fp; fn; tn; prec; sens; spec; repmat(acc, 1, len); f1];

% Macro-average
macroAVG = mean(values, 2);

% Micro-average
microAVG = [macroAVG(1:4); microprec; microsens; microspec; microacc; microf1];

% OUTPUT: final table
stats = table(name, values, macroAVG, microAVG, ...
    'VariableNames',cellstr(varNames));
end