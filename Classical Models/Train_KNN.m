function [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = Train_KNN(trainingData, responseData,K,T,NeiN)
%% KNNK最邻近算法
%% 距离度量：欧几里得距离
%% 优化超参数为领点个数(1~N-1)
num_p = size(trainingData,2);    % 特征数量
au = unique(responseData);
num_T = length(au);              % 分类数
Var_name={};
for i=1:num_p
    Var_name{1,i}=num2str(i);
end
inputTable = array2table(trainingData, 'VariableNames', Var_name);
predictorNames = Var_name;
predictors = inputTable(:, predictorNames);
response = responseData;
isCategoricalPredictor = false(1,num_p);

if T==0
    
else
    isCategoricalPredictor(T)=true;
end


% 训练分类器
% 以下代码指定所有分类器选项并训练分类器。
classificationKNN = fitcknn(...
    predictors, ...
    response, ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', NeiN, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [1:num_T]);

% 使用预测函数创建结果结构体
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
knnPredictFcn = @(x) predict(classificationKNN, x);
trainedClassifier.predictFcn = @(x) knnPredictFcn(predictorExtractionFcn(x));

% 向结果结构体中添加字段
trainedClassifier.ClassificationKNN = classificationKNN;
trainedClassifier.About = '此结构体是从分类学习器 R2022a 导出的训练模型。';
trainedClassifier.HowToPredict = sprintf('要对新预测变量列矩阵 X 进行预测，请使用: \n yfit = c.predictFcn(X) \n将 ''c'' 替换为作为此结构体的变量的名称，例如 ''trainedModel''。\n \nX 必须包含正好 9 个列，因为此模型是使用 9 个预测变量进行训练的。\nX 必须仅包含与训练数据具有完全相同的顺序和格式的\n预测变量列。不要包含响应列或未导入 App 的任何列。\n \n有关详细信息，请参阅 <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appclassification_exportmodeltoworkspace'')">How to predict using an exported model</a>。');

% 提取预测变量和响应
% 以下代码将数据处理为合适的形状以训练模型。
%
% 将输入转换为表
inputTable = array2table(trainingData, 'VariableNames', Var_name);

predictorNames = Var_name;
predictors = inputTable(:, predictorNames);
response = responseData;
isCategoricalPredictor =isCategoricalPredictor;

% 执行交叉验证
rng('default')      % 为了重现性，固定交叉验证数据集
partitionedModel = crossval(trainedClassifier.ClassificationKNN, 'KFold', K);

% 计算验证预测
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% 计算验证准确度
validationAccuracy = kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
