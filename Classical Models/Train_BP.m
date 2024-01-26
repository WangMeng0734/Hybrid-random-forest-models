function [trainedClassifier, validationAccuracy,validationPredictions,validationScores] = Train_BP(trainingData, responseData,K,T,LayN,LanmN)
%% BP神经网络：BPNN
%% 采用单隐含层，激活函数使用常用的ReLu激活函数
%% 可优化超参数：隐含层数量(1~300)、正则化强度(1e-08~100)
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
classificationNeuralNetwork = fitcnet(...
    predictors, ...
    response, ...
    'LayerSizes', LayN, ...
    'Activations', 'relu', ...
    'Lambda', LanmN, ...
    'IterationLimit', 1000, ...
    'Standardize', true, ...
    'ClassNames', [1:num_T]);

% 使用预测函数创建结果结构体
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
neuralNetworkPredictFcn = @(x) predict(classificationNeuralNetwork, x);
trainedClassifier.predictFcn = @(x) neuralNetworkPredictFcn(predictorExtractionFcn(x));

% 向结果结构体中添加字段
trainedClassifier.ClassificationNeuralNetwork = classificationNeuralNetwork;
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
isCategoricalPredictor = isCategoricalPredictor;

% 执行交叉验证
rng('default')      % 为了重现性，固定交叉验证数据集
partitionedModel = crossval(trainedClassifier.ClassificationNeuralNetwork, 'KFold', K);

% 计算验证预测
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

% 计算验证准确度
validationAccuracy = kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
