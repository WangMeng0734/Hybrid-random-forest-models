function objValue = getObjValue(parameter)

%%  Get training data from main function
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');

%%  Get optimal parameters
    ntree = round(parameter(1, 1));          % number of trees
    mtry  = round(parameter(1, 2));          % default is floor(sqrt(size(X,2)

%%  Data parameters
    num_size = length(t_train);

%%  cross validation procedure
    indices = crossvalind('Kfold', num_size, 5);

for i = 1 : 5
    
    % Get the index logical value of the i-th data
    valid_data = (indices == i);
    
    % Negate and obtain the index logical value of the i-th training data
    train_data = ~valid_data;
    
    % 1 test, 4 training
    pv_train = p_train(train_data, :);
    tv_train = t_train(train_data, :);
    
    pv_valid = p_train(valid_data, :);
    tv_valid = t_train(valid_data, :);
    
    % Modeling
    model = classRF_train(pv_train, tv_train, ntree, mtry);

    %  Model prediction
    [t_sim, ~] = classRF_predict(pv_valid, model);
    
    % fitness value
    accuracy(i) = sum((t_sim == tv_valid)) / length(tv_valid);
    
end

%% Taking the classification prediction error rate as the objective function value of optimization
    if size(accuracy, 1) == 0
        objValue = 1;
    else
        objValue = 1 - mean(accuracy);
    end

end