function [Y_new, votes, prediction_per_tree] = classRF_predict(p_train, model, extra_options)
% requires 2 arguments
% p_train: data matrix
% model: generated via classRF_train function
% extra_options.predict_all = predict_all if set will send all the prediction. 
%
% Returns
% Y_hat - prediction for the data
% votes - unnormalized weights for the model
% prediction_per_tree - per tree prediction. the returned object .
%           If predict.all = TRUE, then the individual component of the returned object is a character
%           matrix where each column contains the predicted class by a tree in the forest

    if nargin < 2
		error('need atleast 2 parameters,X matrix and model');
    end
    
    if exist('extra_options', 'var')
        if isfield(extra_options, 'predict_all') 
            predict_all = extra_options.predict_all;
        end
    end
    
    if ~exist('predict_all', 'var')
        predict_all = 0;
    end
            
	[Y_hat, prediction_per_tree, votes] = mexClassRF_predict(p_train', model.nrnodes, ...
        model.ntree, model.xbestsplit, model.classwt, model.cutoff, model.treemap, model.nodestatus, ...
        model.nodeclass, model.bestvar, model.ndbigtree, model.nclass, predict_all);
	
    % keyboard
    votes = votes';
    
    clear mexClassRF_predict
    
    Y_new = double(Y_hat);
    new_labels  = model.new_labels;
    orig_labels = model.orig_labels;
    
    for i = 1:length(orig_labels)
        Y_new(Y_hat == new_labels(i)) = Inf;
        Y_new(isinf(Y_new)) = orig_labels(i);
    end
    
end