function [logistic_model_MICE,stats_mice] = MICE_new(X,y)
max_iterations = 10 ;
X_imputed_mice = MICE_linear_regression(X, max_iterations);
% Estimate beta using maximum likelihood
[logistic_model_MICE,~, stats_mice] = mnrfit(X_imputed_mice, y);


