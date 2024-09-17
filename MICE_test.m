function [X_imputed_mice] = MICE_test(X)
max_iterations = 10 ;
X_imputed_mice = MICE_linear_regression(X, max_iterations);


