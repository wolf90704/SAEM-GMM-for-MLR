function [X_imputed_mice_forest] = MICE_test_forest(X)
max_iterations = 10 ;
X_imputed_mice_forest = MICE_random_forest(X, max_iterations);

