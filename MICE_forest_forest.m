function[mice_ff]      = MICE_forest_forest(X, y)
max_iterations = 10 ;
numTrees  = 150;
X_imputed_mice_forest = MICE_random_forest(X, max_iterations);
% Estimate beta using maximum likelihood

mice_ff = TreeBagger(numTrees, X_imputed_mice_forest, y, 'Method', 'classification');

