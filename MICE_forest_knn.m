function[knn_mice_forest]      = MICE_forest_knn(X, y)
max_iterations = 10 ;
k =3 ;
X_imputed_mice_forest = MICE_random_forest(X, max_iterations);
% Estimate beta using maximum likelihood
knn_mice_forest  = fitcknn(X_imputed_mice_forest, y, 'NumNeighbors', k);
