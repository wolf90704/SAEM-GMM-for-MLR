function [mdl_miss, stats_miss_f] = MICE_forest_new(X, y)
    % Paramètre pour MICE
    max_iterations = 10;

    % Imputation des données avec MICE et forêts aléatoires
    X_imputed_mice_forest = MICE_random_forest(X, max_iterations);

   [mdl_miss,~, stats_miss_f] = mnrfit(X_imputed_mice_forest, y);


    
end
