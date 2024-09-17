function [betas, stats, y_prediction] = CC(X_train, y_train, X_test)
    % Supprimer les lignes contenant des NaN
    nan_rows_train = any(isnan(X_train), 2);
    nan_rows_test = any(isnan(X_test), 2);
    
    X_train = X_train(~nan_rows_train, :);
    y_train = y_train(~nan_rows_train);
    X_test = X_test(~nan_rows_test, :);
   % y_test =  y_test(~nan_rows_test);
[betas, ~, stats] = mnrfit(X_train, y_train,'model','nominal');
    

    % Prédiction sur les données de test
    y_prediction = mnrval(betas, X_test);

    % Convertir les prédictions en classes
    [~, y_prediction] = max(y_prediction, [], 2);
end