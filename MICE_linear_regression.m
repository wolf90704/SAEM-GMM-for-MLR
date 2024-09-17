function X_imputed_mice = MICE_linear_regression(X, max_iterations)
    p = size(X, 2);
    n = size(X, 1);
    mean_values = nanmean(X);  % Calculer la valeur moyenne pour les éléments non NaN dans X_obs
    X_imputed = X;  % Créer une copie de X_obs

    % Remplacer les valeurs NaN par la valeur moyenne
    for l = 1 : p
        X_imputed(isnan(X(:, l)), l) = mean_values(l);
    end

    X_imputed_mice = X_imputed;

    for iteration = 1:max_iterations
        for variable = 1:p
            % Supprimer les valeurs imputées dans la colonne sélectionnée
            X_imputed_mice(isnan(X(:, variable)), variable) = NaN;

            % Les autres colonnes
            observed_train = X_imputed_mice(~isnan(X(:, variable)), 1:p ~= variable);
            observed_predict = X_imputed_mice(isnan(X(:, variable)), 1:p ~= variable);

            missing = X_imputed_mice(~isnan(X(:, variable)), variable);

            % Créer un modèle d'imputation
            imputationModel = fitlm(observed_train, missing);

            % Estimer les valeurs manquantes en utilisant le modèle d'imputation
            imputedValues = predict(imputationModel, observed_predict);

            % Mettre à jour l'ensemble de données imputé
            X_imputed_mice(isnan(X(:, variable)), variable) = imputedValues;
        end
    end
end