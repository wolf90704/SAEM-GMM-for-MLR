function X_imputed_mice_rf = MICE_random_forest(X, max_iterations)
    p = size(X, 2);
    n = size(X, 1);
    X_imputed_rf = X;  % Créez une copie de X_obs

    X_imputed_mice_rf = X_imputed_rf;

    for iteration = 1:max_iterations
        for variable = 1:p
            % Supprimez les valeurs imputées dans la colonne sélectionnée
            X_imputed_mice_rf(isnan(X(:, variable)), variable) = NaN;

            % Les autres colonnes
            observed_train = X_imputed_mice_rf(~isnan(X(:, variable)), 1:p ~= variable);
            observed_predict = X_imputed_mice_rf(isnan(X(:, variable)), 1:p ~= variable);

            missing = X_imputed_mice_rf(~isnan(X(:, variable)), variable);

            if ~isempty(observed_train) && ~isempty(observed_predict)
                % Construisez un modèle Random Forest d'imputation
                rf_model = TreeBagger(50, observed_train, missing, 'Method', 'regression');

                % Prédisez les valeurs manquantes en utilisant le modèle Random Forest
                imputedValues = predict(rf_model, observed_predict);

                % Mettez à jour l'ensemble de données imputé
                X_imputed_mice_rf(isnan(X(:, variable)), variable) = imputedValues;
            end
        end
    end
end