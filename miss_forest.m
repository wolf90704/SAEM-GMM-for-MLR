% Génération d'un jeu de données complet
N = 1000;  % Nombre d'échantillons
P = 5;    % Nombre de caractéristiques

data_complete = randn(N, P);  % Génération de données aléatoires

% Taux de données manquantes MAR (dépendant d'une autre variable)
missing_rate = 0.3;  % Taux de données manquantes

% Variable dépendante qui influence les données manquantes
dependent_variable = randn(N, 1);

% Créez un vecteur de booléens indiquant si les données sont manquantes ou non
is_missing = rand(N, P) < missing_rate & dependent_variable > 0;

% Appliquez le masque de données manquantes aux données complètes
data_incomplete = data_complete;
data_incomplete(is_missing) = NaN;



% Génération d'un jeu de données de test
rng(1);  % Pour obtenir des résultats reproductibles
N = 1000;  % Nombre d'échantillons
P = 5;  % Nombre de caractéristiques

% Taux de données manquantes à tester
missing_rates = 0.1:0.1:0.9;  % Par exemple, de 10 % à 90 %

% Initialisation pour stocker les MAEs
mae_rf = zeros(size(missing_rates));
mae_reg = zeros(size(missing_rates));

for i = 1:length(missing_rates)
    % Créez un ensemble de données de test avec des valeurs manquantes
    X = 3*randn(N, P);
    T = X;
    nanIndices = rand(N, P) < missing_rates(i);
    X(nanIndices) = NaN;
       
    % Imputation des valeurs manquantes en utilisant MICE avec Random Forest
    max_iterations = 10;  % Nombre maximal d'itérations
    k              = 10;
    X_imputed_mice_rf    = MICE_random_forest(X, max_iterations);
    X_imputed_mice       = MICE_linear_regression(X, max_iterations);
    X_imputed_knn        = knn_impute(X, k);
    % Calcul de la MAE entre les valeurs imputées et les valeurs réelles
    mae_rf(i)  = nanmean(nanmean(abs(X_imputed_mice_rf - T)));
    mae_reg(i) = nanmean(nanmean(abs(X_imputed_mice    - T)));
    mae_knn(i) = nanmean(nanmean(abs(X_imputed_knn     - T)));
    
    
end

% Tracé de la courbe MAE en fonction du taux de données manquantes
plot(missing_rates, mae_rf, 'bo-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Random Forest');
hold on;
plot(missing_rates, mae_reg, 'rs-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'Régression Linéaire');
hold on
plot(missing_rates, mae_knn, 'rs-', 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', 'KNN','Color', 'yellow');

xlabel('Taux de Données Manquantes');
ylabel('Mean Absolute Error (MAE)');
legend('Location', 'Best');
grid on;
title('MAE en fonction du Taux de Données Manquantes');


