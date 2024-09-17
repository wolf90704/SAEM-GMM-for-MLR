function [mu_init, Sigma_init, w_init, idx] = kmeans_pp_init_gmm(data, num_components)
    % Fonction d'initialisation K-Means++ pour un modèle de Mélange de Gaussiennes (GMM)
    % Entrées :
    %   - data : Les données d'entrée (n x m), où n est le nombre d'échantillons et m est le nombre de caractéristiques.
    %   - num_components : Le nombre de composantes du GMM.
    % Sorties :
    %   - mu_init : Vecteurs moyennes initialisés pour chaque composante (num_components x m).
    %   - Sigma_init : Matrices de covariance initialisées pour chaque composante (m x m x num_components).
    %   - w_init : Poids initialisés pour chaque composante (1 x num_components).
    %   - idx : Indices d'appartenance de chaque échantillon aux composantes (n x 1).

    % Initialisation des centroïdes avec K-Means++
    centroids = kmeans_pp_init(data, num_components);

    % Exécution de K-Means pour obtenir les indices d'appartenance
   [~, idx] = kmeans(data, num_components, 'Start', centroids, 'MaxIter', 1000);
    % Initialisation des paramètres du GMM
    mu_init = zeros(num_components, size(data, 2));
    Sigma_init = zeros(size(data, 2), size(data, 2), num_components);
    w_init = zeros(1, num_components);

    for i = 1:num_components
        component_data = data(idx == i, :);
        mu_init(i, :) = mean(component_data);
        Sigma_init(:, :, i) = cov(component_data);
        w_init(i) = sum(idx == i) / numel(idx);
    end
end