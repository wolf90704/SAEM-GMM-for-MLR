function centroids = kmeans_pp_init(data, k)
    % Fonction d'initialisation K-Means++
    % Entrées :
    %   - data : Les données d'entrée (n x m), où n est le nombre d'échantillons et m est le nombre de caractéristiques.
    %   - k : Le nombre de centroïdes à initialiser.
    % Sortie :
    %   - centroids : Les centroïdes initialisés (k x m).

    % Sélectionnez le premier centroïde au hasard parmi les données
    centroids = data(randperm(size(data, 1), 1), :);

    % Boucle pour sélectionner les centroïdes restants
    for i = 2:k
        % Calculez les distances minimales entre chaque point de données et les centroïdes actuels
        distances = pdist2(data, centroids);
        min_distances = min(distances, [], 2);

        % Sélectionnez le prochain centroïde en utilisant des probabilités pondérées
        weights = min_distances.^2;
        weights = weights / sum(weights);
        idx = randsample(size(data, 1), 1, true, weights);

        % Ajoutez le nouveau centroïde à la liste
        centroids = [centroids; data(idx, :)];
    end
end