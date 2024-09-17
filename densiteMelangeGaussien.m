function density = densiteMelangeGaussien(sample, means, covariances, weights)
    % sample : Échantillon dont on veut calculer la densité (1 x D)
    % means : Matrice contenant les moyennes des distributions gaussiennes (K x D)
    % covariances : Tableau de matrices contenant les covariances des distributions gaussiennes (D x D x K)
    % weights : Vecteur contenant les poids des distributions gaussiennes (1 x K)

    % Vérification des dimensions des paramètres
    if size(means, 1) ~= size(weights, 2) || size(means, 2) ~= size(sample, 2) || size(covariances, 1) ~= size(sample, 2) || size(covariances, 2) ~= size(sample, 2) || size(covariances, 3) ~= size(weights, 2)
        error('Dimensions incorrectes des paramètres.');
    end

    K = size(means, 1);  % Nombre de gaussiennes
    D = size(means, 2);  % Dimension des gaussiennes

    density = 0;
    for k = 1:K
        % Calcul de la densité pour chaque gaussienne du mélange
        density = density + weights(k) * mvnpdf(sample, means(k, :), covariances(:, :, k));
    end
end