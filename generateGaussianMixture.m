function samples = generateGaussianMixture(numSamples, means, covariances, weights)
    % Vérification des dimensions des entrées
    numComponents = numel(weights);
   
    assert(numComponents == size(means, 1) && numComponents == size(covariances, 3), ...
        'Le nombre de moyennes, de matrices de covariance et de poids doivent être identiques.');

    % Génération des échantillons
    dim = size(means, 2);
    samples = zeros(numSamples, dim);
    for i = 1:numSamples
        % Sélection d'un composant selon les poids
        component = randsample(1:numComponents, 1, true, weights);

        % Génération d'un échantillon pour le composant sélectionné
        samples(i, :) = mvnrnd(means(component, :), covariances(:,:,component));
    end
end