function z = variable_latente_obs(X,theta)
% GMM_densE_st estime les densités de probabilité des données X dans un modèle de mélange gaussien (GMM).

% Extraction des dimensions des données X et du nombre de composantes K du modèle
[N, dim] = size(X);
K = length(theta.tau);
% Identification des points de données entièrement observés
pointsObserved = sum(isnan(X), 2) == 0;

% Initialisation de la matrice des densités estimées
densE_st = zeros(N, K);

% Estimation des densités pour les points de données entièrement observés
for j = 1:K
    densE_st(pointsObserved, j) = mvnpdf(X(pointsObserved, :), theta.mu(:, j)', theta.Sigma(:, :, j));
end

% Pour les points de données avec des valeurs manquantes, calculer la densité sur la distribution marginale des composantes observées
    pointsMissing = find(~pointsObserved);
for n = 1:length(pointsMissing)
    observedComponents = ~isnan(X(pointsMissing(n), :));
    for j = 1:K
        densE_st(pointsMissing(n), j) = mvnpdf(X(pointsMissing(n), observedComponents), theta.mu(observedComponents, j)', theta.Sigma(observedComponents, observedComponents, j));
    end
end

% Mettre à jour les probabilités a posteriori
for j = 1:K
	prob_app(:,j) = theta.tau(j)*densE_st(:,j)./sum(repmat(theta.tau,N,1).*densE_st,2);
end
% Détermination de la variable latente pour chaque échantillon
[~, z] = max(prob_app,[],2);
end







