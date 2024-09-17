
function P = prb_app_obs(X, mu, Sigma, alpha)

% GMM_densE_st estime les densités de probabilité des données X dans un modèle de mélange gaussien (GMM).

% Extraction des dimensions des données X et du nombre de composantes K du modèle
[N, dim] = size(X);
K = length(alpha);
% Identification des points de données entièrement observés
pointsObserved = sum(isnan(X), 2) == 0;

% Initialisation de la matrice des densités estimées
densE_st = zeros(N, K);

% Estimation des densités pour les points de données entièrement observés
for j = 1:K
    densE_st(pointsObserved, j) = mvnpdf(X(pointsObserved, :), mu(j, :),Sigma(:, :, j));
end

% Pour les points de données avec des valeurs manquantes, calculer la densité sur la distribution marginale des composantes observées
    pointsMissing = find(~pointsObserved);
for n = 1:length(pointsMissing)
    observedComponents = ~isnan(X(pointsMissing(n), :));
    for j = 1:K
        densE_st(pointsMissing(n), j) = mvnpdf(X(pointsMissing(n), observedComponents), mu(j,observedComponents),Sigma(observedComponents, observedComponents, j));
    end
end

% Mettre à jour les probabilités a posteriori
for j = 1:K
	P(:,j) = alpha(j)*densE_st(:,j)./sum(repmat(alpha,N,1).*densE_st,2);
end

end