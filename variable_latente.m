function y = variable_latente(X, mu, Sigma, alpha)
% X : matrice d'échantillons (n x p)
% mu : moyennes des gaussiennes (matrice de taille K x p)
% Sigma : matrices de covariance des gaussiennes (matrice de taille p x p x K)
% alpha : proportions des gaussiennes (vecteur de taille K)
% y : vecteur de variable latente (vecteur de taille n)

K = length(alpha); % nombre de gaussiennes
[n, p] = size(X); % dimensions de la matrice d'échantillons

% Calcul des probabilités d'appartenance à chaque gaussienne pour chaque échantillon
P = zeros(n, K);
for k = 1:K
    P(:, k) = alpha(k) * mvnpdf(X, mu(k, :), squeeze(Sigma(:, :, k)));
end

% Détermination de la variable latente pour chaque échantillon
[~, y] = max(P,[],2);
end