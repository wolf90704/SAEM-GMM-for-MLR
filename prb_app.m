function P = prb_app(X, mu, Sigma, alpha)
    K = length(alpha); % nombre de gaussiennes
    [n, p] = size(X); % dimensions de la matrice d'échantillons

    % Calcul des probabilités d'appartenance à chaque gaussienne pour chaque échantillon
    P = zeros(n, K);
    
    for k = 1:K
        P(:, k) = alpha(k) * mvnpdf(X, mu(k, :), squeeze(Sigma(:, :, k)));
    end
    
    % Normalisation des probabilités d'appartenance pour chaque échantillon
    P = P ./ sum(P, 2);
end