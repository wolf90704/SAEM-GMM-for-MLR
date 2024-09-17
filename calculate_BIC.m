function bic = calculate_BIC(X, num_components, gmm)
    % X : Données d'entrée
    % num_components : Nombre de composantes du GMM
    % gmm : Structure contenant les paramètres du GMM (mu, Sigma, tau)

    n = size(X, 1);

    % Calcul de la log-vraisemblance
    log_likelihood = calculate_log_likelihood_gmm(X, num_components, gmm);

    % Nombre de paramètres dans le modèle GMM
    num_params = num_components * (size(gmm.mu, 2) + size(gmm.Sigma, 2)^2 + 1);

    % Calcul du BIC
    bic = -2 * log_likelihood + num_params * log(n);
end