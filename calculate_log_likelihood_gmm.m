function log_likelihood = calculate_log_likelihood_gmm(X, num_components, gmm)
    log_likelihood = 0;
    for i = 1:num_components
        mu = gmm.mu(i, :);
        Sigma = gmm.Sigma(:, :, i);
        tau = gmm.tau(i);
        
        % Calcul de la log-vraisemblance pour chaque composante
        log_likelihood = log_likelihood + tau * mvnpdf(X, mu, Sigma);
    end
    log_likelihood = sum(log(log_likelihood));
end