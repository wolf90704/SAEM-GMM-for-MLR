function [theta logLike] = GMM_M_Step(X, expectations, thetaOld)
    [N, dim] = size(X);
    K = size(expectations.z, 2);
    N_k = sum(expectations.z);

    % Mise à jour des paramètres
    Xtmp = X;
    for j = 1:K
        % Gestion des données manquantes ; remplacement des parties
        % manquantes de x par leur attente (chaque valeur manquente sera remplacer par son esperance cond)
        if sum(isnan(X(:))) > 0
            Xtmp(isnan(X)) = expectations.x{j}(isnan(X));
        end
        
        % Mise à jour de theta.mu
        theta.mu(:, j) = 1/N_k(j) * sum(repmat(expectations.z(:, j), 1, dim) .* Xtmp);
        
        % Mise à jour de theta.Sigma
        theta.Sigma(:, :, j) = 1/N_k(j) * (repmat(expectations.z(:, j), 1, dim) .* (Xtmp - repmat(theta.mu(:, j)', N, 1)))' * (Xtmp - repmat(theta.mu(:, j)', N, 1));
        
        % Mise à jour de theta.tau
        theta.tau(j) = N_k(j) / N;
    end

    % Ajout des matrices A_{nk} = \Lambda_k^{{mm}^{-1}} à Sigma(terme de correction matrice de cov cond pour chaque echantillon contient une v manq) 
    if sum(isnan(X(:))) > 0
        pointsMissing = find(sum(isnan(X), 2) > 0);

        for j = 1:K
            for n = 1:length(pointsMissing)
                componentsMissing = isnan(X(pointsMissing(n), :));
                theta.Sigma(componentsMissing, componentsMissing, j) = theta.Sigma(componentsMissing, componentsMissing, j) + 1/N_k(j) * expectations.z(pointsMissing(n), j) * (expectations.xx{j}(componentsMissing, componentsMissing,pointsMissing(n)));
            end
        end
    end

    densEst = GMM_densE_st(X, theta);
    logLike = sum(log(sum(repmat(theta.tau, N, 1) .* densEst, 2)));
end