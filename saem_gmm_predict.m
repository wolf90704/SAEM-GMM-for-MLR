function [y_predict] = saem_gmm_predict(x_measured, mu_SAEM, Sigma_SAEM, weights_SAEM, beta_SAEM, nb_Sample_mcmc, num_class)
    p = size(x_measured, 2);
    n = size(x_measured, 1);
    beta = beta_SAEM;
    d = 1:p;
    % xs_estt = zeros(n, p, nb_Sample_mcmc);
    xs_estt = x_measured;
    K = length(weights_SAEM);
    xs_est = x_measured;
    theta.tau = weights_SAEM;
    theta.mu = mu_SAEM';
    theta.Sigma = Sigma_SAEM;

    z = variable_latente_obs(x_measured, theta);
  
    y_predict = zeros(n, 1);

    for i = 1:n
        if any(isnan(x_measured(i, :)))
            index_miss_i = find(isnan(x_measured(i, :)));
            index_obs_i = d(setdiff(1:p, index_miss_i));
            x_obs_i = x_measured(i, index_obs_i);
            
            means = zeros(K, length(index_miss_i));
            covariances = zeros(length(index_miss_i), length(index_miss_i), K);
            
            for j = 1:K
                mu_miss_i = mu_SAEM(j, index_miss_i);
                mu_obs_i = mu_SAEM(j, index_obs_i);
                Sigma_obs_miss_i = Sigma_SAEM(index_obs_i, index_miss_i, j);
                Sigma_miss_obs_i = Sigma_obs_miss_i';
                Sigma_obs_obs_i = Sigma_SAEM(index_obs_i, index_obs_i, j);
                Sigma_miss_miss_i = Sigma_SAEM(index_miss_i, index_miss_i, j);
                
                mu_cond_i = mu_miss_i + (x_obs_i - mu_obs_i) * inv(Sigma_obs_obs_i) * Sigma_obs_miss_i;
                Sigma_cond_i = Sigma_miss_miss_i - Sigma_miss_obs_i * inv(Sigma_obs_obs_i) * Sigma_obs_miss_i;
                
                % Check if Sigma_cond_i is symmetric and positive definite
                if ~issymmetric(Sigma_cond_i) || any(eig(Sigma_cond_i) <= 0)
                    % If not, use the nearestSPD function to make it symmetric and positive definite
                    Sigma_cond_i = nearestSPD(Sigma_cond_i);
                end
                
                means(j, :)          = mu_cond_i;
                covariances(:, :, j) = Sigma_cond_i;
            end

            mu_cond    = means(z(i), :);
            Sigma_cond = covariances(:, :, z(i));
            
            for s = 1:nb_Sample_mcmc
                xs_estt(i, index_miss_i) = mvnrnd(mu_cond, Sigma_cond);
                xs_est (:,:,s)           = x_measured;               
                xs_est(i, index_miss_i,s) = xs_estt(i, index_miss_i); 
               
            end

            totalProbabilities = zeros(1, num_class);

            for s = 1:nb_Sample_mcmc
                % Calculate probabilities for this sample i
                probabilities_s = mnrval(beta, xs_est(i, :, s));

                % Add probabilities to the total sum
                totalProbabilities = totalProbabilities + probabilities_s;
                
            end

            [~, y_predict(i)] = max(totalProbabilities, [], 2);
        else
            % Prediction for the test data
            y_pred_probabilities = mnrval(beta, x_measured(i, :));

            % Predictions are in the form of probabilities for each class
            [~, y_predict(i)] = max(y_pred_probabilities, [], 2);
        end  
    end
end
