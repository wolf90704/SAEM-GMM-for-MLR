function [ beta_estimated_SAEM,w_SAEM,mu_SAEM,Sigma_SAEM,stats_saem,log_likelihood] = M_SAEM(X,y,num_components)
p = size(X,2);
n = size(X,1);
s_1_1_j = zeros(1,num_components);
s_1_2_j = zeros(num_components, p);  % p is the number of features
s_1_3_j = zeros(p, p,num_components);
%% MI: Impute by mean imputation
X_imputed = X;
for j = 1:p
    col_mean = nanmean(X(:, j));
    ligne_nanIndices = isnan(X(:, j));
    X_imputed(ligne_nanIndices, j) = col_mean;
end
[B1_mean, ~, stats_saem] = mnrfit(X_imputed, y, 'model', 'nominal');
beta_estimated_meanImp = B1_mean;

%% M_SAEM

[mu_SAEM, Sigma_SAEM, w_SAEM, idx] = kmeans_init_gmm_idx(X_imputed, num_components);

Iter_EM = 150;
mu_SAEM = mu_SAEM';

beta_estimated_SAEM = beta_estimated_meanImp; % kron(beta_SAEM, ones(1, n));

SA_step = 1; % for Metropolis-Hastings sampling
k1 = 50;
tau_SA_control = 1;
nb_Sample_MHS = 1;

for iter = 1:Iter_EM
    iter
    
    if ~isnan(X)
        % No missing data, exit immediately or perform other actions
        disp('No missing data');
        break; % Exit the function or loop here
    end
    
    [x_s, acceptance_rate] = Metropolis_5_classes(X, mu_SAEM, Sigma_SAEM, w_SAEM, beta_estimated_SAEM, nb_Sample_MHS, n, y);
    fprintf('Acceptance rate: %.2f%%\n', acceptance_rate * 100);
    
    %% E step   
    % gam = prb_app_obs(X, mu_SAEM, Sigma_SAEM, w_SAEM);
    gam = prb_app(x_s, mu_SAEM, Sigma_SAEM, w_SAEM);

    %% Approximation Step
    if (iter > k1)
        SA_step = (iter - k1)^(-tau_SA_control);
    end
    
    for j = 1:num_components
        gam_x = 0;
        gam_xx = 0;
        
        % Calculate the summation term
        for i = 1:n
            gam_x = gam_x + gam(i, j) * x_s(i, :);
            gam_xx = gam_xx + gam(i, j) * (x_s(i, :)' * x_s(i, :));
        end
        s_k_1_j(j) = s_1_1_j(j) + SA_step * (sum(gam(:, j)) - s_1_1_j(j));
        s_k_2_j(j, :) = s_1_2_j(j, :) + SA_step * (gam_x - s_1_2_j(j, :));
        s_k_3_j(:, :, j) = s_1_3_j(:, :, j) + SA_step * (gam_xx - s_1_3_j(:, :, j)); 

        % Update s for the next iteration
        s_1_1_j(j) = s_k_1_j(j);
        s_1_2_j(j, :) = s_k_2_j(j, :);
        s_1_3_j(:, :, j) = s_k_3_j(:, :, j);
    end
    
    %% Maximization Step
    w_SAEM(j) = s_1_1_j(j) / n;
    mu_SAEM(j, :) = s_1_2_j(j, :) / s_1_1_j(j);
    Sigma_SAEM(:, :, j) = s_1_3_j(:, :, j) / s_1_1_j(j) - (s_1_2_j(j, :) / s_1_1_j(j))' * (s_1_2_j(j, :) / s_1_1_j(j));
    
    [B1_saem, ~, stats_saem] = mnrfit(x_s, y, 'model', 'nominal');
    
    beta_estimated_SAEM_new = B1_saem;
    beta_estimated_SAEM = (1 - SA_step) * beta_estimated_SAEM + SA_step * beta_estimated_SAEM_new;
    
    % Get predicted probabilities
    predicted_probs = mnrval(B1_saem, x_s);
    
    % Calculate log-likelihood
    log_likelihood = sum(log(predicted_probs(sub2ind(size(predicted_probs), 1:size(predicted_probs, 1), y'))));
end
end
