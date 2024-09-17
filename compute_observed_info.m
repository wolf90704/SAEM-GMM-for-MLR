function I_S = compute_observed_info(X_total, beta_estimated, Y, K)
    % X_total: 3D matrix, size (N, p, S) with observed and missing data samples
    % beta_estimated: estimated parameters for each class, size (K, p+1)
    % Y: observed classes, size (N, 1)
    % K: number of classes
    % I_S: observed information matrix for each beta, size (K, p+1, p+1)

    [N, p, S] = size(X_total); % Extract dimensions
    I_S = zeros(p+1, p+1, K); % Information matrix for each class

    for k = 1:K
        Delta_k = zeros(p+1, 1); % Accumulator for average gradient
        D_k = zeros(p+1, p+1);   % Accumulator for Hessian
        G_k = zeros(p+1, p+1);   % Accumulator for the second term of the observed info
        
        for i = 1:N
            for s = 1:S
                % Use the current sample from the 3D matrix
                X_complete = X_total(i, :, s);
                
                % Create r_i_s by adding the bias (constant 1)
                r_i_s = [1, X_complete];
                
                % Calculate probabilities for each class
                exp_beta_r = exp(beta_estimated * r_i_s'); % Compute e^(beta' * r_i_s)
                denom = 1 + sum(exp_beta_r); % Denominator
                P_y = exp_beta_r / denom; % Probabilities for all classes
                
                % Calculate gradient for class k
                gradient = r_i_s' * (double(Y(i) == k) - P_y(k));
                
                % Calculate Hessian for each pair of classes (k, l)
                for l = 1:K
                    if k == l
                        delta_kl = 1;
                    else
                        delta_kl = 0;
                    end
                    
                    % Compute Hessian for class k and l
                    hessian_kl = - (delta_kl * (exp_beta_r(k) / denom) - ...
                                     (exp_beta_r(k) * exp_beta_r(l)) / denom^2) * (r_i_s' * r_i_s);
                    
                    % Update Hessian
                    D_k = ((s - 1) * D_k + hessian_kl) / s;
                end
                
                % Update gradient
                Delta_k = ((s - 1) * Delta_k + gradient) / s;
                G_k = ((s - 1) * G_k + gradient * gradient') / s;
            end
            
            % Update the information matrix for observation i
            I_S(:, :, k) = I_S(:, :, k) + (D_k + G_k - Delta_k * Delta_k');
        end
        
    end
end
