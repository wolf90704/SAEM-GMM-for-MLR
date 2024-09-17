 function [logistic_model_mean_cond,stats_mean_cond] = mean_cond_new(x_measured,y)
 p = size(x_measured,2);
 n = size(x_measured,1);

%% Imputation mu_conditionnel
    x_imputed_mu_cond = x_measured;
     x_imputed_em = x_measured;
     
     mean_values = nanmean(x_measured);  % Compute mean value for non-NaN elements in X_obs
     X_imputed = x_measured;  % Create a copy of X_obs

% Replace NaN values with mean value
for l = 1 : p
    X_imputed(isnan(x_measured(:,l)),l) = mean_values(l);%replace the elemnts with nan by the correspending mean of each column
end
X_imputed_new = X_imputed;
mu = mean(X_imputed_new);
Sigma = cov(X_imputed_new);


    d=1:size(x_measured,2); 
    for i=1:n
        if  isnan(sum(x_measured(i,:)))
            
            index_miss_i=find(isnan(x_measured(i,:)));
            index_obs_i=d(setdiff(1:end,index_miss_i));
            x_obs_i=x_measured(i,index_obs_i);
            mu_miss_i=mu(index_miss_i);
            mu_obs_i=mu(index_obs_i);
            Sigma_obs_miss_i=Sigma(index_obs_i,index_miss_i);
            Sigma_miss_obs_i=Sigma_obs_miss_i';
            Sigma_obs_obs_i=Sigma(index_obs_i,index_obs_i);
            Sigma_miss_miss_i=Sigma(index_miss_i,index_miss_i);

           mu_cond_i=mu_miss_i+(x_obs_i-mu_obs_i)*inv(Sigma_obs_obs_i)*Sigma_obs_miss_i;
           Sigma_cond_i=Sigma_miss_miss_i-Sigma_miss_obs_i*inv(Sigma_obs_obs_i)*Sigma_obs_miss_i;
           
            % Vérifier si Sigma_cond_i est symétrique et définie positive
            if ~issymmetric(Sigma_cond_i) || any(eig(Sigma_cond_i) <= 0)
                % Si non, utiliser la fonction nearestSPD pour la rendre symétrique et définie positive
                Sigma_cond_i = nearestSPD(Sigma_cond_i);
            end
            
                               
           x_imputed_mu_cond(i,index_miss_i) = mu_cond_i;
           x_imputed_em(i,index_miss_i)=mvnrnd(mu_cond_i,Sigma_cond_i);
           
        end
    end
    [logistic_model_mean_cond,~, stats_mean_cond] = mnrfit(x_imputed_mu_cond, y);
    
    
 end