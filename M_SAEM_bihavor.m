function [ beta_estimated_SAEM_iter,w_SAEM,mu_SAEM,Sigma_SAEM,stats_saem] = M_SAEM_bihavor(X,y,tau_SA_control,num_components)
p = size(X,2);
n = size(X,1);
s_1_1_j = zeros(1,num_components);
s_1_2_j = zeros(num_components, p);  % p est le nombre de caractéristiques
s_1_3_j = zeros(p, p,num_components);
%% MI : Impute by mean imputation
X_imputed = X;
for j = 1:p
    col_mean = nanmean(X(:, j));
    ligne_nanIndices = isnan(X(:, j));
    X_imputed(ligne_nanIndices, j) = col_mean;
end
 [B1_mean, ~, stats_saem] = mnrfit(X_imputed, y,'model','nominal');
    beta_estimated_meanImp = B1_mean(:, 1:4);
   dimensions_beta = size(beta_estimated_meanImp);
%% M_SAEM

 [mu_SAEM, Sigma_SAEM, w_SAEM, idx] = kmeans_init_gmm_idx(X_imputed, num_components);

Iter_EM   = 400;
mu_SAEM = mu_SAEM' ;
beta_estimated_SAEM_iter = zeros([dimensions_beta, Iter_EM]);

beta_estimated_SAEM= beta_estimated_meanImp ;%kron(beta_SAEM,ones(1,n));


SA_step=1; %for Metropolis_Hastings_sampling
k1=50;

nb_Sample_MHS = 1 ;




for iter=1:Iter_EM
    iter
  % [x_s,acceptance_rate]=Metropolis_multiClass_GMM05(X, mu_SAEM, Sigma_SAEM,w_SAEM, beta_estimated_SAEM, nb_Sample_MHS,  n, y);
    [x_s,acceptance_rate] = Metropolis_5_classes(X, mu_SAEM, Sigma_SAEM,w_SAEM, beta_estimated_SAEM, nb_Sample_MHS,  n, y);
      fprintf('Taux d''acceptation : %.2f%%\n', acceptance_rate * 100);
     
  %% E step   
 %gam = prb_app_obs(X, mu_SAEM, Sigma_SAEM,w_SAEM);
 gam = prb_app(x_s, mu_SAEM, Sigma_SAEM,w_SAEM);

    %% Approximation Step
     if (iter>k1)
         SA_step=(iter-k1)^(-tau_SA_control);
     end
     
     
     for j = 1 : num_components
        gam_x = 0;
        gam_xx = 0 ; 
         
     % Calculate the summation term
   for i = 1 : n 
       
     
     gam_x  =  gam_x + gam(i, j) *  x_s (i,:);
     gam_xx = gam_xx + gam(i, j) * (x_s (i,:)' * x_s (i,:));
   end
s_k_1_j(j) = s_1_1_j(j) + SA_step * (sum(gam(:, j)) - s_1_1_j(j));

s_k_2_j(j,:) = s_1_2_j(j,:) + SA_step * (gam_x - s_1_2_j(j,:));

s_k_3_j(:,:,j) = s_1_3_j(:,:,j) + SA_step * (gam_xx - s_1_3_j(:,:,j));    % Update s for the next iteration
    s_1_1_j(j) = s_k_1_j(j);
    s_1_2_j(j,:) = s_k_2_j(j,:);
    s_1_3_j(:,:,j) = s_k_3_j(:,:,j);
    
    
    
    
     
     
     
     
     
     
     
    %% Maximization Step
    disp(mu_SAEM)
    %disp(Sigma_SAEM)

     w_SAEM (j)          = s_1_1_j(j) / n;
     mu_SAEM(j,:)        = s_1_2_j(j,:) / s_1_1_j(j);
  Sigma_SAEM (:,:,j)     = s_1_3_j(:,:,j)  / s_1_1_j(j) - (s_1_2_j(j,:) / s_1_1_j(j))' * (s_1_2_j(j,:) / s_1_1_j(j));
     end
     
     [B1_saem, ~, stats_saem] = mnrfit(x_s,y,'model','nominal');
     
  
     beta_estimated_SAEM_new  = B1_saem;
     beta_estimated_SAEM  =(1-SA_step)*beta_estimated_SAEM  +  SA_step*beta_estimated_SAEM_new;
   
  beta_estimated_SAEM_iter(:,:,iter) = beta_estimated_SAEM; 
    
     if ~isnan(X)
    % Aucune donnée manquante, sortir immédiatement ou effectuer d'autres actions
       disp('Aucune donnée manquante ');
       break; % Quittez la fonction ou la boucle ici
     end
    
    
    
    
end

end