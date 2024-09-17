function [logistic_model_SAEM_GMM] = SAEM_Metropolis_multiclass_GMM05(X,y,num_components)
p = size(X,2);
n = size(X,1);
itermax_gmm = 100;
%% MI : Impute by mean imputation
X_imputed = X;
for j = 1:p
    col_mean = nanmean(X(:, j));
    ligne_nanIndices = isnan(X(:, j));
    X_imputed(ligne_nanIndices, j) = col_mean;
end
 [B1_mean, ~, stats_mean] = mnrfit(X_imputed, y,'model','nominal');
    beta_estimated_meanImp = B1_mean(:, 1:4);

%% SAEM



 [mu_SAEM, Sigma_SAEM, w_SAEM, idx] = kmeans_init_gmm_idx(X_imputed, num_components);
%[Sigma_SAEM, mu_SAEM, w_SAEM] = estimerParametresGMM(X_imputed, num_components);
disp(w_SAEM)
Iter_EM   = 100;
mu_SAEM   = mu_SAEM';


beta_estimated_SAEM= beta_estimated_meanImp ;%kron(beta_SAEM,ones(1,n));


SA_step=1; %for Metropolis_Hastings_sampling

nb_Sample_MHS=1;
k1=50;
tau_SA_control=1;


z = variable_latente(X_imputed, mu_SAEM, Sigma_SAEM, w_SAEM);

for iter=1:Iter_EM
    iter
    

     [x_s,acceptance_rate] = Metropolis_multiClass_GMM05(X,mu_SAEM,Sigma_SAEM,w_SAEM,beta_estimated_SAEM,X_imputed,nb_Sample_MHS,n,y,z);
     
      fprintf('Taux d''acceptation : %.2f%%\n', acceptance_rate * 100);


    %% Approximation Step
     if (iter>k1)
         SA_step=(iter-k1)^(-tau_SA_control);
     end

    %% Maximization Step
    
    
     
     [B1_saem, ~, stats_saem] = mnrfit(x_s,y,'model','nominal');
     
     theta = struct('mu', mu_SAEM', 'Sigma', Sigma_SAEM, 'tau', w_SAEM);
     [~, thetaa] = GaussMixt(x_s, num_components, theta, itermax_gmm);
    
     w_SAEM_new          =  thetaa.tau ;
     mu_SAEM_new         =  thetaa.mu' ;
     Sigma_SAEM_new      =  thetaa.Sigma ;
     
     
     
  
     beta_estimated_SAEM_new  = B1_saem;
     beta_estimated_SAEM      =(1-SA_step)*beta_estimated_SAEM+SA_step*beta_estimated_SAEM_new;
   
    
    
    
   Sigma_SAEM  =(1-SA_step) *Sigma_SAEM + SA_step*Sigma_SAEM_new;
    mu_SAEM    =(1-SA_step) *mu_SAEM    +  SA_step*mu_SAEM_new;
    w_SAEM     = (1-SA_step)*w_SAEM     +  SA_step*w_SAEM_new;
     
    
    
    z          = variable_latente(x_s, mu_SAEM, Sigma_SAEM, w_SAEM);
    
    
     if ~isnan(X)
    % Aucune donnée manquante, sortir immédiatement ou effectuer d'autres actions
       disp('Aucune donnée manquante ');
       break; % Quittez la fonction ou la boucle ici
     end
    
    
    
    
end
logistic_model_SAEM_GMM = mnrfit(x_s, y', 'model', 'nominal');

