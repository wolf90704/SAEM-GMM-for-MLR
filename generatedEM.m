function [x_s]=Slice_Sampling_SEM(x_measured,mu_SAEM,Sigma_SAEM,beta_SAEM,nb_Sample,Mask,n,y,x_mean_imputation,x,mu,Sigma)

d=1:size(x_measured,2); 
x_s=x_measured;

for i=1:n
        
    if  isnan(sum(x_measured(i,:)))
        
        
        
        index_miss_i=find(isnan(x_measured(i,:)));
        index_obs_i=d(setdiff(1:end,index_miss_i));
        x_obs_i=x_measured(i,index_obs_i);
        mu_miss_i=mu_SAEM(index_miss_i);
        mu_obs_i=mu_SAEM(index_obs_i);
        Sigma_obs_miss_i=Sigma_SAEM(index_obs_i,index_miss_i);
        Sigma_miss_obs_i=Sigma_obs_miss_i';
        Sigma_obs_obs_i=Sigma_SAEM(index_obs_i,index_obs_i);
        Sigma_miss_miss_i=Sigma_SAEM(index_miss_i,index_miss_i);       
        
        

       mu_cond_i=mu_miss_i+(x_obs_i-mu_obs_i)*inv(Sigma_obs_obs_i)*Sigma_obs_miss_i;
       Sigma_cond_i=Sigma_miss_miss_i-Sigma_miss_obs_i*inv(Sigma_obs_obs_i)*Sigma_obs_miss_i;
       
      % Force la symétrie de Sigma_cond_i en utilisant la décomposition SVD
       Sigma_cond_i = nearestSPD(Sigma_cond_i);
         
       x_s(i,index_miss_i)=mvnrnd(mu_cond_i,Sigma_cond_i);%Initialisation
        

    end
    
end

