function [x_s]=generatedEM_new(x_measured,mu_SAEM,Sigma_SAEM,n)

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
       
       % Vérifier si Sigma_cond_i est symétrique et définie positive
            if ~issymmetric(Sigma_cond_i) || any(eig(Sigma_cond_i) <= 0)
                % Si non, utiliser la fonction nearestSPD pour la rendre symétrique et définie positive
                Sigma_cond_i = nearestSPD(Sigma_cond_i);
            end  
       
       
       x_s(i,index_miss_i)=mvnrnd(mu_cond_i,Sigma_cond_i);%Initialisation
        

    end
    
end

