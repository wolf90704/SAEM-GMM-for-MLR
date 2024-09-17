function [xs_est,acceptance_rate]=Metropolis_multiClass_GMM05(x_measured, mu_SAEM, Sigma_SAEM,weights_SAEM, beta_SAEM, nb_Sample_mcmc,  n, y,z)
  
  
    d         = 1:size(x_measured, 2);
    nbSamples = 1 ;
    x_s_1     = x_measured;
    xs_estt   = x_measured;
    acceptance_count = 0;
    BURN      = 	20;
    K         = length (weights_SAEM);
    xs_est    = x_measured; % Initialise xs_est avec x_measured
    nb        = 0  ;

for i=1:n
    
   
   if any(isnan(x_measured(i,:)))
             nb =nb + 1 ;
            index_miss_i = find(isnan(x_measured(i,:)));
            index_obs_i  = d(setdiff(1:end, index_miss_i));
            x_obs_i      = x_measured(i, index_obs_i);
    
            means = zeros( K,length(index_miss_i));
            covariances = zeros(length(index_miss_i), length(index_miss_i), K);
        
         for j = 1 : K
                                   
            mu_miss_i               = mu_SAEM(j,index_miss_i);
            mu_obs_i                = mu_SAEM(j,index_obs_i);
            Sigma_obs_miss_i        = Sigma_SAEM(index_obs_i , index_miss_i,j);
            Sigma_miss_obs_i        = Sigma_obs_miss_i';
            Sigma_obs_obs_i         = Sigma_SAEM(index_obs_i, index_obs_i,j);
            Sigma_miss_miss_i       = Sigma_SAEM(index_miss_i, index_miss_i,j);
           

            mu_cond_i                  = mu_miss_i + (x_obs_i - mu_obs_i) * inv(Sigma_obs_obs_i) * Sigma_obs_miss_i;
            Sigma_cond_i               = Sigma_miss_miss_i - Sigma_miss_obs_i * inv(Sigma_obs_obs_i) * Sigma_obs_miss_i;
            
             % Vérifier si Sigma_cond_i est symétrique et définie positive
            if ~issymmetric(Sigma_cond_i) || any(eig(Sigma_cond_i) <= 0)
                % Si non, utiliser la fonction nearestSPD pour la rendre symétrique et définie positive
                Sigma_cond_i = nearestSPD(Sigma_cond_i);
            end                                    
            
            means (j,: )               =  mu_cond_i ;
            covariances (:,:,j)        = Sigma_cond_i;
           
         
         end 
     z = prb_app_obs(x_measured, mu_SAEM, Sigma_SAEM,weights_SAEM);
     mu_cond     = means(z(i),:);
     Sigma_cond  = covariances (:,:,z(i));
     
     xs_estt(i, index_miss_i,1) =    mvnrnd(mu_cond, Sigma_cond);
     g = @(x) mvnpdf(x, mu_cond, Sigma_cond);
     
    
       switch y(i)
          
             
        case 1
        
                    f=@(x)   ( (  exp(beta_SAEM(1,1)+x*beta_SAEM(index_miss_i+1,1)+ x_s_1(i,index_obs_i)*beta_SAEM(index_obs_i+1,1))...
                    /(1 + exp(beta_SAEM(1,1) + x * beta_SAEM(index_miss_i + 1,1) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,1)) + ...
           exp(beta_SAEM(1,2) + x * beta_SAEM(index_miss_i + 1,2) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,2)) + ...
           exp(beta_SAEM(1,3) + x * beta_SAEM(index_miss_i + 1,3) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,3)) + ...
           exp(beta_SAEM(1,4) + x * beta_SAEM(index_miss_i + 1,4) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,4)))*mvnpdf(x, mu_cond, Sigma_cond)));
            
        case 2
                               
                     f=@(x)   ( (  exp(beta_SAEM(1,2)+x*beta_SAEM(index_miss_i+1,2)+ x_s_1(i,index_obs_i)*beta_SAEM(index_obs_i+1,2))...
                    /(1 + exp(beta_SAEM(1,1) + x * beta_SAEM(index_miss_i + 1,1) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,1)) + ...
                     exp(beta_SAEM(1,2) + x * beta_SAEM(index_miss_i + 1,2) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,2)) + ...
                     exp(beta_SAEM(1,3) + x * beta_SAEM(index_miss_i + 1,3) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,3)) + ...
                     exp(beta_SAEM(1,4) + x * beta_SAEM(index_miss_i + 1,4) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,4)))*mvnpdf(x, mu_cond, Sigma_cond)));           

       case 3
                
                    f=@(x)   ( (  exp(beta_SAEM(1,3)+x*beta_SAEM(index_miss_i+1,3)+ x_s_1(i,index_obs_i)*beta_SAEM(index_obs_i+1,3))...
                    /(1 + exp(beta_SAEM(1,1) + x * beta_SAEM(index_miss_i + 1,1) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,1)) + ...
             exp(beta_SAEM(1,2) + x * beta_SAEM(index_miss_i + 1,2) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,2)) + ...
             exp(beta_SAEM(1,3) + x * beta_SAEM(index_miss_i + 1,3) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,3)) + ...
             exp(beta_SAEM(1,4) + x * beta_SAEM(index_miss_i + 1,4) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,4)))*mvnpdf(x, mu_cond, Sigma_cond)));       
        case 4        
                    f=@(x)   ( (  exp(beta_SAEM(1,4)+x*beta_SAEM(index_miss_i+1,4)+ x_s_1(i,index_obs_i)*beta_SAEM(index_obs_i+1,4))...
                           /(1 + exp(beta_SAEM(1,1) + x * beta_SAEM(index_miss_i + 1,1) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,1)) + ...
             exp(beta_SAEM(1,2) + x * beta_SAEM(index_miss_i + 1,2) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,2)) + ...
             exp(beta_SAEM(1,3) + x * beta_SAEM(index_miss_i + 1,3) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,3)) + ...
             exp(beta_SAEM(1,4) + x * beta_SAEM(index_miss_i + 1,4) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,4)))*mvnpdf(x, mu_cond, Sigma_cond)));           
                
        case 5        
             f=@(x)   ((( 1 )/(1 + exp(beta_SAEM(1,1) + x * beta_SAEM(index_miss_i + 1,1) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,1)) + ...
                              exp(beta_SAEM(1,2) + x * beta_SAEM(index_miss_i + 1,2) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,2)) + ...
                              exp(beta_SAEM(1,3) + x * beta_SAEM(index_miss_i + 1,3) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,3)) + ...
                              exp(beta_SAEM(1,4) + x * beta_SAEM(index_miss_i + 1,4) + x_s_1(i,index_obs_i) * beta_SAEM(index_obs_i + 1,4)))*mvnpdf(x, mu_cond, Sigma_cond)));                    
             
       
          
        end
           
        t = 1;
            for s = 1: BURN + nb_Sample_mcmc
                x_s_1(i, index_miss_i, s+1) = mvnrnd(mu_cond, Sigma_cond);
                
                rap_1 = f(x_s_1(i, index_miss_i, s+1)) / g(x_s_1(i, index_miss_i, s+1));
                rap_2 = f(xs_estt(i, index_miss_i)) / g(xs_estt(i, index_miss_i));
                w = rap_1 / rap_2;
               if rand < w
                    xs_estt(i, index_miss_i) = x_s_1(i, index_miss_i, s+1);
                    
                    acceptance_count = acceptance_count + 1;
                end
          
            if s > BURN
            xs_est(i, index_miss_i,t) = xs_estt(i, index_miss_i); 
            t = t + 1 ;
        end
    end
  
    end
    end
    acceptance_rate = acceptance_count / (nb * (nb_Sample_mcmc + BURN));
end

    
    


