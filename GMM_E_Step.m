function expectations = GMM_E_Step(X,theta)
%

[N dim] = size(X);
K = length(theta.tau);
   
densEst = GMM_densE_st(X,theta);
pointsMissing = find(sum(isnan(X),2)>0);


% Mettre à jour les probabilités a posteriori
for j = 1:K
	expectations.z(:,j) = theta.tau(j)*densEst(:,j)./sum(repmat(theta.tau,N,1).*densEst,2);
end


% calcule des esperances conditionels 
for j = 1:K
    expectations.x{j} = zeros(N, dim);  % Initialisation de la matrice expectations.x{j}

    for n = 1:length(pointsMissing)
        missingComponents = isnan(X(pointsMissing(n), :));  % Booléen indiquant les composants manquants de X pour le point n

        % Calcul de mat en utilisant les matrices de précision de theta.Prec
        mat                =     theta.Sigma(missingComponents,~ missingComponents, j) *  inv (theta.Sigma(~missingComponents, ~missingComponents, j));

        % Calcul des valeurs pour expectations.x{j} en utilisant les éléments des données manquants, theta.mu et mat
       
        expectations.x{j}(pointsMissing(n), missingComponents)                     =  theta.mu(missingComponents, j)    +     mat * (X(pointsMissing(n), ~missingComponents)' - theta.mu(~missingComponents, j));
  
        expectations.xx{j}(missingComponents, missingComponents,pointsMissing(n))  =  theta.Sigma(missingComponents, missingComponents, j)  -    mat   *   theta.Sigma( ~missingComponents, missingComponents, j);

        
        
    end
    
end



end % end function
