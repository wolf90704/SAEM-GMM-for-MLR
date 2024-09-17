function [expectations, theta, logLike] = EM_General(X, expectations, theta, handles, itermax)
    logLike = 0;

    for iter = 1:itermax
        % ÉTAPE E
        expectations = handles.estep(X, theta);
        
        % ÉTAPE M
        [theta, logLike(iter+1)] = handles.mstep(X, expectations, theta);
        
        % Vérification de la convergence
        
           %   fprintf('%d: %.4f\n', iter, logLike(iter+1));
       
         if abs((logLike(iter+1) - logLike(iter)) / logLike(iter+1)) < 10^-6
        %    fprintf('Convergence à literation %d.\n', iter);
          break
        end  
    end
end



