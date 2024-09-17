function [x_imputed_mlt_em] = Multiple_EM_test(x_measured)
%% MI : Impute by mean imputation

X=x_measured;
X_imputed = X;
p = size(X,2);
n = size(X,1);
for j = 1:p
    col_mean = nanmean(X(:, j));
    ligne_nanIndices = isnan(X(:, j));
    X_imputed(ligne_nanIndices, j) = col_mean;
end

%% multiple-tirage Imputation EM
x_imputed_mlt_em =[];
y_imputed_mlt_em =[];
affiche = [];

mu_imputed_mlt_em = mean(X_imputed);
sigma_imputed_mlt_em = cov(X_imputed);

nb_Sample_MHS=10;
for TEM = 1:nb_Sample_MHS

    
    x_imputed_mlt_em =   generatedEM_new(x_measured,mu_imputed_mlt_em,sigma_imputed_mlt_em,n);

    mu_imputed_mlt_em = mean(x_imputed_mlt_em);
    sigma_imputed_mlt_em = cov(x_imputed_mlt_em);
    affiche = [affiche;mu_imputed_mlt_em];
end
 


end