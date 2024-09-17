function [X_imputed] = MeanImputation_test(X)
p = size(X,2);
n = size(X,1);

% Impute missing values with mean
X_imputed = X;
for j = 1:p
    col_mean = nanmean(X(:,j));
    ligne_nanIndices = isnan(X(:, j));
    X_imputed(ligne_nanIndices, j) = col_mean;
end


end
